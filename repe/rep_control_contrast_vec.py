"""
Based on https://github.com/andyzoujm/representation-engineering/blob/main/repe/rep_control_contrast_vec.py
with some modifications.
"""
import warnings
import random
import numpy as np
from functools import partial
from typing import List, Optional, Tuple, Union, Callable
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralModel, MistralForCausalLM, MistralPreTrainedModel
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation import GreedySearchDecoderOnlyOutput, GenerateDecoderOnlyOutput
import torch
from torch import nn
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from examples.honesty.rep_manager_utils import RepManager
from repe import rep_utils

# === should work for Llama and Mistral ===
def contrast_greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    **model_kwargs
) -> Union[GreedySearchDecoderOnlyOutput, torch.LongTensor]:

    # ===== pop repe/contrast control args ====
    alpha = model_kwargs.pop('alpha', None)
    contrast_tokens = model_kwargs.pop('contrast_tokens', None)
    compute_contrast = model_kwargs.pop('compute_contrast', None)
    pos_input_ids = model_kwargs.pop('pos_input_ids', None)
    pos_attention_mask = model_kwargs.pop('pos_attention_mask', None)
    neg_input_ids = model_kwargs.pop('neg_input_ids', None)
    neg_attention_mask = model_kwargs.pop('neg_attention_mask', None)
    control_layer_ids = model_kwargs.pop('control_layer_ids', None)

    assert not compute_contrast or not model_kwargs.get('use_cache', False), "Contrast Greedy Search not yet support generate with use_cache, please set model.generate(**kwargs, use_cache=False)"

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pos_input_ids=pos_input_ids,
            pos_attention_mask=pos_attention_mask,
            neg_input_ids=neg_input_ids,
            neg_attention_mask=neg_attention_mask,
            contrast_tokens=contrast_tokens,
            compute_contrast=compute_contrast,
            alpha=alpha,
            control_layer_ids=control_layer_ids
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.attentions,)
                )

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        past_key_values = self._extract_past_from_model_output(outputs)[1]
        # model_kwargs = self._update_model_kwargs_for_generation(
        #     outputs, model_kwargs
        # )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return GreedySearchDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=past_key_values,
        )
    else:
        return input_ids


def forward_contrast_vector(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,

        # ==== repe coeff ====
        alpha: int | None = None,
        contrast_tokens: int | None = None,
        compute_contrast: bool | None = False,
        pos_input_ids: torch.LongTensor | None = None,
        pos_attention_mask: torch.LongTensor | None = None,
        pos_position_ids: torch.LongTensor | None = None,
        neg_input_ids: torch.LongTensor | None = None,
        neg_attention_mask:  torch.LongTensor | None = None,
        neg_position_ids: torch.LongTensor | None = None,
        control_layer_ids: list[int] | None = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Args:
            control_layer_ids: Indices of model layers to control
        """
        if use_cache:
            warnings.warn("use_cache is not supported in forward_contrast_vector()")
        if control_layer_ids is not None and len(control_layer_ids) > 0 and\
            (min(control_layer_ids) < 0 or max(control_layer_ids) >= self.config.num_hidden_layers):
            
            raise ValueError("All indices in control_layer_ids must lie in range "\
                f"[0, {self.config.num_hidden_layers}), but got control_layer_ids: {control_layer_ids}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if (isinstance(self, LlamaModel) and self._use_flash_attention_2) or (isinstance(self, MistralModel) and self.config._attn_implementation == "flash_attention_2"):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif ((isinstance(self, LlamaModel) and self._use_sdpa) or (isinstance(self, MistralModel) and self.config._attn_implementation == "sdpa")) and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )


        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        activations = None
        if compute_contrast:
            # ======== Compute repe =========    
            embeds_p = self.embed_tokens(pos_input_ids)
            embeds_n = self.embed_tokens(neg_input_ids)
            hidden_states_p, hidden_states_n = embeds_p, embeds_n

            pos_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                pos_attention_mask, embeds_p.shape[:2], embeds_p, past_key_values_length
            )

            neg_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                neg_attention_mask, embeds_n.shape[:2], embeds_n, past_key_values_length
            )

        delta_state_norm = 0
        for layer_id, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # If gradient checkpointing is enabled during training
                forward_function = partial(
                    self._gradient_checkpointing_func,
                    decoder_layer.__call__
                )
            else:
                # Otherwise, use the decoder layer directly
                forward_function = decoder_layer

            # When you need to call forward_function, provide the necessary arguments
            layer_outputs = forward_function(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache
            )

            hidden_states = layer_outputs[0]
            
            # 1. if layer in target layer, we recompute activations and add
            # 2. else will add previous computed activations
            if compute_contrast:
                # ======== Compute activations for target layers =========    
                with torch.no_grad():
                    hidden_states_p = forward_function(
                        hidden_states_p,
                        attention_mask=pos_attention_mask,
                        position_ids=pos_position_ids,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache
                    )[0].detach()

                    hidden_states_n = forward_function(
                        hidden_states_n,
                        attention_mask=neg_attention_mask,
                        position_ids=neg_position_ids,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache
                    )[0].detach()
                if layer_id in control_layer_ids:
                    activations = alpha * (hidden_states_p[:, contrast_tokens:] - hidden_states_n[:, contrast_tokens:])
                    #delta_state_norm += torch.linalg.norm(activations)
                    c_length = activations.shape[1]

                    hidden_states[:, -c_length:, :] += activations
                    hidden_states_p[:, -c_length:, :] += activations
                    hidden_states_n[:, -c_length:, :] += activations
                        
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        #delta_state_norm /= len(control_layer_ids)
        #print(f"Avg delta state: {delta_state_norm}")

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def _always_true(self, *args, **kwargs):
    return True
    

def new_mistral_forward(
    self: MistralForCausalLM,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    # ===== repe ====
    **new_forward_kwargs
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    Same as MistralForCausalLM.forward() from transformers.models.mistral.modeling_mistral,
    except for addition of repe_kwargs
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **new_forward_kwargs,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def new_prepare_inputs_for_generation(self, old_prepare_inputs_for_generation, *args, **kwargs):
    """
    Override MistralForCausalLM.prepare_inputs_for_generation()
    to pass new_forward_kwargs (see forward_contrast_vector() for rep kwargs) to forward.
    """
    old_kwargs = kwargs.copy()
    # old_prepare_inputs_for_generation requires past_key_values to be None if cache_position is None
    if kwargs.get("cache_position") is None:
        old_kwargs.pop("past_key_values", None)
    
    model_inputs = old_prepare_inputs_for_generation(*args, **old_kwargs)

    # Ignore cache_position
    model_inputs.pop("cache_position", None)

    return model_inputs

class ContrastVecMistralForCausalLM(MistralPreTrainedModel):
    """
    ContrastVecMistralForCausalLM from:
    https://github.com/andyzoujm/representation-engineering/blob/main/repe/rep_control_contrast_vec.py,
    modified to have the same interface as NonLinearControlMistralForCausalLM
    """
    def __init__(self, rep_manager: RepManager, control_strength_scale: tuple[float, float]):
        """
        Args:
            rep_manager
            control_strength_scale: In the form (min_control_strength, max_control_strength).
                Values used to scale control_strength in generate(). max_control_strength
                should be set to the control strength which maximally increases the concept
                without significantly impacting the quality of the response (i.e. producing
                a degenerate response). E.g. a min_control_strength of -0.2 results in
                generate(concept_strength=-0.5, ...) mapping to a concept_strength of -0.5*0.2.
        """
        super().__init__(rep_manager.model.config)
        self.model: MistralForCausalLM = rep_manager.model
        self.config = self.model.config
        self.rep_manager = rep_manager
        self.tokenizer = rep_manager.tokenizer
        self.original_gen_funcs = [
            self.model.model.forward,
            self.model.forward,
            self.model.prepare_inputs_for_generation,
            # self.model._sample,
            self.model._validate_model_kwargs
        ]
        self.control_strength_scale = control_strength_scale
        min_strength, max_strength = control_strength_scale
        if min_strength >= 0 or max_strength <= 0:
            raise ValueError("(min_strength, max_strength) must have min_strength < 0 "\
                f"and max_strength > 0, but got {control_strength_scale}")

    def override_generation_funcs_(self, **new_forward_kwargs):
        """
        Override functions in the underlying model
        
        Args:
            **new_forward_kwargs: Kwargs for new_mistral_forward(). Note kwargs are passed
                here since self.model._sample doesn't pass kwargs to self.model.forward
                (new_mistral_forward()).
        """
        # Override MistralModel
        self.model.model.forward = partial(forward_contrast_vector, self.model.model)
        # Override MistralForCausalLM
        self.model.forward = partial(new_mistral_forward, self.model, **new_forward_kwargs)
        self.model.prepare_inputs_for_generation = partial(new_prepare_inputs_for_generation,
            self.model, self.model.prepare_inputs_for_generation)
        # self.model._sample = partial(contrast_greedy_search, self.model)
        # turn off _validate_model_kwargs, TODO: implement _validate_model_kwargs
        self.model._validate_model_kwargs = partial(_always_true, self)

    def reset_generation_funcs_(self):
        """
        Undo the function overrides for the underlying model,
        so that it behaves as expected.
        """
        self.model.model.forward, \
            self.model.forward, \
            self.model.prepare_inputs_for_generation, \
            self.model._validate_model_kwargs = self.original_gen_funcs

            # self.model._sample, \

    def generate(self,
        prompt: list[dict],
        control_strength: float = 1.0,
        echo_prompt: bool = False,
        inference_prompt_to_str: Callable = rep_utils.apply_chat_template,
        train_rep_reader_kwargs: dict = {},
        base_model_forward_kwargs: dict = {},
        **_
    ):
        """
        Args:
            prompt: Huggingface chat template containing prompt
            echo_prompt: If False, remove the prompt from the output
            base_model_forward_kwargs: Kwargs used by the original forward method
                for the model (e.g. length_penalty).
        """
        ### Preprocessing
        # Scale concept strength
        if control_strength >= 0:
            control_strength *= self.control_strength_scale[1]
        else:
            control_strength *= -self.control_strength_scale[0]

        ## Process prompts
        # Prepend instructions to act with and without concept (e.g. honesty)
        prompts, labels = rep_utils.apply_concept_template(
            prompt, tokenizer=self.tokenizer, **train_rep_reader_kwargs["concept_template_kwargs"]
        )
            
        # Convert prompt to str
        prompt = inference_prompt_to_str(chat_template=prompt, tokenizer=self.tokenizer)
        pos_prompt = prompts[np.argmax(labels)]
        neg_prompt = prompts[np.argmin(labels)]
        inputs = [prompt, pos_prompt, neg_prompt]

        # Tokenize and set up args for model
        inputs = self.tokenizer(
            inputs,
            padding=True,
            add_special_tokens=False, # Prompts are already formatted with <s> tags
            return_tensors="pt"
        ).to(self.model.device)
        padded_prompt_len = len(inputs['input_ids'][0])

        new_forward_kwargs = {}
        for idx, prefix in enumerate(["", "pos_", "neg_"]):
            new_forward_kwargs.update({
                prefix + "input_ids": inputs['input_ids'][idx].unsqueeze(dim=0),
                prefix + "attention_mask": inputs['attention_mask'][idx].unsqueeze(dim=0)
            })

        new_forward_kwargs.update({
            "contrast_tokens": min(padded_prompt_len, train_rep_reader_kwargs["model_kwargs"]["max_new_tokens"]),
             # If True, ignore the representation and directly get the result of the base model
            "compute_contrast": True,
            "alpha": control_strength, # try 0.1+, maybe 0.1 for mistrals
            "control_layer_ids": list(np.array(self.rep_manager.hidden_layers) + self.config.num_hidden_layers)
        })
        _sample_kwargs = dict(
            input_ids=new_forward_kwargs.get("input_ids"),
            attention_mask=new_forward_kwargs.get("attention_mask")
        )

        prepare_input_kwargs = {"use_cache": False}
        for prefix in ("pos", "neg"):
            prepare_input_kwargs["attention_mask"] = new_forward_kwargs[f"{prefix}_attention_mask"]
            new_forward_kwargs[f"{prefix}_position_ids"] = self.model.prepare_inputs_for_generation(
                new_forward_kwargs[f"{prefix}_input_ids"], **prepare_input_kwargs)["position_ids"]

        self.override_generation_funcs_(**new_forward_kwargs)
        
        ### Run model
        output = self.model.generate(
            **_sample_kwargs,
            # **new_forward_kwargs,
            use_cache=False,
            pad_token_id=self.tokenizer.eos_token_id,
            **base_model_forward_kwargs
        )

        ### Postprocessing        
        # Remove prompt (including padding) so that the prompt isn't shown
        # again when displaying the output
        if not echo_prompt:
            if isinstance(output, torch.Tensor):
                output = output[:, padded_prompt_len:]
            else:
                output.sequences = output.sequences[:, padded_prompt_len:]
        
        self.reset_generation_funcs_()
        return {"input_length": padded_prompt_len, "output": output}

class NoRepMistralForCausalLM(MistralPreTrainedModel):
    """
    Class to run Mistral.generate() (without a representation method controlling
    the model) with an interface similar to ContrastVecMistralForCausalLM
    """
    def __init__(self, model, tokenizer):
        """
        Args:
            model
            tokenizer
        """
        super().__init__(model.config)
        self.model: MistralForCausalLM = model
        self.config = self.model.config
        self.tokenizer = tokenizer

    def generate(self,
        prompt: list[dict],
        echo_prompt: bool = False,
        train_rep_reader_kwargs: dict = {},
        base_model_forward_kwargs: dict = {},
        **_
    ):
        """
        Args:
            prompt: Huggingface chat template containing prompt
            echo_prompt: If False, remove the prompt from the output
        """

        ### Process prompts
        # Prepend instructions to act with and without concept (e.g. honesty)
        inputs = rep_utils.apply_concept_template(
            prompt, tokenizer=self.tokenizer, **train_rep_reader_kwargs["concept_template_kwargs"]
        )[0][0]

        # Tokenize and set up args for model
        inputs = self.tokenizer(
            inputs,
            padding=True,
            add_special_tokens=False, # Prompts are already formatted with <s> tags
            return_tensors="pt"
        ).to(self.model.device)
        padded_prompt_len = len(inputs['input_ids'][0])

        ### Run model
        output = self.model.generate(
            **inputs,
            use_cache=False,
            pad_token_id=self.tokenizer.eos_token_id,
            **base_model_forward_kwargs
        )

        ### Postprocessing        
        # Remove prompt (including padding) so that the prompt isn't shown
        # again when displaying the output
        if not echo_prompt:
            if isinstance(output, torch.Tensor):
                output = output[:, padded_prompt_len:]
            else:
                output.sequences = output.sequences[:, padded_prompt_len:]
        
        return {"input_length": padded_prompt_len, "output": output}

class UncontrolledModelForHELM(MistralPreTrainedModel):
    """
    Parent class for models to generate outputs compatible with CustomHuggingFaceServer
    """
    def __init__(self, model, tokenizer):
        """
        Args:
            model
            tokenizer
        """
        super().__init__(model.config)
        self.model: MistralForCausalLM = model
        self.config = self.model.config
        self.tokenizer = tokenizer

    def generate_helm_compatible_output_(self,
        prompt: str,
        echo_prompt: bool = False,
        **_
    ):
        """
        Helper function for converting prompt strings to model outputs
        when not using a model
        """
        
        inputs = self.tokenizer(
            prompt,
            padding=True,
            add_special_tokens=False, # Prompts are already formatted with <s> tags
            return_tensors="pt"
        )['input_ids']
        padded_prompt_len = inputs.shape[-1] - 1

        # Generate nothing to make output have the required format
        output = GenerateDecoderOnlyOutput(
            sequences=inputs,
            scores=(torch.zeros((1, self.model.vocab_size), ),)
        )
      
        # Remove prompt (including padding) so that the prompt isn't shown
        # again when displaying the output
        if not echo_prompt:
            output.sequences = output.sequences[:, padded_prompt_len:]
        
        return {"input_length": padded_prompt_len, "output": output}

class RandomMistralForCausalLM(UncontrolledModelForHELM):
    """
    Class to generate random (valid) multiple choice answers 
    with an interface compatible with ContrastVecMistralForCausalLM
    """
    def __init__(self, model, tokenizer):
        """
        Args:
            model
            tokenizer
        """
        super().__init__(model, tokenizer)

    def generate(self,
        prompt: list[dict],
        echo_prompt: bool = False,
        **_
    ):
        """
        Args:
            prompt: Huggingface chat template containing prompt
            echo_prompt: If False, remove the prompt from the output
        """
        # Set the answer in the prompt chat template to a valid random answer
        question = prompt[-2]['content']
        possible_answers = rep_utils.get_possible_answers(question)
        random_idx = random.randrange(len(possible_answers))
        random_answer = possible_answers[random_idx]
        prompt[-1]['content'] = f"{prompt[-1]['content']} {random_answer}"

        # Tokenize
        inputs = rep_utils.apply_chat_template(prompt, self.tokenizer)
        return self.generate_helm_compatible_output_(inputs, echo_prompt)
    
class NonLinearClassifierModel(UncontrolledModelForHELM):
    def __init__(self, rep_manager: RepManager):
        """
        Wrapper around rep_manager.classify_tokens to have
        a similar interface to NonLinearControlMistralForCausalLM for
        compatibility with CustomHuggingFaceServer for HELM.
        Appends a classification label to the prompt, predicted
        using the non-linear representation.
        """
        super().__init__(rep_manager.model, rep_manager.tokenizer)
        self.layers = rep_manager.hidden_layers
        self.rep_manager = rep_manager

    def generate(self,
        prompt: list[dict] | dict[str, list[dict]],
        echo_prompt: bool = False,
        train_rep_reader_kwargs: dict = {},
        softmax: bool = True,
        **_
    ) -> dict:
        """
        Classify (determine the concept score of) the prompt.
        
        If the prompt is a chat template, append the classification
        label to the prompt as the generated text response.
        
        If the prompt is a list of chat templates, the classifier
        will classify each prompt. In this case, generated_output["output"].scores
        is used to represent the normalized concept scores of each token corresponding
        to each multi-choice option (e.g. concept score for A, B, C),
        rather than logits, normalized to the range (0, 1).

        Args:
            prompt: Huggingface chat template containing prompt.
            inference_prompt_to_str: In the form func(prompt: dict, tokenizer) -> str
            softmax: If True, normalize concept scores such that they add to 1.
                For binary classification (i.e. not multiple choice), the scores
                add to 1 regardless of `softmax`.
            **_: Unused kwargs for compatible interface
        """
        multi_choice = isinstance(prompt, dict)
        def classify_chat_template(chat_template: dict) -> tuple[str, float]:
            # Initialise representation
            # Use uncontrolled model to generate the responses for forming the representation
            self.rep_manager.train_rep_reader_on_prompt(chat_template, **train_rep_reader_kwargs)
            # Convert prompt to str
            prompt_str = rep_utils.apply_chat_template_for_incomplete_response(chat_template, tokenizer=self.tokenizer)
            output = self.rep_manager.classify_tokens([prompt_str], output_range=(0, 1))[0]
            return prompt_str, output
        
        scores = {}
        if multi_choice:
            for answer_label, chat_template in prompt.items():
                _, curr_score = classify_chat_template(chat_template)

                # NOTE: Although scores are normalized to lie in the range [0, 1],
                # they do not add to 1 for multiple choice
                scores[answer_label] = curr_score
            if softmax:
                normalized_scores = torch.softmax(torch.tensor(list(scores.values())), dim=0).tolist()
                scores = dict(zip(scores.keys(), normalized_scores))
            predicted_label = max(scores, key=scores.get)
        else:
            _, output = classify_chat_template(prompt)
            scores = {
                "True": output, 
                "False": 1 - output
            }
            predicted_label = "True" if output >= 0.5 else "False"     
        
        prompt_str = predicted_label

        generated_output = self.generate_helm_compatible_output_(prompt_str, echo_prompt)

        # Use the concept scores normalised to the range [0, 1] as the output logit array
        token_scores = torch.full(
            generated_output["output"].scores[0].shape,
            fill_value=0.0
        )

        for label, score in scores.items():
            label_token = self.rep_manager.tokenizer(label,
                return_tensors="pt",
                return_token_type_ids=False,
                add_special_tokens=False
            )["input_ids"].item()
            token_scores[0, label_token] = score
        
        generated_output["output"].scores = (token_scores,)

        return generated_output
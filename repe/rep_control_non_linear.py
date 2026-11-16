
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.mistral.modeling_mistral import MistralModel, MistralForCausalLM, MistralConfig, MistralPreTrainedModel
import torch
import time
from typing import Callable
import warnings
from torch.nn import CrossEntropyLoss, Linear
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from functools import partial
from repe.rep_readers_non_linear import NonLinearRep, SurfaceRep
from examples.honesty.rep_manager_utils import RepManager
from repe import rep_utils
from repe.rep_control_contrast_vec import _always_true

def forward_with_control_func(
        self,
        # Model kwargs
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        rotary_embedding = None,
        update_causal_mask = None,

        # Control kwargs
        control_strength: float = None,
        control_func: Callable[..., torch.Tensor] = None,
        control_func_kwargs: dict = {},
        contrast_tokens: int = None,
        control_layer_ids: list[int] = []
    ) -> tuple | BaseModelOutputWithPast:
        """
        Function to override the forward method of the model (the MistralModel that 
        MistralForCausalLM wraps) to include control functionality.
        
        Args:
            control_strength: Value effecting how much to control the model. If None,
                the model is not controlled.
            rotary_embedding: Positional embedding such as
                transformers.models.mistral.modelling_mistral.MistralRotaryEmbedding.
        References:
            From rep_control_contrast_vec.py which is based on MistralModel.forward(),
            modified to handle non-linear representation
        """
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

        if self.gradient_checkpointing and self.training and use_cache:
            warnings.warn(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values: DynamicCache = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = rotary_embedding(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            # Convert index relative to first layer to index relative to last layer
            layer_id = layer_idx - len(self.layers)
            
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

            layer_outputs = forward_function(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )

            hidden_states: torch.Tensor = layer_outputs[0]
            
            # Control the activations for the current layer
            if control_strength is not None:
                if layer_id in control_layer_ids:
                    hidden_states = control_func(
                        hidden_states,
                        layer=layer_id,
                        new_score=control_strength,
                        **control_func_kwargs
                    )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

def new_mistral_forward(
    self: MistralForCausalLM,
    # Model kwargs
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: list[torch.FloatTensor] = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    # Control kwargs
    **control_kwargs
) -> tuple | CausalLMOutputWithPast:
    """
    Forwards arguments to the modified self.model.forward() function 
    (i.e. forward_with_control_func()), so that extra arguments can be
    included (i.e. the control kwargs).
    
    Args:
        control_kwargs: Kwargs for forward_with_control_func
        
    References:
        From MistralForCausalLM.forward(), modified to handle control kwargs
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
        cache_position=cache_position,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        rotary_embedding=self.model.rotary_emb,
        update_causal_mask=self.model._update_causal_mask,
        **control_kwargs
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
        # Ensure tensors are on the same device
        shift_labels = shift_labels.to(shift_logits.device)
        loss_fct = CrossEntropyLoss()
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

def new_prepare_inputs_for_generation(self, old_prepare_inputs_for_generation, input_ids, **model_kwargs):
    # Get kwargs used by the model unrelated to control
    model_inputs: dict = old_prepare_inputs_for_generation(input_ids, **model_kwargs)
    
    # Don't include additional kwargs not related to control
    model_inputs.pop("cache_position", None)
    model_kwargs.pop("cache_position", None)
    
    # Add kwargs related to control to model_inputs
    for key, value in model_kwargs.items():
        if key not in model_inputs:
            model_inputs[key] = value
    return model_inputs 

class NonLinearControlMistralForCausalLM(MistralPreTrainedModel):
    def __init__(self, rep_manager: RepManager):
        """
        Wrapper around rep_manager.model which generates responses by
        controlling the layers in rep_manager.hidden_layers.
        """
        super().__init__(rep_manager.model.config)
        self.model: MistralForCausalLM = rep_manager.model
        self.tokenizer = rep_manager.tokenizer
        self.layers = rep_manager.hidden_layers
        self.rep_manager = rep_manager
        self.original_model_methods = (
            self.model.prepare_inputs_for_generation,
            self.model.forward,
            self.model.model.forward,
            self.model._validate_model_kwargs
        )

    def override_generation_funcs_(self):
        self.model.prepare_inputs_for_generation = partial(new_prepare_inputs_for_generation, self.model, self.model.prepare_inputs_for_generation)
        self.model.forward = partial(new_mistral_forward, self.model)
        self.model.model.forward = partial(forward_with_control_func, self.model.model)
        self.model._validate_model_kwargs = partial(_always_true, self)

    def reset_generation_funcs_(self):
        # Reset forward() methods
        self.model.prepare_inputs_for_generation = self.original_model_methods[0]
        self.model.forward = self.original_model_methods[1]
        self.model.model.forward = self.original_model_methods[2]
        self.model._validate_model_kwargs = self.original_model_methods[3]

    def generate(self,
        prompt: list[dict],
        control_strength: float = 1.0,
        echo_prompt: bool = False,
        inference_prompt_to_str: Callable = rep_utils.apply_chat_template,
        train_rep_reader_kwargs: dict = {},
        base_model_forward_kwargs: dict = {},
        control_func_kwargs: dict = {},
        **_
    ):
        """
        Args:
            prompt: Huggingface chat template containing prompt
            inference_prompt_to_str: Function to convert `prompt` to a str, in the form func(prompt: dict, tokenizer) -> str.
            train_rep_reader_kwargs: Kwargs for RepManager:train_rep_reader_on_prompt() in
                rep_manager_utils.py.
            **_: Unused kwargs for compatible interface
        """
        # This overwrites the forward function of the parent model.
        # To allow for using that model simultaneously without needing
        # to duplicate the parameters or changing its behaviour, we
        # temporarily overwrite them and set them back at the end of this
        # function
        start_time = time.time()
        # Convert prompt to str
        prompt_str = inference_prompt_to_str(chat_template=prompt, tokenizer=self.tokenizer)
        
        # Initialise representation
        # Use uncontrolled model to generate the responses for forming the representation
        self.rep_manager.train_rep_reader_on_prompt(prompt, prompt_str=prompt_str, **train_rep_reader_kwargs)
        
        control_func = self.rep_manager.rep_reader.control_layer
        control_kwargs = dict(
            control_func=control_func,
            control_strength=control_strength,
            control_func_kwargs=control_func_kwargs,
            control_layer_ids=self.rep_manager.hidden_layers
        )
        input_ids = self.tokenizer(
            prompt_str,
            padding=True,
            add_special_tokens=False, # Prompts are already formatted with <s> tags
            return_tensors="pt"
        ).to(self.model.device)

        # Generate controlled response
        self.override_generation_funcs_()
        
        curr_time = time.time()
        
        output = self.model.generate(
            **input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            **control_kwargs,
            **base_model_forward_kwargs
        )
        
        self.reset_generation_funcs_()
        
        # Remove prompt so that the prompt isn't shown
        # again when displaying the output
        prompt_len = len(input_ids['input_ids'][0])
        if not echo_prompt:
            if isinstance(output, torch.Tensor):
                output = output[:, prompt_len:]
            else:
                output.sequences = output.sequences[:, prompt_len:]


        return {"input_length": prompt_len, "output": output}

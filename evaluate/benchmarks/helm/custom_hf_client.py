"""
Override HuggingFaceServer to handle quantized models
"""
import os
import re
import random
import warnings
import torch
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers import BitsAndBytesConfig
from helm.common.hierarchical_logger import htrack_block, hlog
import helm.clients.huggingface_client as hf_client
from helm.tokenizers.tokenizer import Tokenizer
from helm.tokenizers.huggingface_tokenizer import WrappedPreTrainedTokenizer

from helm.common.cache import CacheConfig

from evaluate.helpers.eval_init_model import load_rep_method
from evaluate.eval_config import RepConfig, RepInferenceConfig, BenchmarkMetadata
from repe import rep_utils

def load_configs_from_str(rep_config: str, rep_inference_config: str):
    # Extract configs from deployment file and convert to python object
    configs = []
    for ConfigName, conf_str in [("RepConfig", rep_config), ("RepInferenceConfig", rep_inference_config)]:
        if re.search(f"^{ConfigName}\(", conf_str) is None:
            raise ValueError(f"{conf_str} must be a string representation of a {ConfigName} object")
        configs.append(eval(conf_str))
    
    conf: RepConfig = configs[0]
    inference_conf: RepInferenceConfig = configs[1]

    if conf.representation_method in ["curve", "surface"]:
        max_components = len(inference_conf.train_rep_reader_kwargs["concept_template_kwargs"]["prompt_templates"])
        if conf.rep_reader_kwargs["n_components"] > max_components:
            warnings.warn("n_components cannot exceed number of prompts in the concept template. "
                f"Overwriting n_components from {conf.rep_reader_kwargs['n_components']} to {max_components}")
            conf.rep_reader_kwargs["n_components"] = max_components

    return conf, inference_conf

class CustomHuggingFaceServer(hf_client.HuggingFaceServer):
    """
    A thin wrapper around a HuggingFace AutoModelForCausalLM for HuggingFaceClient to call.
    References:
        Based on https://github.com/stanford-crfm/helm/blob/main/src/helm/clients/huggingface_client.py,
        modified to load and handle a representation method    
    """
    def __init__(self,
        pretrained_model_name_or_path: str,
        wrapped_tokenizer: WrappedPreTrainedTokenizer,
        **kwargs
    ):

        with htrack_block(f"Loading Hugging Face model {pretrained_model_name_or_path}"):
            self.conf, self.inference_conf = load_configs_from_str(kwargs["rep_config"], kwargs["rep_inference_config"])
            
            self.inference_conf.train_rep_reader_kwargs["concept_template_kwargs"]["process_prompt"] = self.process_prompt
            
            # Load model
            model, tokenizer, _ = load_rep_method(self.conf)
            self.model = model

            # Need to set device for HuggingFaceServer
            self.device = model.device

        if tokenizer is not None:
            with htrack_block(f"Loading Hugging Face tokenizer for model {pretrained_model_name_or_path}"):
                self.wrapped_tokenizer = wrap_tokenizer(tokenizer)

        if self.conf.representation_method == "random" and not self.inference_conf.benchmark.is_multi_choice:
            raise ValueError("'Random' method is only supported for multiple choice")

    def split_nshot_prompt(self, prompt: str) -> list[str]:
        """
        Removes examples (if any) from the prompt if zero shot is enabled, but
        doesn't remove anything if few-shot is enabled. 
        Returns:
            prompt: A list containing few-shot examples (if any), which each
                form an element of the list, followed by the prompt.

        """
        benchmark = self.inference_conf.benchmark
        if not benchmark.supports_few_shot and not self.conf.zero_shot:
            raise ValueError(f"Few-shot is not supported for benchmark {benchmark.name}")

        using_classification_model = not self.conf.control_mode
        nshot_filter = RepInferenceConfig.get_nshot_list_filter(benchmark.name)
        prompt = nshot_filter(
            prompt=prompt,
            is_multi_choice=benchmark.is_multi_choice,
            zero_shot=self.conf.zero_shot,
            using_classification_model=using_classification_model
        )
        
        return prompt

    def nshot_list_to_chat_template(self, prompt: list[str]) -> list:
        """
        Converts str prompt to chat template.

        Returns:
            prompt: In chat template format
        """
        to_chat_template = RepInferenceConfig.get_chat_template_converter(
            benchmark_name=self.inference_conf.benchmark.name,
            classification=not self.conf.control_mode
        )
        return to_chat_template(prompt)

    def process_prompt(self, prompt: list[dict], control_strength: float = 1.0) -> list[dict]:
        """
        Modify a few-shot prompt so that the answer for each few-shot example is
        consistent with the concept strength.
        
        Args:
            control_strength:
                If +1, all few-shot answers are correct (same as original prompt)
                If -1, all answers are randomly chosen from possible incorrect answers
                Otherwise, probability of correct answer increases with concept strength
        Returns:
            modified prompt
        """
        # Few-shot -> Generate alternative answers
        if self.conf.control_mode and (control_strength < 1):
            # Ignore the last prompt-response pair since it doesn't have a completed answer
            # Since zero-shot only has at most 1 pair, the loop is completely ignored
            for example_id in range(0, len(prompt) - 2, 2):
                # Extract correct and incorrect answers from current sample response
                question, answer = prompt[example_id]['content'], prompt[example_id + 1]['content']
                correct_answer = answer[-1]
                incorrect_answers = rep_utils.get_possible_answers(question)
                incorrect_answers.remove(correct_answer)
                
                # Get random number in range [-1, 1) (i.e. range of control_strength) 
                rand_float = (2 * random.random() - 1)
                # The lower the concept strength, the more likely that
                # the correct answer will be replaced with a random incorrect answer
                if control_strength <= rand_float:
                    incorrect_choice_id = random.randint(0, len(incorrect_answers) - 1)
                    incorrect_choice = incorrect_answers[incorrect_choice_id]
                    prompt[example_id + 1]['content'] = answer[:-1] + incorrect_choice
                # prompt[i + 1] = '\n'.join(example)

        return prompt

    def chat_template_to_str(self, chat_template: list[dict], tokenizer):
        """
        Convert chat template to str according to the config.

        Returns:
            template_idx: Index of the instruction in the
                instruction template dict to prepend to the prompt
                used for inference (i.e. base prompt).
        """
        template_idx = None
        if self.conf.representation_method == "none" or self.conf.apply_concept_to_base_prompt:
            # Prepend the instruction for concept strength == 1
            # for no representation and concept strength == 0 otherwise
            template_kwargs = self.inference_conf.train_rep_reader_kwargs["concept_template_kwargs"]
            if self.conf.representation_method == "none":
                template_idx = 0
            else:
                template_idx = round(len(template_kwargs["prompt_templates"]) / 2)

            prompt_str = rep_utils.apply_concept_template(
                chat_template, tokenizer=tokenizer, **template_kwargs
            )[0][template_idx]
        elif self.inference_conf.format_as_chat_template:
            prompt_str = rep_utils.apply_chat_template_for_incomplete_response(chat_template, tokenizer)
        else:
            prompt_str = chat_template

        return prompt_str

    def serve_request(self, raw_request: hf_client.HuggingFaceRequest) -> dict:
        prompt = self.split_nshot_prompt(raw_request["prompt"])
        # Don't convert prompts that are already in chat template format
        # (i.e. list[dict] rather than list[str])
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], str):
            prompt = self.nshot_list_to_chat_template(prompt)
        raw_request["prompt"] = self.process_prompt(prompt)

        # The model generates numerical values (i.e. control scores) by overwriting
        # the logits, with dummy text output, so that the same interface as control
        # can be used
        classification = not self.conf.control_mode
            
        with self.wrapped_tokenizer as tokenizer:
            if classification:
                prompt_str = ""
            else:
                prompt_str = self.chat_template_to_str(raw_request["prompt"], tokenizer)
            
            # Convert str to tokens
            encoded_input = tokenizer(prompt_str, return_tensors="pt", return_token_type_ids=False).to(
                self.device
            )
        stopping_criteria: StoppingCriteriaList | None = None
        optional_args = self.inference_conf.base_model_forward_kwargs
        if len(raw_request["stop_sequences"]) > 0:
            with self.wrapped_tokenizer as tokenizer:
                stop_sequence_ids = tokenizer(
                    raw_request["stop_sequences"], return_token_type_ids=False, add_special_tokens=False
                )
            if len(stop_sequence_ids.input_ids) == 1 and len(stop_sequence_ids.input_ids[0]) == 1:
                optional_args["eos_token_id"] = stop_sequence_ids.input_ids[0][0]
            else:
                stopping_criteria = StoppingCriteriaList()
                for stop_sequence_input_ids in stop_sequence_ids.input_ids:
                    stopping_criteria.append(hf_client.StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))

        # Check if we need to compute the perplexity of the prompt (#1497)
        compute_logprobs_only = (
            raw_request["max_new_tokens"] == 0
            and raw_request["num_return_sequences"] == 1
            and raw_request["echo_prompt"]
        )

        # Use HuggingFace's `generate` method.
        if compute_logprobs_only:
            with torch.no_grad():
                output = self.model([raw_request["prompts"]])
            sequences = encoded_input["input_ids"]
            scores = output.logits
            input_length = len(encoded_input.input_ids[0])
        else:
            
            output = self.model.generate(
                raw_request["prompt"],
                control_strength=self.inference_conf.control_strength,
                train_rep_reader_kwargs=self.inference_conf.train_rep_reader_kwargs,
                control_func_kwargs=self.inference_conf.control_func_kwargs,
                image_generation_kwargs=self.conf.image_generation_kwargs,
                echo_prompt=True,
                inference_prompt_to_str=self.chat_template_to_str,
                base_model_forward_kwargs={
                    # "temperature": raw_request["temperature"],
                    "num_return_sequences": raw_request["num_return_sequences"],
                    "max_new_tokens": raw_request["max_new_tokens"],
                    "top_p": raw_request["top_p"],
                    "do_sample": True,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "stopping_criteria": stopping_criteria,
                    **optional_args,
                }
            )
            input_length, output = output["input_length"], output["output"]
            sequences = output.sequences
            scores = output.scores

        # Classifiers already return normalised concept scores, so softmax is not needed
        if classification:
            to_logprobs = lambda logits: torch.log(logits)
        else:
            to_logprobs = lambda logits: torch.nn.functional.log_softmax(logits, dim=0)
            
        prompt_tokens_logprobs = []
        if compute_logprobs_only:
            # Append the logprob of the first token of the prompt.
            prompt_tokens_logprobs.append(0.0)

            # Compute logprobs of prompt tokens.
            for completion_id in range(raw_request["num_return_sequences"]):
                for i in range(len(sequences[completion_id]) - 1):
                    logprobs = to_logprobs(scores[completion_id][i])
                    prompt_tokens_logprobs.append(logprobs[sequences[completion_id][i + 1]].item())

        # Compute logprobs of generated tokens for each completed sequence.
        all_generated_tokens_logprobs = []
        for completion_id in range(raw_request["num_return_sequences"]):
            generated_tokens_logprobs = []
            for i in range(len(sequences[completion_id]) - input_length):
                logprobs = to_logprobs(scores[i][completion_id])
                # Get log probability of chosen token.
                j = i + input_length
                generated_tokens_logprobs.append(logprobs[sequences[completion_id][j]].item())
            all_generated_tokens_logprobs.append(generated_tokens_logprobs)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[input_length:] for sequence in sequences]

        with self.wrapped_tokenizer as tokenizer:
            all_tokens = [[tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
            all_decoded_text = tokenizer.batch_decode(sequences)

        completions = []
        for decoded_text, tokens, generated_tokens_logprobs in zip(
            all_decoded_text, all_tokens, all_generated_tokens_logprobs
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": generated_tokens_logprobs,
                    "prompt_logprobs": prompt_tokens_logprobs,
                }
            )

        return {"completions": completions, "input_length": input_length}



hf_client.HuggingFaceServer = CustomHuggingFaceServer

def clear_hf_server_cache():
    with hf_client.HuggingFaceServerFactory._servers_lock:
        hf_client.HuggingFaceServerFactory._servers = {}

class CustomHuggingFaceClient(hf_client.HuggingFaceClient):
    """Client for OpenLM: https://github.com/mlfoundations/open_lm"""
    def __init__(self,
        cache_config: CacheConfig, 
        tokenizer: Tokenizer,
        pretrained_model_name_or_path: str | None = None,
        **kwargs
    ):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs = {"quantization_config": quantization_config, **kwargs}
        
        super().__init__(
            cache_config=cache_config, 
            tokenizer=tokenizer,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **kwargs
        )

def wrap_tokenizer(tokenizer) -> WrappedPreTrainedTokenizer:
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    return WrappedPreTrainedTokenizer(tokenizer)
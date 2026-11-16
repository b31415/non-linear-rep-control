import os
import re
import gc
import json
import random
import warnings
from math import ceil, log2
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from repe import repe_pipeline_registry, rep_utils
from repe.rep_readers_non_linear import get_hidden_states_for_rep_reader, DIRECTION_FINDERS, NonLinearRepReadingPipeline


class RepManager():
    REP_METHOD_CHOICES = ("contrast-vector", "curve", "surface", "taylor", "least-squares")
    ONLINE_METHODS = ("contrast-vector", "curve", "surface", "taylor", "least-squares")
    NON_LINEAR_METHODS = ("curve", "surface", "taylor", "least-squares")
    NL_PROJ_METHODS = ("curve", "surface")
    METHOD_CHOICES = ("none", "random", *REP_METHOD_CHOICES)

    """Unified class to load and handle representation methods and Pipelines"""
    def __init__(self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        rep_reader_kwargs: dict,
        method: str = "surface",
        facts_data_path: str = "./data/facts/facts_true_false.csv",
        prompts_data_path: str = "./examples/honesty/data/honesty_rep_prompts.json",
        n_statements: int = 32,
        hidden_layer_args: tuple = ("uniform", None)
    ):
        """
        Sets up pipeline and Rep with default args for Honesty classification.
        Args:
            method: Choices are "contrast-vector", "curve", "surface", "taylor" and "least-squares"
            hidden_layer_args:
                1st arg:
                    "weighted": Learn weights for all layers using a linear layer.
                    "uniform": Use a weight of 1 for each layer.
                2nd arg:
                    arguments of range() to specify which layers. Defaults to all layers.
        """
        if hidden_layer_args[1] is None:
            hidden_layer_args = (hidden_layer_args[0], (-1, -model.config.num_hidden_layers, -1))
        self.hidden_layer_args = hidden_layer_args
        self.hidden_layers = list(range(*hidden_layer_args[1]))

        self.tags = ("<s>[INST]", "[/INST]")
        self.facts_data_path = facts_data_path
        self.tokenizer = tokenizer
        self.model = model

        self.is_online = (method in self.ONLINE_METHODS)

        if method not in self.REP_METHOD_CHOICES:
            raise ValueError(f"Unsupported method: {method}")
        else:
            self.method = method
            self.non_linear = (method in DIRECTION_FINDERS.keys())
        
        self.rep_reader_kwargs = rep_reader_kwargs

        # Setup Pipeline
        if self.non_linear:
            self.pipeline_mode = 'non-linear-rep-reading'
        else:
            self.pipeline_mode = 'rep-reading'
        repe_pipeline_registry()
        self.rep_reading_pipeline = pipeline(
            self.pipeline_mode,
            model=model,
            tokenizer=tokenizer
        )

        # Set linear/non-linear-specific commands and load dataset
        print("Getting dataset for extracting representations...")
        dataset = None
        use_layer_weights = (hidden_layer_args[0] != "uniform")

        if self.non_linear:
            self.direction_method = method
            if use_layer_weights:
                with open(prompts_data_path) as f:
                    prompts = json.load(f)
                prompt_prefixes = get_hidden_states_for_rep_reader(prompts)
                dataset = honesty_levels_dataset(prompt_prefixes, facts_data_path, tokenizer, *self.tags, n_statements=n_statements)
            
            self.rep_reader = None
        else:
            self.direction_method = 'pca'
            if use_layer_weights or not self.is_online:
                dataset = honesty_function_dataset(facts_data_path, tokenizer, *self.tags)
                if not self.is_online:
                    print("Training honesty representation reader...", end="")
                    self.rep_reader = self.rep_reading_pipeline.get_directions(
                        train_inputs = dataset['train']['data'],
                        train_labels = dataset['train']['labels'], 
                        hidden_layers = self.hidden_layers, 
                        direction_method = self.direction_method,
                        direction_finder_kwargs = self.rep_reader_kwargs
                    )
                    print(" Done.")
            
        self.weights_plotted = False
        self.train_layer_weights(
            dataset,
            n_statements = n_statements,
            device = model.device,
            dtype = model.dtype
        )

    def train_layer_weights(self, dataset: Dataset, n_statements: int, dtype: torch.dtype, device=torch.device):
        """DEPRECATED"""
        if dataset is None:
            print("Defaulting to uniform weights")
            self.hidden_layer_args = ("uniform", *self.hidden_layer_args[1:])
        else:
            dataset = dataset["train"]["whole_statements"]

        # Don't use a bias if the layer won't be learned
        n_weights = len(self.hidden_layers)
        include_bias = (self.hidden_layer_args[0] == "weighted")
        self.layer_weights = nn.Linear(n_weights, 1,
            bias=include_bias,
            device=device,
            dtype=dtype
        )

        if self.hidden_layer_args[0] == "weighted":
            torch.nn.init.normal_(self.layer_weights.weight, 0.5/n_weights, 0.3/n_weights)
            self.layer_weights.train()
            loss_func = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(self.layer_weights.parameters())
            dataloader = DataLoader(dataset, batch_size=n_statements//4, shuffle=True)

            print("Training layer weights...")
            for statements, labels in tqdm(dataloader):
                optimizer.zero_grad()
                preds = self.classify_tokens(statements)
                labels = labels.to(device=device, dtype=preds.dtype)# + 1) / 2
                loss = loss_func(preds, labels)
                loss.backward()
                optimizer.step()
                # Only allow positive weights to improve interpretability as a weighted average
                with torch.no_grad():
                    self.layer_weights.weight.clamp_(min=0.001, max=1/n_weights)

            weights = self.layer_weights.weight.squeeze().tolist()
            print(f"Layer weights: {weights}, bias: {self.layer_weights.bias[0]}")
            plt.figure()
            plt.plot(self.hidden_layers, weights)
            plt.title("Learned parameters for aggregating layer honesty scores")
            plt.xlabel("Layers")
            plt.ylabel(f"Weights (bias = {self.layer_weights.bias.item():.3f})")
            self.weights_plotted = True
        else:
            with torch.no_grad():
                torch.nn.init.constant_(self.layer_weights.weight, 1/n_weights)

        self.layer_weights.eval()

    def run_test_set(self, test_set = None, batch_size: int = 16):
        if test_set is None:
            test_set = honesty_function_dataset(self.facts_data_path, self.tokenizer, *self.tags)['test']['data']
        H_tests = self.rep_reading_pipeline(
            test_set,
            hidden_layers=self.hidden_layers, 
            rep_reader=self.rep_reader,
            batch_size=batch_size)
        return H_tests
    
    def train_rep_reader_on_prompt(self,
        prompt: list[dict],
        concept_template_kwargs: dict,
        n_variations_per_concept_level: int = 5,
        model_kwargs: dict = {"max_new_tokens": 40},
        prompt_str = None,
        remove_instructions: bool = False,
        **get_directions_kwargs
    ) -> None:
        """
        Extracts a representation using responses to the prompt generated by the model. 
        Args:
            prompt: Prompt in the form of a Huggingface chat template.
                E.g. [{"role": "user", "content": prompt_str}]
            concept_template_kwargs: prompt_templates or concept_words
            n_variation_per_honesty_class: Generates this number of responses to the prompt
                for each honesty class. The representation method then averages the hidden
                states for each class.
            model_kwargs: Kwargs for model.generate.
                See https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate
            prompt_str: The `prompt' chat template convert to a string in the same format
                as the prompt passed to the model for inference. Default: uses rep_utils.apply_chat_template
                to convert `prompt'.
            remove_instructions: Whether to convert generated prompts from the form
                [INST] <INSTRUCTION> <PROMPT> [/INST] <RESPONSE> to 
                [INST] <PROMPT> [/INST] <RESPONSE>
            get_directions_kwargs: Kwargs for self.rep_reading_pipeline.get_directions().
        """
        if "prompt_templates" not in concept_template_kwargs.keys():
            raise ValueError("concept_template_kwargs must contain prompt_templates. See rep_utils.apply_concept_template() for format.")
        self.rep_reading_pipeline: NonLinearRepReadingPipeline
        prompts, labels = rep_utils.apply_concept_template(prompt, tokenizer=self.tokenizer, **concept_template_kwargs)
        generated_response_labels = list(np.repeat(labels, n_variations_per_concept_level))

        # Generate n variations for each prompt, then average hidden states for responses with the same honesty label
        batch_size = 1
        curr_prompt = 0
        max_batch_size_reached = False
        generated_responses = []

        gc.collect()
        torch.cuda.empty_cache()
        initial_vram_free = torch.cuda.mem_get_info()[0]
        
        while curr_prompt < len(prompts):
            batch = prompts[curr_prompt:min(curr_prompt + batch_size, len(prompts))]
            curr_prompt += batch_size

            input_ids = self.tokenizer(batch, padding=True, add_special_tokens=False, return_tensors="pt").to(self.model.device)
            # num_beams doesn't work for n = 1
            beam_search_kwargs = {
                "num_return_sequences": n_variations_per_concept_level,
                "num_beams": n_variations_per_concept_level,
                "num_beam_groups": n_variations_per_concept_level,
                "diversity_penalty": 1.0,
                "early_stopping": True
            } if n_variations_per_concept_level > 1 else {}

            try:
                outputs = self.model.generate(**input_ids, pad_token_id=2, **beam_search_kwargs, **model_kwargs)
            except RuntimeError as e:
                if batch_size == 1:
                    # Can't handle exception since this is the minimum batch size
                    raise e
                else:
                    # Round down to the next power of 2
                    print("Reducing batch size")
                    batch_size = (2**ceil(log2(batch_size))) - 1


            if not max_batch_size_reached:
                # Heuristic to estimate the max batch size:
                # Use the largest number of batches that can fit given the
                # amount used by the first batch
                batch_lengths = [len(p) for p in batch]
                prompt_lengths = [len(p) for p in prompts]
                current_vram_free = torch.cuda.mem_get_info()[0]
                prompt_len_ratio = max(prompt_lengths) / max(batch_lengths)
                batch_vram_usage = (initial_vram_free - current_vram_free) * prompt_len_ratio
                
                if batch_vram_usage == 0:
                    # Avoid division by zero in case batch_vram_usage == 0
                    batch_size = len(prompts)
                else:
                    batch_size = int(max(initial_vram_free / batch_vram_usage, 1))
                max_batch_size_reached = True

            # Get instructions + responses as strings
            curr_generated_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_responses.extend(curr_generated_responses)

            gc.collect()
            torch.cuda.empty_cache()

        # Remove special tokens from prompts, since the special tokens were 
        # already removed from outputs to get rid of the padding
        prompts = self.tokenizer.batch_decode(
            self.tokenizer(prompts, padding=False, add_special_tokens=False)['input_ids'],
            skip_special_tokens=True
        )
        
        if remove_instructions:
            # Get prompt used for inference
            if prompt_str is None:
                original_prompt = rep_utils.apply_chat_template(chat_template=prompt, tokenizer=self.tokenizer)
            else:
                original_prompt = prompt_str
            prompts = len(generated_responses) * [original_prompt]

            end_of_prompt_tag = self.tags[1] # E.g. "[/INST]"
            responses_only = []
            for response in generated_responses:
                # Get position of start of response (i.e., end of prompt + size of tag)
                end_of_prompt_idx = response.find(end_of_prompt_tag)
                if end_of_prompt_idx == -1:
                    responses_only.append(response)
                    warnings.warn(f"Could not find end of prompt tag {end_of_prompt_tag}, "\
                        "so original the prompt was not removed from the response.")
                else:
                    start_of_response_idx = end_of_prompt_idx + len(end_of_prompt_tag)
                    response = response[start_of_response_idx:]
                    responses_only.append(original_prompt + response)

            start_of_response_idx = len(original_prompt)
            generated_responses = responses_only

        # Use the number of tokens in the prompt as the token index to start averaging states
        # from for each response in order to only extract representations using response states.
        # E.g. if [63, 61, 60] are the prompt lengths, with 2 variations, the result will be [63, 63, 61, 61, 60, 60]
        n_tokens_in_prompts = [len(curr_tokens) \
            for curr_tokens in self.tokenizer(prompts, padding=False, add_special_tokens=False)['input_ids'] \
            for _ in range(n_variations_per_concept_level)]
        

        self.rep_reader = self.rep_reading_pipeline.get_directions(
            train_inputs=generated_responses,
            train_labels=generated_response_labels,
            hidden_layers=self.hidden_layers,
            rep_token = n_tokens_in_prompts,
            direction_method = self.direction_method,
            direction_finder_kwargs = self.rep_reader_kwargs,
            **get_directions_kwargs
        )
    
    def classify_tokens(self,
        responses: list[str],
        pos: list[int] | tuple[str, str] | None = None,
        thresholds: list[float] | None = None,
        output_range: tuple[float, float] = (-1, 1),
        quality: str = 'low',
        average_tokens: bool = True,
        response_start_pos: list = None
    ) -> list[float] | list[str] | list[list[float]] | list[list[str]]:
        """
        Classifies a one or more tokens or an entire response using the representation.

        Args:
            responses: Batch of responses. ASSUMES start of sentence token has
                been added (e.g. using a chat template)
            pos:
                <pos> is a list: For each integer, the token with that index relative to the end of the response
                will be classified, then the score for each token will be averaged.
                <pos> is a tuple (containing a pair of strings): The strings will be treated like a pair
                of brackets and the total honesty score of the tokens within the last pair of brackets in the response
                will be calculated. The brackets are just used to identify the token(s) to classify and are removed
                    so that they do not effect the score.
                <pos> is "response": Each element of <responses> contains a prompt and response formatted use
                    the tokenizer chat template. Classify the last response.
                <pos> is None: Classify all positions (if response_start_pos is given, classify positions after this).
            thresholds: 1 (or 2) floats representing the decision boundaries between true, (neutral) and false. E.g. [-0.5, 0.5].
                Thresholds are defined with respect to the <output_range>.
            output_range: The honesty value is normalised to lie within the given output range.
            quality:
                'high': More computationally expensive. Calculates the score for the current token by comparing the overall
                    score for the response to the counterfactual response with the token removed. Intended for token-level
                    classification (i.e. not sentence-level classification, which aggregates the scores for all tokens anyway)
                'low': Default. Directly get the score from the representation of the current token.
            average_tokens: If True, compute the average classification score of all tokens in the response.
            response_start_pos: List of the char idx of the start of the response part (i.e. after the [/INST] tag) for each
                element of <responses>. Only required if quality == 'high'.
        Return:
            honesty_values: If <thresholds> is None, an honesty score is returned for each response (the aggregate score over
                all specified positions). Otherwise, this score is converted to either 'T', 'N' or 'F'. If average_tokens is False,
                the output will be a list of lists with shape [n_responses, n_tokens].
        """
        if self.rep_reader is None:
            raise RuntimeError("A representation must be computed before calling this function such as by using self.train_rep_reader_on_prompt()")
        if response_start_pos is not None and quality != 'high':
            warnings.warn("'response_start_pos' value was ignored")

        device = self.layer_weights.weight.device
        dtype = self.layer_weights.weight.dtype
        initial_pos_type = type(pos)

        # Lambda to get the sign of the honesty representation for a given layer 
        if self.non_linear:
            sign = lambda _: 1
        else:
            sign = lambda layer: self.rep_reader.direction_signs[layer][0]
        
        if initial_pos_type == tuple:
            reverse_close = pos[1][::-1]
            reverse_open = pos[0][::-1]
            # Match the first open-close pair of the reversed string (i.e. first from the end of the string)
            pattern = re.compile(fr"{reverse_close}.*?{reverse_open}")

        
        honesty_scores = []

        for response in responses:
            response_tokens = self.tokenizer(response, add_special_tokens=False)
            n_tokens = len(response_tokens.tokens())

            ### Calculate all the classification scores for the current response ###
            if quality == 'high':
                if response_start_pos is None:
                    raise TypeError("'response_start_pos' cannot be None when 'quality' is 'high'")
                response_start_pos = response_tokens.char_to_token(response_start_pos)
                baseline_pos = list(range(response_start_pos - n_tokens, 0))
                baseline_honesty_scores = self.classify_tokens(responses, pos=baseline_pos,
                    output_range=output_range, average_tokens=False)[0]
                baseline_avg_score = torch.mean(baseline_honesty_scores)

            ### Calculate token indices to classify (pos) ###
            if initial_pos_type != list:

                # Classify all tokens (starting from response_start_pos, if given)
                if initial_pos_type == type(None):
                    if response_start_pos is None:
                        # Don't classify start of sentence token
                        response_start_pos = 1
                    response_start_pos = response_tokens.char_to_token(response_start_pos)
                    pos = list(range(response_start_pos - n_tokens, 0))

                # If pos is a tuple[str, str], convert it to a list[int] by finding the
                # indices of tokens inside the brackets"""
                elif initial_pos_type == tuple:
                    # Get the index of the start and end of the last pair of brackets
                    match = re.search(pattern, response[::-1])
                    if match is None:
                        raise ValueError(f'Match for {pos} not found in response "{response}"')
                    start_char_idx = len(response) - match.end()
                    end_char_idx = len(response) - match.start()

                    # Remove brackets to avoid effecting classification score
                    response = response[:start_char_idx] + \
                        response[start_char_idx + len(reverse_open):end_char_idx - len(reverse_close)] + \
                        response[end_char_idx:]
                    
                    # Get the index of the end of the text that was
                    # inside the brackets after removing the brackets (start stays the same)
                    end_char_idx -= len(reverse_open) + len(reverse_close)

                    # Get the indices (in the token array) of all tokens inside those brackets
                    start_token_idx = response_tokens.char_to_token(start_char_idx)
                    end_token_idx = response_tokens.char_to_token(end_char_idx)
                    end_token_idx = 0 if end_token_idx is None else end_token_idx # start_token_idx is the last token
                    pos = list(range(start_token_idx, end_token_idx))
                    
                    # Make the tokens relative to the end of the string
                    pos = [id - n_tokens for id in pos]
            

            response_tokens = response_tokens['input_ids']
            scores_for_curr_response = torch.zeros(
                (1 if quality == 'high' else len(self.hidden_layers), len(pos)),
                device=device, dtype=dtype
            )
            
            for pos_idx, p in enumerate(pos):
                # Get the honesty scores for all layers for the current position
                if quality == 'high':
                    # Delete the token at the current pos (add len(response_tokens) to work in the case where p + 1 = 0)
                    response_with_token_removed = [*response_tokens[:p], *response_tokens[p + 1 + len(response_tokens):]]
                    response_with_token_removed = self.tokenizer.decode(response_with_token_removed)

                    # Get the honesty scores for the tokens after the token that was at position p
                    pos_with_token_removed = list(range(p + 1, 0))
                    scores = self.classify_tokens(responses, pos=pos_with_token_removed,
                        output_range=output_range, average_tokens=False)[0]
                    # Get the average honesty score by reusing the scores before p from
                    # <baseline_honesty_scores> and the new scores for tokens after p
                    scores = torch.mean(torch.tensor([*baseline_honesty_scores[:p], *scores], device=device, dtype=dtype))
                    # If the score for the response (baseline) is greater than the response with
                    # the token removed then we classify the token as having a negative effect
                    scores = baseline_avg_score - scores
                    print(f"score: {scores.item()}, baseline: {baseline_avg_score.item()}, {response_with_token_removed}")
                else:
                    scores = self.rep_reading_pipeline([response],
                        rep_reader=self.rep_reader,
                        rep_token=p,
                        hidden_layers=self.hidden_layers)
                
                    # Sign correction: RepReaders from the original RepE paper take the sign into account separately
                    scores = [sign(layer) * scores[0][layer][0] for layer in self.hidden_layers]

                scores_for_curr_response[:, pos_idx] = torch.tensor(scores, device=device, dtype=dtype)

            # Get the overall honesty score for the response
            if average_tokens:
                scores_for_curr_response = torch.mean(scores_for_curr_response, axis=1)

            # Convert tensor from shape [n_layers, n_pos] to [n_pos]
            # Save the weighted average until the end instead of computing it for each pos
            # so that the honesty_scores can be packaged into a single tensor and gradients can be computed.
            if quality != 'high': # Hidden states are already averaged for high quality mode
                scores_for_curr_response = self.layer_weights(scores_for_curr_response.T)
                #scores_for_curr_response = torch.median(scores_for_curr_response, dim=0).values.unsqueeze(0)

            """Scale honesty scores to lie within the output range"""
            r0, r1 = output_range
            scores_for_curr_response = ((scores_for_curr_response * (r1 - r0) + r1 + r0) / 2).tolist()

            """Convert honesty scores to classes"""
            if thresholds is not None:
                def honesty_score_to_class(score):
                    if score < min(thresholds):
                        return 'F'
                    elif score >= max(thresholds):
                        return 'T'
                    else:
                        return 'N'
                    
                scores_for_curr_response = [honesty_score_to_class(score) for score in scores_for_curr_response]

            if average_tokens:
                # Return shape [n_responses]
                scores_for_curr_response = scores_for_curr_response[0]
            
            honesty_scores.append(scores_for_curr_response)
        
        return honesty_scores

def load_model(model_name: str, compile: bool = True, flash_attn: bool = True, fast_tokenizer: bool = True, adapter_path: str | None = None):
    """
    Args:
        compile: Disable for greater compatibility
        flash_attn: Disable for greater compatibility (e.g. allow right padding for tokenizer)
        fast_tokenizer: Whether to use a fast tokeniser. Use False when training.
        adapter_path: Path to an optional PEFT adapter relative to the current working directory. 
    Returns:
        model, tokenizer
    """
    # Clears GPU memory taken by previous models (e.g. if this is called multiple times)
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        # Speed optimisations
        quantization_config = quantization_config,
        attn_implementation = "flash_attention_2" if flash_attn else None
    )
    model.eval()
    if compile:
        model = torch.compile(model, mode="reduce-overhead")
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            model_id = adapter_path,
            is_trainable = False,
            adapter_name = os.path.basename(adapter_path)
        )
        
    use_fast_tokenizer = fast_tokenizer and "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

class WholeStatementDataset(Dataset):
    """Dataset used to train weights for weighted average of layers."""
    def __init__(self, statements, labels):
        super().__init__()
        self.statements = statements
        self.labels = labels

    def __getitem__(self, index):
        return self.statements[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

def honesty_levels_dataset(
    honesty_prefixes: list[tuple[int, str]],
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    user_tag: str = "",
    assistant_tag: str = "",
    seed: int = 0,
    n_statements: int = 32,
    stride: int = 4
) -> dict[str, dict[str, list | Dataset]]:
    """
    Processes data to create training and testing datasets based on honesty for several
    different levels of honesty.

    Args:
    - honesty_prefixes: (honesty_level, prompt_prefix) pairs.
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.
    - n_statements (int): The dataset length is n_statements x n_honesty_classes 
    - stride (int): Truncate statements to n, n + <stride>, n + 2*<stride>, etc
        words long. Increase stride to get more variety for small values of <n_statements>.
    Returns:
    - Dict in the form {
        'train': {'labels': [...], 'data': [...], 'whole_statements': Dataset},
        'test': {...}
      }
      where whole_statements is a dataset returning (statements, labels)
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()[:2*n_statements]
    false_statements = true_statements = df[df['label'] == 0]['statement'].values.tolist()[:2*n_statements]

    instruction = "With this persona, tell me some general knowledge facts about the world."
    honesty_statements = []

    # Process statements
    versions_per_statement = []
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)
        # min part is to prevent short statements from being truncated too much.
        # len(tokens) = [1, 2, 3, 4, 5, 6, 7, 8, 9] => max_t_len = [0, 1, 2, 3, 3, 3, 4, 5, 6]
        max_truncated_length = max(len(tokens) - 3, min(len(tokens) - 1, 3))

        truncated_lengths = range(1, max_truncated_length, stride)
        versions_per_statement.append(len(truncated_lengths) * len(honesty_prefixes))

        for idx in truncated_lengths:
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            for level, prefix in honesty_prefixes:
                honesty_statements.append((level, f"{user_tag} {prefix} {instruction} {assistant_tag} {truncated_statement}"))

    def data_to_dict(start_whole: int, end_whole: int) -> dict[str, list]:
        """
        Args:
            start_whole: Index of the first statement to use for the current
                part of the dataset (i.e. train set or test set).
            end_whole: Index of the end of the current subset.
        """
        # The main dataset is formed by the truncated sentences
        start_truncated = sum(versions_per_statement[:start_whole])
        end_truncated = sum(versions_per_statement[:end_whole])
        data = honesty_statements[start_truncated:end_truncated]
        labels, data = [list(l) for l in zip(*data)]

        # Also generate the untruncated sentences, including false statements
        whole_statements = []
        for statements, score in [(true_statements, 1.0), (false_statements, -1.0)]:
            whole_statements.extend([(statement, score) for statement in statements[start_whole:end_whole]])

        whole_statements, whole_labels = zip(*whole_statements)
        statement_dataset = WholeStatementDataset(whole_statements, whole_labels)
        return {'labels': labels, 'data': data, 'whole_statements': statement_dataset}
    

    # Create train and test dataset
    dataset = {
        'train': data_to_dict(0, n_statements),
        'test': data_to_dict(n_statements, 2*n_statements)
    }
    print(f"Train data: {len(dataset['train']['data'])}")
    print(f"Test data: {len(dataset['test']['data'])}")
    return dataset

#----------------------------------
def honesty_function_dataset(data_path: str, tokenizer: PreTrainedTokenizer, user_tag: str = "", assistant_tag: str = "", seed: int = 0) -> dict:
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)
        
        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append(f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + truncated_statement)
            untruthful_statements.append(f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} " + truncated_statement)

    # Create training data
    ntrain = 512
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }

def plot_detection_results(input_ids, rep_reader_scores_dict, THRESHOLD, start_answer_token=":"):

    cmap=LinearSegmentedColormap.from_list('rg',["r", (255/255, 255/255, 224/255), "g"], N=256)
    colormap = cmap

    # Define words and their colors
    words = [token.replace('▁', ' ') for token in input_ids]

    # Create a new figure
    fig, ax = plt.subplots(figsize=(12.8, 5), dpi=200)

    # Set limits for the x and y axes
    xlim = 10000
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 10)

    # Remove ticks and labels from the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Starting position of the words in the plot
    x_start, y_start = 1, 9

    # Initialize positions and maximum line width
    x, y = x_start, y_start
    max_line_width = xlim

    y_pad = 0.4
    word_width = 0

    iter = 0

    selected_concepts = ["honesty"]
    norm_style = ["mean"]
    selection_style = ["neg"]

    for rep, s_style, n_style in zip(selected_concepts, selection_style, norm_style):

        rep_scores = np.array(rep_reader_scores_dict[rep])
        mean, std = np.median(rep_scores), rep_scores.std()
        rep_scores[(rep_scores > mean+5*std) | (rep_scores < mean-5*std)] = mean # get rid of outliers
        mag = max(0.3, np.abs(rep_scores).std() / 10)
        min_val, max_val = -mag, mag
        norm = Normalize(vmin=min_val, vmax=max_val)

        if "mean" in n_style:
            rep_scores = rep_scores - THRESHOLD # change this for threshold
            rep_scores = rep_scores / np.std(rep_scores[5:])
            rep_scores = np.clip(rep_scores, -mag, mag)
        if "flip" in n_style:
            rep_scores = -rep_scores
        
        rep_scores[np.abs(rep_scores) < 0.0] = 0

        # ofs = 0
        # rep_scores = np.array([rep_scores[max(0, i-ofs):min(len(rep_scores), i+ofs)].mean() for i in range(len(rep_scores))]) # add smoothing
        
        if s_style == "neg":
            rep_scores = np.clip(rep_scores, -np.inf, 0)
            rep_scores[rep_scores == 0] = mag
        elif s_style == "pos":
            rep_scores = np.clip(rep_scores, 0, np.inf)


        # Initialize positions and maximum line width
        x, y = x_start, y_start
        max_line_width = xlim
        started = False
            
        for word, score in zip(words[5:], rep_scores[5:]):

            if start_answer_token in word:
                started = True
                continue
            if not started:
                continue
            
            color = colormap(norm(score))

            # Check if the current word would exceed the maximum line width
            if x + word_width > max_line_width:
                # Move to next line
                x = x_start
                y -= 1 # Line spacing

            # Compute the width of the current word
            text = ax.text(x, y, word, fontsize=13)
            word_width = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width
            word_height = text.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).height

            # Remove the previous text
            if iter:
                text.remove()

            # Add the text with background color
            text = ax.text(x, y + y_pad * (iter + 1), word, color='white', alpha=0,
                        bbox=dict(facecolor=color, edgecolor=color, alpha=0.8, boxstyle=f'round,pad=0', linewidth=0),
                        fontsize=13)
            
            # Update the x position for the next word
            x += word_width + 0.1
        
        iter += 1


def plot_lat_scans(input_ids, rep_reader_scores_dict, layer_slice):
    for rep, scores in rep_reader_scores_dict.items():

        start_tok = input_ids.index('▁A')
        print(start_tok, np.array(scores).shape)
        standardized_scores = np.array(scores)[start_tok:start_tok+40,layer_slice]
        # print(standardized_scores.shape)

        bound = np.mean(standardized_scores) + np.std(standardized_scores)
        bound = 2.3

        # standardized_scores = np.array(scores)
        
        threshold = 0
        standardized_scores[np.abs(standardized_scores) < threshold] = 1
        standardized_scores = standardized_scores.clip(-bound, bound)
        
        cmap = 'coolwarm'

        fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
        sns.heatmap(-standardized_scores.T, cmap=cmap, linewidth=0.5, annot=False, fmt=".3f", vmin=-bound, vmax=bound)
        ax.tick_params(axis='y', rotation=0)

        ax.set_xlabel("Token Position")#, fontsize=20)
        ax.set_ylabel("Layer")#, fontsize=20)

        # x label appear every 5 ticks

        ax.set_xticks(np.arange(0, len(standardized_scores), 5)[1:])
        ax.set_xticklabels(np.arange(0, len(standardized_scores), 5)[1:])#, fontsize=20)
        ax.tick_params(axis='x', rotation=0)

        ax.set_yticks(np.arange(0, len(standardized_scores[0]), 5)[1:])
        ax.set_yticklabels(np.arange(20, len(standardized_scores[0])+20, 5)[::-1][1:])#, fontsize=20)
        ax.set_title("LAT Neural Activity")#, fontsize=30)
    plt.show()

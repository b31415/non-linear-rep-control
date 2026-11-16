import re
import json
from dataclasses import dataclass
import numpy as np
import warnings
from typing import Callable

from repe.rep_readers_non_linear import DEFAULT_CONCEPT_PROMPT_TEMPLATE, TaylorRep
from examples.honesty.rep_manager_utils import RepManager

LAYER_WEIGHTS = {
    i: 1 for i in range(-5, -26, -1)
}


class RepConfig:
    def __init__(self,
        token_level: bool = False,
        dataset_fraction: float = 1,
        eval_dataset: str | None = None,
        eval_dir: str = "./evaluate",
        max_eval_instances: int = 200,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        model_type: str = "text",
        use_dpo_adapter: bool = False,
        zero_shot: bool = True,
        apply_concept_to_base_prompt: bool = False,
        control_mode: bool = True,
        hidden_layers: tuple = (-5, -25, -1),#(-1, -32, -2),
        rep_reader_kwargs: dict = {"n_components": 2},
        max_new_tokens: int = 15,
        representation_method: str = ("none", None),
        image_generation_kwargs: dict | None = None
    ):
        """
        Args:
            representation_method: method_str or (method_str, controller_kwargs)
            model_type: "text".
            apply_concept_to_base_prompt:
                If True, when not using a rep method (i.e. using a model to generate text directly),
                prepends a concept instruction to the prompt.
                Ignored when using a representation method.
            control_mode: Use for control benchmarks (e.g. BBQ, RealToxicityPrompts, etc.),
                even when not using a representation for control, since the same 
                interface is used. Adds control functionality to models for rep methods.
                For HELM classification benchmarks (e.g. civil_comments), 
                use control_mode=True if generating a classification label via text generation 
                rather than directly using a classification model (e.g. converting a numerical 
                value to a classification label)
            rep_reader_kwargs:
                When using the surface representation method, rep_reader_kwargs must contain
                the key "n_components".
                When using "taylor" or "least-squares", must contain the key "order". See RepInferenceConfig.
        Attributes:
            repe_mode: Whether the current method is a representation engineering method (e.g.,
                repe_mode is False when controlling only using prompts).
        """
        # Use either "none" (vanilla model with no RepE method) or a method supported by RepManager()
        # Evaluation args
        self.token_level = token_level
        self.dataset_fraction = dataset_fraction
        self.eval_dataset = eval_dataset
        self.eval_dir = eval_dir

        # Number of test samples per dataset
        self.max_eval_instances = max_eval_instances
        
        # Which model and method to use
        self.model_name = model_name
        self.model_type = model_type
        if model_name == "mistralai/Mistral-7B-Instruct-v0.2" and model_type != "text":
            raise ValueError(f"Invalid model_type for model_name={model_name}")
        self.use_dpo_adapter = use_dpo_adapter

        # Args for using no representation
        self.zero_shot = zero_shot # Set to False for 6-shot
        self.apply_concept_to_base_prompt = apply_concept_to_base_prompt

        # Representation-specific args
        self.control_mode = control_mode
        self.hidden_layers = hidden_layers
        self.rep_reader_kwargs = rep_reader_kwargs
        self.max_new_tokens = max_new_tokens

        # Derived values
        self.adapter_path: str | None = "./examples/honesty/models/mistral/truthy-dpo" if use_dpo_adapter else None
        # Call setter during construction
        # Should be an element of RepManager.METHOD_CHOICES
        self.representation_method: str = representation_method
        if self.representation_method == "surface" and "n_components" not in self.rep_reader_kwargs.keys():
            raise ValueError("Argument 'rep_reader_kwargs' is missing key 'n_components'.")
        if self.representation_method in ["taylor", "least-squares"] and "order" not in self.rep_reader_kwargs.keys():
            raise ValueError("Argument 'rep_reader_kwargs' is missing key 'n_components'.")
        
                
        ### Kwargs specific to image generation models
        self.image_generation_kwargs = {} if image_generation_kwargs is None else image_generation_kwargs
            
    def __repr__(self):
        # Convert dict containing all attributes into
        # dict contain only arguments for self.__init__()
        # so eval() can be used
        kwargs = self.__dict__.copy()
        rep_method_tuple = (kwargs.pop('_representation_method'), kwargs.pop('_controller_kwargs'))
        kwargs['representation_method'] = rep_method_tuple

        kwargs.pop('adapter_path')

        # Convert dict to "RepConfig(...)" and remove any remaining variables starting with '_'
        kwargs_str = ', '.join([f'{k}={v!r}' for k, v in kwargs.items() if k[0] != '_'])
        return f"RepConfig({kwargs_str})"

    @property
    def representation_method(self):
        return self._representation_method
    
    # Automatically recompute values that depend on <representation_method>
    @representation_method.setter
    def representation_method(self, method: str | tuple[str | dict]):
        """
        Args:
            method:
                method_str: A string in RepManager.METHOD_CHOICES,
                    ignoring capitalisation and spaces (converted to '-').
                    controller_kwargs will be set to the default for that method
                OR
                tuple containing:
                    method_str
                    controller_kwargs: Kwargs to pass to the constructor
        """
        if not isinstance(method, tuple):
            method = (method, None)

        method_str, kwargs = method
        method_str = method_str.lower().replace(' ', '-')

        self._representation_method = method_str
        self._repe_mode = (self._representation_method in RepManager.REP_METHOD_CHOICES)

        # Set kwargs for controller
        if kwargs is None:
            if method_str == "contrast-vector":
                kwargs = dict(control_strength_scale=(-1, 1))
            else:
                kwargs = dict()
        self._controller_kwargs = kwargs

    # Prevent setting <repe_mode> directly since it is derived from <representation_method>
    @property
    def repe_mode(self):
        return self._repe_mode
    
    @property
    def controller_kwargs(self):
        return self._controller_kwargs

@dataclass
class BenchmarkMetadata:
    name: str | None = None
    concept_words: tuple = ('honest', 'dishonest', 'tell the truth')
    is_multi_choice: bool = False
    supports_few_shot: bool = False

class RepInferenceConfig:
    """
    Sets up the arguments to be passed to NonLinearControlMistralForCausalLM.generate()
    """

    def __init__(self,
            representation_method: str,
            benchmark: str | BenchmarkMetadata,
            control_strength: float = 3,
            concept_template_fraction: float = 0.6,
            order: int | None = None,
            n_variations_per_concept_level: int = 1,
            control_func_kwargs: dict = {"layer_weights": LAYER_WEIGHTS},
            remove_instructions: bool = True
        ):
        """
        Args:
            representation_method: E.g. "none", "surface"
            benchmark: Benchmark metadata. If str, the name of a known
                HELM benchmark to be converted to BenchmarkMetadata.
            control_strength: Positive values increase the amount of the concept
                in the response and negative values decrease it. Defaults to
                maximum concept strength (using an arbitrarily large number,
                resulting in finding the closest concept strength on the RBF to this number)
            concept_template_fraction: Use this fraction of the concept levels in the
                concept template. See _generate_concept_template_subset.
            order: Order of the Taylor or least squares approximation using 
            representation_method "taylor" or "least-squares", respectively.
            control_func_kwargs: Kwargs to pass to <RepClass>:control_layer() (e.g. SurfaceRep:control_layer()), excluding control_strength/new_score
            remove_instructions: Supported only for "taylor" and "least-squares".
                Whether to convert generated prompts from the form
                [INST] <INSTRUCTION> <PROMPT> [/INST] <RESPONSE> to 
                [INST] <PROMPT> [/INST] <RESPONSE>

        Args specific to non-linear methods (ignored for "contrast-vector" and no representation):
            concept_template_fraction: Set to less than 1 to use a subset of the level of
                granularities of the default concept template to increase speed. Only applies to least-squares and surface.
            n_variations_per_concept_level: Number of responses to generate per
                concept level instruction.
        """
        if "control_strength" in control_func_kwargs.keys() or "new_score" in control_func_kwargs.keys():
            raise ValueError("Duplicate key. Specify desired concept score using control_strength argument rather than control_func_kwargs.")
        if representation_method in ["taylor", "least-squares"] and order is None:
            raise ValueError(f"{representation_method} method requires passing 'order' argument")
        if "layer_weights" not in control_func_kwargs.keys():
            control_func_kwargs["layer_weights"] = LAYER_WEIGHTS

        ### Non-linear control args ###
        if representation_method in RepManager.NON_LINEAR_METHODS:
            self.train_rep_reader_kwargs = {
                "n_variations_per_concept_level": n_variations_per_concept_level
            }
            if representation_method in ["taylor", "least-squares"]:
                self.order = order
                self.train_rep_reader_kwargs["remove_instructions"] = remove_instructions
            if representation_method != "taylor":
                self.concept_template_fraction = concept_template_fraction
        else:
            self.train_rep_reader_kwargs = {}
            

        ### Common args ###
        # If True, a chat template is passed to the method for inference. Otherwise, a string.
        self.representation_method = representation_method
        self.control_strength = control_strength
        self.format_as_chat_template = True
        self.benchmark = benchmark if isinstance(benchmark, BenchmarkMetadata) else self._get_benchmark_metadata(benchmark)
        max_new_tokens = 1 if self.benchmark.is_multi_choice else 40
        self.control_func_kwargs = control_func_kwargs

        template = DEFAULT_CONCEPT_PROMPT_TEMPLATE
            
        self.train_rep_reader_kwargs.update({
            "model_kwargs": {
                "max_new_tokens": max_new_tokens
            },
            "concept_template_kwargs": {
                "prompt_templates": self._generate_concept_template_subset(representation_method, template),
                "concept_words": self.benchmark.concept_words
            }
        })

        self.base_model_forward_kwargs = {}


    def _generate_concept_template_subset(self, representation_method: str, template: dict) -> dict:
        """
        Generate a subset of the template according to the concept_template_fraction 
        or the order if using "taylor" methods.
        """
        def get_closest_template_index_to_concept_score(score):
            """
            Gets the index of the score in the template closest 
            to the given score, breaking
            ties by rounding towards 0 (rounding symmetrically)
            """
            template_scores = np.array(list(template.keys()))
            distance_from_score = np.abs(template_scores - score)
            closest_indices = np.where(distance_from_score == distance_from_score.min())[0]
            # Break ties
            best_index = closest_indices[np.argmin(np.abs(template_scores[closest_indices]))]
            return best_index

        # The Taylor method requires concept levels that match the Taylor coefficient
        # equation for the given order   
        if representation_method == "taylor":
            concept_levels = TaylorRep.order_to_concept_levels(self.order)

            if len(concept_levels) > len(template):
                raise ValueError("Taylor approximation order is too high for the number of prompts "\
                    f"in the template (self.order={self.order}, len(template)={len(template)})")
            
            prompts = list(template.values())
            subset = {score: prompts[get_closest_template_index_to_concept_score(score)] for score in concept_levels}

        else:
            if representation_method not in RepManager.REP_METHOD_CHOICES:
                # Get the closest matching template. I.e. If the template contains
                # templates for strengths ... 0.6, 0.8, 1 and the desired
                # strength is 0.75, get the index of 0.8 in the template.
                index = get_closest_template_index_to_concept_score(self.control_strength)
                closest_available_control_strength = list(template.keys())[index]
                if closest_available_control_strength != self.control_strength:
                    warnings.warn(f"Control strength {self.control_strength} does not have a template. Rounding to {closest_available_control_strength}")
                indices = [index]
            elif representation_method in RepManager.NON_LINEAR_METHODS:
                if representation_method == "least-squares":
                    template.pop(0, None)
                n_indices = np.round(self.concept_template_fraction * len(template)).astype(int)
                indices = np.round(
                    np.linspace(0, len(template) - 1, n_indices)
                ).astype(int)
            else:
                # Linear representations use binary concept classification
                indices = [0, -1]
            
            keys = np.array(list(template.keys()))[indices]
            subset = {k: template[k] for k in keys}
            print(f"#prompts in concept template subset: {len(subset)}")
        return subset

    
    @staticmethod
    def _get_benchmark_metadata(benchmark_name: str):
        """Get benchmark metadata by name"""
        with open("./evaluate/concept_word_presets.json") as preset_file:
            presets = json.load(preset_file)
        
        def get_concept_words(preset_name: str) -> tuple[str, str, str]:
            preset = presets[preset_name]
            return preset['concept'], preset['concept_antonym'], preset['concept_description']
        
        benchmarks = [
            BenchmarkMetadata(
                name="bbq", is_multi_choice=True,
                supports_few_shot=True,
                concept_words=get_concept_words("Bias")
            ),
            BenchmarkMetadata(
                name="real_toxicity_prompts", is_multi_choice=False,
                concept_words=get_concept_words("Toxicity")
            ),
            BenchmarkMetadata(
                name="truthful_qa", is_multi_choice=True,
                concept_words=get_concept_words("Truthfulness")
            ),
            BenchmarkMetadata(
                name="truthful_qa_generation", is_multi_choice=False,
                concept_words=get_concept_words("Truthfulness")
            )
        ]
        benchmark = [b for b in benchmarks if b.name == benchmark_name][0]
        return benchmark
    
    def __repr__(self):
        # Convert dict containing all attributes into
        # dict contain only arguments for self.__init__()
        # so eval() can be used
        kwargs = self.__dict__.copy()
        # Extract kwargs to pass to constructor of RepInferenceConfig
        # from attributes
        rep_inf_kwargs = ["n_variations_per_concept_level", "remove_instructions"]
        for key in rep_inf_kwargs:
            if key in kwargs["train_rep_reader_kwargs"].keys():
                kwargs[key] = kwargs["train_rep_reader_kwargs"][key]
        
        kwargs.pop("train_rep_reader_kwargs")
        kwargs.pop("format_as_chat_template")
        kwargs.pop("base_model_forward_kwargs")

        # Convert dict to "RepConfig(...)" and remove any remaining variables starting with '_'
        kwargs_str = ', '.join([f'{k}={v!r}' for k, v in kwargs.items() if k[0] != '_'])
        return f"RepInferenceConfig({kwargs_str})"
    
    @staticmethod
    def get_nshot_list_filter(benchmark_name: str) -> Callable:
        """
        Get function to split a prompt into a list of strings
        and filter out strings that don't satisfy the arguments
        or format for a given benchmark.
        """
        if benchmark_name == "civil_comments":
            return nshot_list_civil_comments_filter
        else:
            return nshot_list_default_filter

    @staticmethod
    def get_chat_template_converter(benchmark_name: str, classification: bool) -> Callable:
        """
        Get function to convert a prompt (as a list of few-shot examples)
        to a chat template.
        """
        if benchmark_name == "truthful_qa" and classification:
            return chat_template_converter_classify_truthfulqa
        elif classification:
            return chat_template_converter_classify_default
        else:
            return chat_template_converter_control_default

def nshot_list_default_filter(prompt: str, is_multi_choice: bool, zero_shot: bool, **_):
    prompt = prompt.split('\n\n')

    if not is_multi_choice:
        # No splitting questions or question vs answer
        # (e.g. RealToxicityPrompts requires just continuing a sentence)
        return [{"role": "assistant", "content": prompt[0]}]
    elif len(prompt) > 1:
        # Zero-shot -> Remove few-shot example in the middle        
        if zero_shot:
            prompt = [prompt[0], prompt[-1]]

        # Merge the first 2 lines into 1 line (the explanation and the first question)
        prompt[1] = '\n\n'.join(prompt[:2])
        prompt = prompt[1:]
    return prompt

def nshot_list_civil_comments_filter(prompt: list[str], zero_shot: bool, using_classification_model: bool, **_):
    """
    Examples:
        If generating True or False tokens using a baseline model,
            using_classification_model=True,
            zero_shot is optional
        If using repe_contrast_vec or surface for classification:
            using_classification_model=False,
            zero_shot=True
    """
    if not zero_shot and using_classification_model:
        raise ValueError("Classification models are applied to a passage of text, "\
            "without questions or answers, so few-shot is not applicable.")
    
    # Split passages (note some passages may contain '\n\n')
    prompt = prompt.split("\n\nPassage: ")
    # Add "Passage: " back since it was removed by splitting
    for i in range(1, len(prompt)):
        prompt[i] = "Passage: " + prompt[i]

    if zero_shot:
        prompt = [prompt[-1]]

    # Remove "\nQuestion: ...Answer: ..."
    if using_classification_model:
        prompt = [passage.split("\nQuestion: ")[0] for passage in prompt]

    return prompt

def chat_template_converter_classify_truthfulqa(prompt: list[str], qa_prefix: bool = False) -> dict[str, list[dict[str, str]]]:
    """
    Used for classification with truthful_qa benchmark

    Args:
        qa_prefix: If False, remove "Question: ", "Answer: " and "A.", "B.", "C.", etc. from the prompt.
    Returns:
        prompt_template: In the form: {
            "A": <chat template containing response for option A>},
            "B": ...,
            ...
        }
    """
    # Few-shot is not supported for classification, so the list has 1 element
    prompt = prompt[-1]
    prompt = prompt.split("\n")
    
    if not qa_prefix:
        # Remove "Question: "
        prompt[0] = prompt[0][len("Question: "):]
        #  Remove "A.", "B.", etc.
        for i in range(1, len(prompt) - 1):
            match_start, match_end = re.search("\. ", prompt[i]).span()
            prompt[i] = (prompt[i][:match_start], prompt[i][match_end:])
        # Remove "Answer:"
        prompt[-1] = ""
    else:
        # Add a space following "Answer:"
        prompt[-1] = prompt[-1] + " "

    prompt_template = {
        answer_label: [
            {"role": "user", "content": prompt[0]},
            {"role": "assistant", "content": prompt[-1] + answer_response}
        ] for answer_label, answer_response in prompt[1:-1]
    }
    return prompt_template

def chat_template_converter_classify_default(prompt: list[str]) -> list[dict]:
    """
    Used for benchmarks: civil_comments
    """
    prompt_template = []
    content = "\n\n".join(prompt)
    prompt_template.append({"role": "user", "content": ""})
    prompt_template.append({"role": "assistant", "content": content})
    return prompt_template

def chat_template_converter_control_default(prompt: list[str]) -> list[dict]:
    prompt_template = []
    prompt_examples = [example.split('Answer:') for example in prompt]

    for example, answer in prompt_examples:
        prompt_template.append({"role": "user", "content": example})
        prompt_template.append({"role": "assistant", "content": f"Answer:{answer}"})
    return prompt_template
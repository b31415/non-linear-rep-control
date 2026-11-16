"""
Loads the model and representation method specified in eval_config.py
so that it can be used for evaluation by RepClient or eval_run.py.
"""
import torch
from transformers import PreTrainedModel, AutoTokenizer

from repe.rep_control_non_linear import NonLinearControlMistralForCausalLM
from repe.rep_control_contrast_vec import (
    ContrastVecMistralForCausalLM,
    NoRepMistralForCausalLM,
    RandomMistralForCausalLM,
    NonLinearClassifierModel
)
from examples.honesty.rep_manager_utils import RepManager
from examples.honesty.rep_manager_utils import load_model as load_language_model
from evaluate.eval_config import RepConfig
from repe import repe_pipeline_registry
repe_pipeline_registry()
from functools import partial


def load_rep_method(conf: RepConfig, model = None, tokenizer = None) -> tuple[PreTrainedModel, AutoTokenizer, RepManager | None]:
    """
    Args:
        conf: Config containing args used for representation evaluation
        model: Load the controller for this preloaded model.
            If None, load the model
    Returns:
        model
        tokenizer
        rep_manager: None if repe_mode == False
    """
    if conf.model_type == "text":
        if model is None or tokenizer is None:
            model, tokenizer = load_language_model(
                model_name=conf.model_name,
                compile=(not conf.repe_mode and conf.token_level),
                flash_attn=conf.token_level,
                adapter_path=conf.adapter_path
            )
        
        # Whether to use an honesty representation rather than just the model on its own
        if conf.repe_mode:
            rep_manager = RepManager(
                model=model,
                tokenizer=tokenizer,
                method=conf.representation_method,
                rep_reader_kwargs=conf.rep_reader_kwargs,
                facts_data_path="./data/facts/facts_true_false.csv",
                prompts_data_path="./examples/honesty/data/honesty_rep_prompts.json",
                n_statements=16,
                hidden_layer_args=("uniform", conf.hidden_layers)
            )
            if conf.control_mode:
                if rep_manager.non_linear:
                    Controller = NonLinearControlMistralForCausalLM
                else:
                    Controller = ContrastVecMistralForCausalLM 
            else:
                if rep_manager.non_linear:
                    Controller = NonLinearClassifierModel
                else:
                    raise NotImplementedError("Classification using linear models is not implemented")
            model = Controller(rep_manager=rep_manager, **conf.controller_kwargs)
        else:
            rep_manager = None
            if conf.control_mode:
                if conf.representation_method == "random":
                    model = RandomMistralForCausalLM(model=model, tokenizer=tokenizer)
                else:
                    model = NoRepMistralForCausalLM(model=model, tokenizer=tokenizer)       
        
    else:
        raise ValueError(f"Unexpected value for conf.model_type: {conf.model_type}")
    
    return model, tokenizer, rep_manager




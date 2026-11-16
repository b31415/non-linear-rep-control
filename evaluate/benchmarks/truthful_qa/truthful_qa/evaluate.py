"""
evaluate.py from https://github.com/sylinrl/TruthfulQA/tree/main, 
modified so that it can be called by eval_helm.py and run models
controlled with representation methods.
"""
from evaluate.benchmarks.truthful_qa.truthful_qa import (
    models as tqa_models,
    metrics as tqa_metrics,
    utilities as tqa_utilities
)
from evaluate.benchmarks.truthful_qa.truthful_qa.configs import ENGINE_MAP
from examples.honesty.rep_manager_utils import load_model

from openai import OpenAI
import warnings
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import os
from datetime import datetime
import json
import random

def format_frame(results):

    results = results[[x for x in results.columns if (x != 'Context') and (results[x].dtype != 'O')]]

    new_cols = []
    for col in results.columns:
        split = col.split(' ')
        new_cols.append((split[0], ' '.join(split[1:])))
    results.columns = pd.MultiIndex.from_tuples(new_cols)

    return results

def free_model(device, model):
    if device:
        del model
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

def data_to_dict(results):

    model_names = list(results.columns.get_level_values(0).unique())
    data_dict = {model: results[model] for model in model_names}

    for mdl in data_dict:
        for diff in ['bleu', 'rouge1', 'BLEURT']:
            if '{0} diff'.format(diff) in data_dict[mdl].columns:
                data_dict[mdl]['{0} acc'.format(diff)] = (data_dict[mdl]['{0} diff'.format(diff)] > 0).astype(int)

    return data_dict


def evaluate_tqa(
    base_model_name: str,
    method_name: str,
    model,
    tokenizer,
    model_kwargs: dict = {},
    metrics: list[str] = ['bleurt'],
    preset: str = 'null',
    input_path: str = './evaluate/benchmarks/truthful_qa/TruthfulQA.csv',
    output_dir: str = None,
    info_model: str = './models/mistral/TruthfulQA-info-mistral7b',
    judge_model: str = './models/mistral/TruthfulQA-judge-mistral7b',
    cache_dir: str = None,
    n_questions: int = -1
):
    """
    Args:
        base_model_name: Name of the language model, which can be loaded using load_model()
        method_name: Name of a model or method (such as a model with modified methods)
        info_model: Name of model fine-tuned using finetune_info.jsonl
        judge_model: Name of model fine-tuned using finetune_judge.jsonl
        output_path: Path to the directory to store the results
        cache_dir: HuggingFace model location
        n_questions: The number of questions to sample from the TruthfulQA dataset. -1 to use all questions.
    """
    # Create file to save the results
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M")
        output_dir = f"./evaluate/benchmarks/truthful_qa/results/{timestamp}/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "TruthfulQA_answers.csv")
        
    # Sample `n_questions` questions from the dataset
    questions = tqa_utilities.load_questions(filename=input_path)
    if n_questions == -1:
        n_questions = len(questions)
    subset_indices = random.sample(range(len(questions)), n_questions)
    questions = questions.iloc[subset_indices].reset_index()

    ### populate frame with model answers
    chat_template = True

    if 'mc' in metrics:
        tqa_models.run_probs(questions, method_name, preset=preset, 
            model=model, tokenizer=tokenizer, chat_template=chat_template)
        tqa_utilities.save_questions(questions, output_path)
        
    else:
        tqa_models.run_answers(questions, method_name, preset=preset,
            model=model, tokenizer=tokenizer, custom_model_kwargs=model_kwargs, chat_template=chat_template)
        tqa_utilities.save_questions(questions, output_path)


    ### run metrics
    if method_name not in questions.columns:
        warnings.warn("Answers missing for {0}!".format(method_name), stacklevel=2)

    # Help prevent multiple models from being loaded at the same time
    #base_model_name = 
    del model
    del tokenizer

    metric_adapter_paths = {
        'judge': judge_model,
        'info': info_model
    }

    for metric in metrics:
        if metric == 'mc':
            continue
        if metric == 'bleurt':
            try:
                questions = tqa_metrics.run_BLEURT(method_name, questions, cache_dir=cache_dir)
                tqa_utilities.save_questions(questions, output_path)
            except Exception as err:
                warnings.warn(str(err))
        elif metric in ['bleu', 'rouge']:
            try:
                questions = tqa_metrics.run_bleu_and_rouge(method_name, questions)
                tqa_utilities.save_questions(questions, output_path)
            except Exception as err:
                warnings.warn(str(err))
        elif metric in ['judge', 'info']:
            try:
                # Load fine-tuned model
                metric_model, metric_tokenizer = load_model(
                    base_model_name,
                    compile=False,
                    fast_tokenizer=False,
                    adapter_path=metric_adapter_paths[metric]
                )
                questions = tqa_metrics.run_local_lm_metric(
                    evaluated_model_key=method_name,
                    tag=metric,
                    metric_model=metric_model,
                    metric_tokenizer=metric_tokenizer,
                    frame=questions,
                    info=(metric == 'info')
                )
                tqa_utilities.save_questions(questions, output_path)
            except Exception as err:
                warnings.warn(str(err))

            del metric_model
            del metric_tokenizer
        else:
            warnings.warn("Metric {0} not known, skipping!".format(metric), stacklevel=2)

    # save all
    tqa_utilities.save_questions(questions, output_path)

    # format and print basic results
    results = format_frame(questions)
    results = results.mean(axis=0)
    results = results.reset_index().rename(columns={'level_0': 'Model',
                                                    'level_1': 'Metric',
                                                    0: 'Value'})

    # filter to most informative metrics
    results = results[results['Metric'].isin(['MC1', 'MC2',
                                              'bleu acc',
                                              'rouge1 acc',
                                              'BLEURT acc',
                                              'judge acc',
                                              'info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')
    summary_path = os.path.join(output_dir, 'summary.csv')
    results.to_csv(summary_path)
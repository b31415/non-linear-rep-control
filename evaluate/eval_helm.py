"""
Evaluate using representation engineering methods for control on HELM.
"""
import os
import sys
import json
import glob
import shutil
import warnings
import argparse
import multiprocessing as mp

from copy import deepcopy
from typing import Iterator
from tabulate import tabulate
from termcolor import colored
from datetime import datetime
from evaluate.eval_config import RepConfig, RepInferenceConfig
from helm.benchmark.run import main as helm_main
from evaluate.helpers.eval_init_model import load_rep_method
from evaluate.benchmarks.truthful_qa.truthful_qa.evaluate import evaluate_tqa

# Huggingface model save directory
CACHE_DIR = os.getenv('HF_HOME')

def run_tqa_generation(
    rep_conf: RepConfig,
    rep_inference_conf: RepInferenceConfig
) -> None:
    model, tokenizer, _ = load_rep_method(rep_conf)
    model_kwargs = dict(
        control_strength=rep_inference_conf.control_strength,
        train_rep_reader_kwargs=rep_inference_conf.train_rep_reader_kwargs,
        control_func_kwargs=rep_inference_conf.control_func_kwargs
    )
    evaluate_tqa(
        base_model_name=rep_conf.model_name,
        method_name=rep_conf.representation_method,
        model=model,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
        cache_dir=CACHE_DIR,
        metrics=['judge', 'info'],
        n_questions=rep_conf.max_eval_instances
    )

def create_deployment_file(
    rep_conf: RepConfig,
    rep_inference_conf: RepInferenceConfig,
    method: str,
    model_name: str,
    all_models: list[str]
) -> None:
    """
    Create a model deployment file for HELM, with
    the config added to the args in the file so that
    the required representation method is used
    """
    deployment_template = TEXT_DEPLOYMENT_TEMPLATE
    with open(TEMP_DEPLOYMENT_FILE, "w") as f:
        # Add the template and args for the desired model
        print(
            "model_deployments:\n"
            f"  - name: nlrep/{method}\n"
            f"    model_name: {model_name}", file=f
        )
        with open(deployment_template) as template:
            shutil.copyfileobj(template, f)
        print(f'      args:', file=f)
        print(f'        rep_config: "{rep_conf}"', file=f)
        print(f'        rep_inference_config: "{rep_inference_conf}"', file=f)
        # Each model in model_metadata.yaml requires a
        # corresponding deployment so also add the
        # models that aren't being run
        for model in all_models:
            if model != model_name:
                print(f'\n  - name: {model}\n    model_name: {model}', file=f)
                with open(deployment_template) as template:
                    shutil.copyfileobj(template, f)

def get_next_method(depth: int, config_args: dict, rep_conf: dict, rep_inference_conf: dict, results_log_file: str) -> Iterator[list[int, RepConfig, RepInferenceConfig, list]]:
    """
    Iterate through the method configurations in
    `config_args` using depth-first search. If a node in the
    config args tree is not a leaf node, the args apply to all
    child nodes.

    Returns:
        depth: Number of tabs to indent the results by when logging
        rep_conf: Kwargs for eval_config:RepConfig
        rep_inference_conf: Kwargs for eval_config:RepInferenceConfig
        skip_benchmarks: Only run benchmarks in args.datasets that are not
            in skip_benchmarks for the current method
    """
    if "RepConfig" in config_args.keys():
        rep_conf.update(config_args.pop("RepConfig"))
    if "RepInferenceConfig" in config_args.keys():
        rep_inference_conf.update(config_args.pop("RepInferenceConfig"))
    if "skip_benchmarks" in config_args.keys():
        skip_benchmarks = config_args.pop("skip_benchmarks")
    else:
        skip_benchmarks = []

    # Recursion base case: Current method has no sub-methods
    # so execute benchmarks using the current configs
    if "methods" not in config_args.keys():
        yield depth + 1, rep_conf, rep_inference_conf, skip_benchmarks
    else:
        for method, sub_config_args in config_args["methods"].items():
            print(f"{4*depth*' '}- {method}", file=open(results_log_file, "a"))
            yield from get_next_method(
                depth=depth + 1,
                config_args=sub_config_args,
                rep_conf=rep_conf.copy(),
                rep_inference_conf=rep_inference_conf.copy(),
                results_log_file=results_log_file
            )

def log_curr_stats(depth: int, dataset: str, model_name: str, suite_dir: str, results_log_file: str) -> list[list[tuple]]:
    """
    Append the stats for the current run entry to RESULTS_LOG_FILE
    
    Args:
        dataset: E.g. "bbq"
        model_name: A name from model_metadata.yaml
        suite_dir: Directory of the suite for the current run
    Returns:
        tables: Shape 1 x n_metrics x 2
            where each pair is in the form (metric_name: str, value: float)
    """
    # Get stats.json files matching the current run entry
    formatted_model_name = model_name.replace('/', '_')
    stats_dir = f"{dataset}*model={formatted_model_name}"
    stats_paths = glob.glob(os.path.join(suite_dir, f"groups/{dataset}.json"))
    
    indent = 4 * depth * ' '
    tables = []
    with open(results_log_file, "a") as results_log:
        if len(stats_paths) == 0:
            print(f"{indent}No stats found for {stats_dir}", file=results_log)

        for path in stats_paths:
            # Collect the metrics from the file into a list
            table = []
            with open(path) as f:
                stats = json.load(f)[0]
                metric_names, metric_values = stats["header"], stats["rows"][0]

                for idx in range(1, len(metric_values)):
                    if "value" not in metric_values[idx].keys():
                        continue
                    table.append((metric_names[idx]["value"], round(metric_values[idx]["value"], 6)))
            
            ## Print the list of metrics as a table
            # Print the name of the run entry
            print(indent + stats_dir, file=results_log)
            # Convert list to a formatted table
            formatted_rows = tabulate(list(zip(*table)), headers="firstrow").split('\n')
            # Indent the table with `depth` tabs
            formatted_table = '\n'.join([indent + row for row in formatted_rows])
            print(formatted_table, file=results_log)
            tables.append(table)

    return tables

def get_time():
    return datetime.now().strftime(fr"%y.%m.%d%a-%H.%M.%S")

def get_suite_results_dir(suite_dir):
    """
    Args:
        suite_dir: Name of the directory to save the results. Not
            a path but can include '/'.
    Returns:
        path to suite_dir
    """
    return os.path.join(HELM_DIR, f"benchmark_output/runs/{suite_dir}")

def helm_run(args):
    sys.argv = args
    helm_main()

def run_methods_in_config(
    suite_dir: str,
    method_config: dict,
    datasets: list[str],
    benchmark_type: str
) -> tuple[list, dict[str, dict[str, list]]]:
    """
    Args:
        suite_dir: Name of the directory to save the results. Not
            a path but can include '/'.
        method_config: E.g. config_presets.json
        datasets: HELM benchmarks to run
        benchmark_type: "image" or "text" generation.
    Returns:
        suites
        all_stats: In the form {suite_name: {dataset_name: [
                [metric 1: value1, metric2: value2]
            ]}}
    """
    mp.set_start_method('spawn', force=True)

    # Path to directory where the HELM benchmark outputs will be saved
    suite_results_dir = get_suite_results_dir(suite_dir)
    results_log_file = os.path.join(suite_results_dir, "benchmark_results.txt")

    # RESULTS_LOG_FILE is used before HELM creates suite_results_dir, so make it here
    os.makedirs(suite_results_dir, exist_ok=True)

    suites = []
    all_stats = {}

    # Copy to avoid mangling the original in case it is later used by the calling function
    method_config_copy = deepcopy(method_config)
    
    for depth, rep_dict, rep_inference_dict, skip_benchmarks in \
        get_next_method(0, method_config_copy, {}, {}, results_log_file=results_log_file):
        
        rep_conf = RepConfig(**rep_dict)
        method = rep_conf.representation_method
        curr_time = get_time()
        curr_suite = f"{suite_dir}/{curr_time}"
        curr_suite_dir = os.path.join(suite_results_dir, curr_time)
        suites.append(curr_suite)
        all_stats[curr_suite] = {}

        # Get model name for model_metadata.yaml
        method_metadata = METHOD_METADATA[benchmark_type]
        if method not in method_metadata.keys():
            warnings.warn(f"No model metadata matching method '{method}' in {method_metadata.keys()}")
            model_name = method_metadata["none"]
        else:
            model_name = method_metadata[method]
        
        # Get the list of all possible model/method deployments
        all_models = [*list(METHOD_METADATA["image"].values()), *list(METHOD_METADATA["text"].values())]
        
        for dataset in datasets:
            if dataset in skip_benchmarks:
                continue

            # Add title block at the start of every dataset run
            section_title = '-------------------- NEW RUN --------------------'
            print(colored(section_title, "cyan"))
            for text, value in [("Method", method), ("Dataset", dataset)]:
                msg = f"| {text}: {value}"
                n_spaces = len(section_title) - len(msg) - 1
                msg += n_spaces * ' ' + '|'
                print(colored(msg, "cyan"))
            print(colored(len(section_title)*'-', "cyan"))

            # Create the deployment file for the current method and dataset
            rep_conf.eval_dataset = dataset
            rep_inference_conf = RepInferenceConfig(
                representation_method=rep_conf.representation_method,
                benchmark=rep_conf.eval_dataset,
                **rep_inference_dict
            )
            if dataset == "truthful_qa_generation":
                run_tqa_generation(rep_conf, rep_inference_conf)
            else:
                create_deployment_file(rep_conf, rep_inference_conf, method, model_name, all_models)

                # Delete cache between runs since the controlled model is likely to change
                prod_env_cache_dir = os.path.join(HELM_DIR, f'./prod_env/cache/')
                for file in os.listdir(prod_env_cache_dir):
                    file_path = os.path.join(prod_env_cache_dir, file)
                    if os.path.exists(file_path):
                        os.remove(file_path)

                # Call helm_run() instead of running helm-run via command line
                # to allow for putting breakpoints in HELM code
                helm_args = [
                    'helm_run',
                    '-c', os.path.join(HELM_DIR, f'./run_entries/{dataset}.conf'),
                    '-n', '1',
                    '--max-eval-instances', str(rep_conf.max_eval_instances),
                    '--models-to-run', model_name,
                    '--local-path', os.path.join(HELM_DIR, f"prod_env/"),
                    '-o', os.path.join(HELM_DIR, "benchmark_output/"),
                    '--suite', curr_suite
                ]
                #sys.argv = helm_args
                p = mp.Process(target=helm_run, args=(helm_args,))
                p.start()
                p.join()

                print("Running: " + colored(' '.join(helm_args), "green"))
                print(colored("\nRun complete. Computing stats.\n", "green"))

                # Get stats
                absolute_helm_dir = os.path.join(os.getcwd(), HELM_DIR)
                os.system(f"cd {absolute_helm_dir} && helm-summarize --suite {curr_suite}")
                # os.system(f"helm-summarize --suite {curr_suite}")
            all_stats[curr_suite][dataset] = log_curr_stats(depth, dataset, model_name, curr_suite_dir, results_log_file=results_log_file)

            # Reset
            # Delete HFServer so that it is reconstructed for the new dataset
            #clear_hf_server_cache()
    return suites, all_stats

METHOD_METADATA = {
    "text": {
        "contrast-vector": "andyzoujm/representation-engineering",
        "surface": "nlrep/surface-rep",
        "curve": "nlrep/curve-rep",
        "none": "mistralai/Mistral-7B-Instruct-v0.2",
        "random": "mistralai/Mistral-7B-Instruct-v0.2",
        "least-squares": "mistralai/Mistral-7B-Instruct-v0.2",
        "taylor": "mistralai/Mistral-7B-Instruct-v0.2"
    }
}

HELM_DIR = "./evaluate/benchmarks/helm"
TEXT_DEPLOYMENT_TEMPLATE = os.path.join(HELM_DIR, "./prod_env/text_model_deployment_template.yaml")
TEMP_DEPLOYMENT_FILE = os.path.join(HELM_DIR, "./prod_env/model_deployments.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method-config-path",
        default="./evaluate/config_presets.json",
        help="Path to a .json file relative to repository root, containing RepConfig args for all methods to run"
    )
    parser.add_argument("--datasets",
        nargs='+',
        choices=["bbq", "real_toxicity_prompts", "truthful_qa", "truthful_qa_generation"],
        help="The datasets to evaluate the method on"
    )
    args = parser.parse_args()

    # Run image and text generation benchmarks separately
    all_datasets = args.datasets
    curr_text_datasets = list(set(all_datasets))
    
    for benchmark_type, datasets in [("text", curr_text_datasets)]:
        # Clear cache, including cached generated images
        shutil.rmtree(os.path.join(HELM_DIR, "./prod_env/cache/output"), ignore_errors=True)
        for sqlite_file in glob.glob(os.path.join(HELM_DIR, f"./prod_env/cache/*.sqlite")):
            os.remove(sqlite_file)


        SUITE = f"{get_time()}_{benchmark_type}"
        if len(datasets) == 0:
            continue
        with open(args.method_config_path) as f:
                config_args = json.load(f)
        suites, _ = run_methods_in_config(suite_dir=SUITE, method_config=config_args, datasets=datasets, benchmark_type=benchmark_type)
        

        print('*'*70)
        print(colored(f"All runs are done for {benchmark_type} benchmarks. View results of the last suite with:", "green"))
        for suite in suites:
            print(colored(f"\tcd {HELM_DIR} && helm-server --suite {suite}", "green"))
        print('*'*70)




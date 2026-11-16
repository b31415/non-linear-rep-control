# Nonlinear Representation Engineering Methods
This repository demonstrates multiple nonlinear representation engineering methods, including the SurfaceRep method proposed in [Concept Control for LLM Safety Using Radial Basis Function Representations](https://link.springer.com/chapter/10.1007/978-981-95-4969-6_17).

## Introduction

## Setup
Using python3.10, run `pip install -r ./requirements.txt`.
Run the main files `./evaluate/eval_helm.py` or `./evaluate/concept_demo.py` relative to this directory, such as by opening this directory in VS Code and running using the `launch.json` file.

To be able to run the RealToxicityPrompts benchmark, follow the setup for PerspectiveAPI at https://crfm-helm.readthedocs.io/en/v0.5.4/benchmark/ using the `credentials.conf` file in the `./evaluate/benchmarks/helm/prod_env` directory.

`./evaluate/eval_helm.py` is used to collect quantitative results. Run via `python -m evaluate.eval_helm --method-config-path "./evaluate/config_presets.json" --datasets "real_toxicity_prompts" "bbq" "truthful_qa" "truthful_qa_generation"` to collect some of the quantitative results for toxicity, bias, multi-choice truthfulness and short answer truthfulness and informativeness evaluation.

The code that fits the Taylor and least squares polynomial representations is in `./repe/rep_readers_non_linear.py`. 


# References
This repository uses code from the following in `./repe`:
[1] A. Zou, L. Phan, S. Chen, et al., Representation engineering: A top-down approach to AI transparency, 2023

[2] Amos, M., Song, Y., and Pagnucco, M. Concept control using radial basis field representations. In Australasian Joint Conference on Artificial Intelligence (AJCAI), 2025.

as well as code or data from:
[3] Liang, P., et al.: Holistic evaluation of language models. Transactions on Machine Learning Research (2023)

[4] Lin, S., Hilton, J., Evans, O.: TruthfulQA: Measuring how models mimic human falsehoods. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 3214–3252 (2022)
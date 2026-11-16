"""
HELM client for evaluating Non-linear representation methods
based on https://github.com/stanford-crfm/helm/blob/main/src/helm/clients/simple_client.py
"""
from threading import Lock
from typing import Optional
import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from helm.common.cache import CacheConfig

from evaluate.benchmarks.helm.custom_hf_client import CustomHuggingFaceClient

class RepClient(CustomHuggingFaceClient):
    """Client for OpenLM: https://github.com/mlfoundations/open_lm"""

    def __init__(self, cache_config: CacheConfig, pretrained_model_name_or_path: Optional[str] = None, **kwargs):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs = {"quantization_config": quantization_config, **kwargs}
        
        super().__init__(
            cache_config=cache_config, pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs
        )
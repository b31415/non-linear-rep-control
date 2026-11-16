"""
References:
    This is a modified version of https://github.com/stanford-crfm/helm/blob/main/src/helm/clients/image_generation/huggingface_diffusers_client.py
"""
from threading import Lock
from typing import Any, Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from helm.common.cache import CacheConfig, Cache
from helm.common.file_caches.file_cache import FileCache
from helm.common.gpu_utils import get_torch_device_name, is_cuda_available
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, GeneratedOutput, wrap_request_time
from helm.common.tokenization_request import (
    DecodeRequest,
    DecodeRequestResult,
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.clients.client import Client, CachingClient
from helm.clients.image_generation.image_generation_client_utils import get_single_image_multimedia_object

from evaluate.benchmarks.helm.custom_hf_client import load_configs_from_str
from evaluate.helpers.eval_init_model import load_rep_method
from image_generation.rep_control_diffusion_models import RepControlledPipeline

_models_lock: Lock = Lock()
_models: Dict[str, Any] = {}

class CustomHuggingFaceDiffusersClient(Client):
    def __init__(self, hf_auth_token: str, cache_config: CacheConfig, file_cache: FileCache, **kwargs):
        self._hf_auth_token: str = hf_auth_token
        self._cache = Cache(cache_config)
        self._file_cache: FileCache = file_cache

        self._promptist_model = None
        self._promptist_tokenizer = None
        
        self.conf, self.inference_conf = load_configs_from_str(kwargs["rep_config"], kwargs["rep_inference_config"])
            
        with htrack_block(f"Loading Hugging Face model"):
            # Load model
            model: RepControlledPipeline
            model, _, _ = load_rep_method(self.conf)
            self.model = model

            # Need to set device for HuggingFaceServer
            self.device = model.device

    def make_request(self, request: Request) -> RequestResult:
        try:
            from diffusers import DiffusionPipeline
            from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
        except ModuleNotFoundError as e:
            handle_module_not_found_error(e, ["heim"])

        raw_request = dict(prompt=request.prompt)
        # Add representation control-specific kwargs to the
        # dict of kwargs passed to the model
        if self.conf.repe_mode:
            rep_kwargs = self.inference_conf.train_rep_reader_kwargs
            raw_request.update(dict(
                n_variations_per_concept_level=rep_kwargs['n_variations_per_concept_level'],
                control_strength=self.inference_conf.control_strength,
                **self.inference_conf.train_rep_reader_kwargs['concept_template_kwargs']
            ))

        img_kwargs = self.conf.image_generation_kwargs
        if img_kwargs is not None:
            if 'dimensions' in img_kwargs:
                if 'height' in img_kwargs['dimensions']:
                    raw_request['height'] = img_kwargs['dimensions']['height']
                if 'width' in img_kwargs['dimensions']:
                    raw_request['width'] = img_kwargs['dimensions']['width']
            if 'model_kwargs' in img_kwargs:
                if 'num_inference_steps' in img_kwargs['model_kwargs']:
                    raw_request['num_inference_steps'] = img_kwargs['model_kwargs']['num_inference_steps']
                if 'guidance_scale' in img_kwargs['model_kwargs']:
                    raw_request['guidance_scale'] = img_kwargs['model_kwargs']['guidance_scale']

        try:
            def do_it() -> Dict[str, Any]:
                prompt: str = request.prompt

                with htrack_block(f"Generating images for prompt: {prompt}"):
                    promptist_prompt: Optional[str] = None

                    images = []
                    for _ in range(request.num_completions):
                        image = self.model(**raw_request).images[0]  # type: ignore
                        images.append(image)

                    assert (
                        len(images) == request.num_completions
                    ), f"Expected {request.num_completions} images, but got {len(images)}"

                    result: Dict = {"file_locations": []}
                    if promptist_prompt is not None:
                        # Save the Promptist version of the prompts in the cache, just in case we need it later
                        result["promptist_prompt"] = promptist_prompt

                    for image in images:
                        # Write out the image to a file and save the path
                        file_location: str = self._file_cache.generate_unique_new_file_path()  # type: ignore
                        image.save(file_location)
                        hlog(f"Image saved at {file_location}")
                        result["file_locations"].append(file_location)
                    return result

            # Include the model name and number of completions in the cache key
            cache_key = CachingClient.make_cache_key(
                {"model": request.model_engine, "n": request.num_completions, **raw_request}, request
            )
            results, cached = self._cache.get(cache_key, wrap_request_time(do_it))
        except RuntimeError as ex:
            error: str = f"HuggingFaceDiffusersClient error: {ex}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = [
            GeneratedOutput(
                text="", logprob=0, tokens=[], multimodal_content=get_single_image_multimedia_object(file_location)
            )
            for file_location in results["file_locations"]
        ]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=results["request_time"],
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("This client does not support tokenizing.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("This client does not support decoding.")

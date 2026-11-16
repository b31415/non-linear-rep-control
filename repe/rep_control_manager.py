import torch
from examples.honesty.rep_manager_utils import RepManager
from repe.rep_control_non_linear import NonLinearControlMistralForCausalLM
def load_controller(rep_manager: RepManager, method: str = "surface"):
    """
    Args:
        method: Choices are "contrast-vector", "curve" and "surface
    WARNING: Deprecated
    """
    if method == "surface":
        control_model = NonLinearControlMistralForCausalLM.from_pretrained(
            rep_manager.model.name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="sequential",
            rep_manager=rep_manager
        )
    elif method == "curve":
        raise NotImplementedError("Control using method 'curve' has not been implemented")
    elif method == "contrast-vector":
        raise NotImplementedError("Control using method 'contrast-vector' has not been implemented")
    else:
        raise ValueError(f"Unexpected method {method}")
    return control_model
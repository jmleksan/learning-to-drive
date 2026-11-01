"""PyTorch device selection."""
import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Return a torch device. Use 'cuda' to require a GPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                "Check nvidia-smi in WSL and that PyTorch has a CUDA build."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device {requested!r}; use auto, cuda, or cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"

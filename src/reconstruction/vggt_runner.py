from __future__ import annotations

import importlib
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np


def enforce_offline_runtime() -> None:
    """Disable common hub clients and reject network socket connections for this process."""
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "offline"})
    if getattr(socket, "_vggt_offline_guard", False):
        return

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Network access is disabled by the offline VGGT runtime")

    socket.create_connection = blocked
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket._vggt_offline_guard = True


def gpu_report(device_request: str, allow_cpu: bool):
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device_request.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("CUDA was requested but is unavailable. Set model.allow_cpu_fallback=true to use CPU.")
        print("WARNING: CUDA unavailable; using slow CPU fallback")
        return torch.device("cpu")
    device = torch.device(device_request)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}")
        print(f"GPU memory: {props.total_memory / 2**30:.2f} GiB")
    return device


def _numpy_tree(value: Any) -> Any:
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    if isinstance(value, dict):
        return {key: _numpy_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_numpy_tree(item) for item in value)
    return value


def run_vggt(frame_paths: list[Path], model_cfg: dict[str, Any], source_path: Path,
             checkpoint: Path) -> dict[str, Any]:
    """Run a local checkout of the official VGGT package without any download fallback."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
    if not (source_path / "vggt").is_dir():
        raise FileNotFoundError(
            f"VGGT local source not found: {source_path}. Copy the official repository here; no clone is attempted.")
    enforce_offline_runtime()
    import torch
    sys.path.insert(0, str(source_path))
    try:
        VGGT = importlib.import_module("vggt.models.vggt").VGGT
        load_images = importlib.import_module("vggt.utils.load_fn").load_and_preprocess_images
        pose_decode = importlib.import_module("vggt.utils.pose_enc").pose_encoding_to_extri_intri
    finally:
        sys.path.pop(0)

    device = gpu_report(str(model_cfg.get("device", "cuda")), bool(model_cfg.get("allow_cpu_fallback", False)))
    model = VGGT()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    images = load_images([str(path) for path in frame_paths]).to(device)
    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    autocast_enabled = device.type == "cuda"
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            predictions = model(images[None])
        if not isinstance(predictions, dict):
            raise TypeError("VGGT model output is not a prediction dictionary")
        if "pose_enc" in predictions and not ({"extrinsic", "intrinsic"} <= predictions.keys()):
            extrinsic, intrinsic = pose_decode(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"], predictions["intrinsic"] = extrinsic, intrinsic
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError("CUDA out of memory. Reduce frames.num_frames and/or frames.resize_max.") from error
    result = _numpy_tree(predictions)
    result["input_images"] = images.detach().float().cpu().numpy()
    return result

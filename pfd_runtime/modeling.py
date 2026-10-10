from __future__ import annotations

import copy
import hashlib
import random
import sys

import numpy as np
import torch

from .config import ROOT, resolve

sys.path.insert(0, str(ROOT / "rtdetr_pytorch"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def construct(config):
    from src.zoo.rtdetr.pfd_model import build_model
    return build_model(config)


def criterion(config):
    from src.zoo.rtdetr.pfd_model import build_criterion
    return build_criterion(config)


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize(config, seed, pretrained=None):
    baseline_config = copy.deepcopy(config)
    baseline_config["variant"] = "baseline"
    set_seed(seed)
    baseline = construct(baseline_config)
    common = baseline.state_dict()
    checkpoint_metadata = None
    if pretrained:
        path = resolve(pretrained)
        checkpoint = torch.load(path, map_location="cpu")
        weights = checkpoint.get("ema", checkpoint.get("model", checkpoint))
        if isinstance(weights, dict) and "module" in weights:
            weights = weights["module"]
        selected = {name: value for name, value in weights.items() if name in common and value.shape == common[name].shape}
        expected = [name for name in common if not any(token in name for token in ("score_head", "denoising_class_embed"))]
        missing = [name for name in expected if name not in selected and not name.endswith("num_batches_tracked")]
        if missing:
            raise ValueError(f"COCO checkpoint does not match shared RT-DETR-R18 parameters: {missing[:8]}")
        baseline.load_state_dict(selected, strict=False)
        common = baseline.state_dict()
        reinitialized = {name: {"source_shape": list(weights[name].shape) if name in weights else None,
                                "target_shape": list(value.shape)}
                         for name, value in common.items() if name not in selected}
        checkpoint_metadata = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "loaded_tensors": len(selected), "reinitialized_tensors": reinitialized,
                               "unused_source_tensors": sorted(set(weights) - set(selected))}
    canonical_config = copy.deepcopy(config)
    canonical_config.update(variant="full", sswr={"hidden": 32, "windows": [3, 5, 9], "gate_floor": 0.1, "mode": "full"},
                            rdpf={"hidden": 64, "floor": 0.1, "mode": "full", "target": "truncated_gaussian", "loss": "balanced", "support_gain": 0.25})
    set_seed(seed + 10000)
    canonical = construct(canonical_config).state_dict()
    set_seed(seed + 20000)
    model = construct(config)
    model_state = model.state_dict()
    additions = {name: value for name, value in canonical.items() if name not in common and name in model_state and model_state[name].shape == value.shape}
    model.load_state_dict(additions, strict=False)
    shared = {}
    shared_key_mapping = {}
    for name, value in common.items():
        target_name = name
        if config.get("variant") == "p2_decoding" and name.startswith("decoder.input_proj."):
            pieces = name.split(".")
            pieces[2] = str(int(pieces[2]) + 1)
            target_name = ".".join(pieces)
        if target_name in model_state and model_state[target_name].shape == value.shape:
            shared[target_name] = value
            if target_name != name:
                shared_key_mapping[target_name] = name
    model.load_state_dict(shared, strict=False)
    metadata = {"seed": seed, "initialization": "coco" if pretrained else "random", "shared_tensors": len(shared),
                "shared_state_sha256": tensor_digest(shared), "module_reference_sha256": tensor_digest(additions),
                "shared_key_mapping": shared_key_mapping, "pretrained": checkpoint_metadata}
    set_seed(seed + 1000000)
    return model, metadata


def load_trained(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = construct(config["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), config, checkpoint

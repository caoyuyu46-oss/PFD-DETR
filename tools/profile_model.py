import argparse
import importlib.metadata
import platform

import torch
from torch import nn
from thop import profile
from thop.vision.basic_hooks import count_convNd, count_linear

from _bootstrap import ROOT
from pfd_runtime.config import load_config, resolve, save_json
from pfd_runtime.modeling import construct, set_seed


def profile_configuration(model_config, scope="conv-linear"):
    set_seed(0)
    model = construct(model_config).eval()
    stored = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    size = model_config.get("image_size", 640)
    height, width = (size, size) if isinstance(size, int) else size
    rules = None
    if scope == "conv-linear":
        rules = {type(module): None for module in model.modules()}
        rules.update({nn.Conv2d: count_convNd, nn.Linear: count_linear})
    elif scope != "thop-default":
        raise ValueError("scope must be conv-linear or thop-default")
    macs, hooked_parameters, layers = profile(model, inputs=(torch.zeros(1, 3, height, width),),
                                             custom_ops=rules, verbose=False, ret_layer_info=True)
    return {"architecture": model_config.get("architecture", "legacy_official"),
            "variant": model_config.get("variant", "full"), "model_configuration": model_config,
            "input_shape": [1, 3, height, width], "mode": "eval", "parameters": trainable,
            "stored_parameters": stored, "frozen_parameters": stored - trainable,
            "trainable_parameters": trainable, "MACs": int(macs), "GFLOPs": 2. * macs / 1e9,
            "scope": scope, "thop_hook_parameter_count_not_used": int(hooked_parameters),
            "fixed_transforms_and_functional_operators_counted": False,
            "scope_detail": ("Executed nn.Conv2d and nn.Linear hooks only; excludes normalization, activations, pooling, "
                             "fixed Haar/Sobel, local statistics, spatial rearrangement, interpolation, elementwise operations, "
                             "attention matrix products, functional projections inside MultiheadAttention, and sampling operations."
                             if scope == "conv-linear" else "THOP default module hooks; unsupported and functional operators are excluded."),
            "components": {name: {"MACs": int(value[0]), "GFLOPs": 2. * value[0] / 1e9} for name, value in layers.items()},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "thop": importlib.metadata.version("thop")},
            "accuracy_or_latency_measured": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=("conv-linear", "thop-default"), default="conv-linear")
    args = parser.parse_args()
    configuration = load_config(args.config)
    report = profile_configuration(configuration["model"], args.scope)
    save_json(resolve(args.output), report)
    print(f"Parameters: {report['parameters']:,}; GFLOPs: {report['GFLOPs']:.9f}; scope: {report['scope']}")


if __name__ == "__main__":
    main()

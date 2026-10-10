import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

from torch import nn
import torch.nn.functional as F

from .pfd_modules import PlainResidual


ROOT = Path(__file__).resolve().parents[4]


def _source(name):
    locks = json.loads((ROOT / "configs/control_sources.json").read_text(encoding="utf-8"))
    lock = next(item for item in locks if item["name"] == name)
    directory = ROOT / "external_sources" / name
    for item in lock["files"]:
        path = directory / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} source. Run python tools/download_control_sources.py --control {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
    return directory


def official_wtconv(channels, kernel_size=5, wt_levels=1, wt_type="db1", bias=True):
    directory = _source("WTConv") / "wtconv"
    for name, path in (("_pfd_wtconv", directory), ("_pfd_wtconv.util", directory / "util")):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    module = importlib.import_module("_pfd_wtconv.wtconv2d")
    return module.WTConv2d(channels, channels, kernel_size=kernel_size, stride=1,
                          bias=bias, wt_levels=wt_levels, wt_type=wt_type)


def _torch_carafe(x, mask, kernel_size, group=1, up=1):
    if group != 1:
        raise ValueError("The pinned PyTorch CARAFE fallback supports group=1")
    batch, channels, height, width = x.shape
    out_height, out_width = mask.shape[-2:]
    if (out_height, out_width) != (height * up, width * up):
        raise ValueError("CARAFE mask and feature sizes do not match")
    padded = F.pad(x, [kernel_size // 2] * 4, mode="reflect")
    patches = F.unfold(padded, kernel_size=kernel_size).reshape(
        batch, channels * kernel_size * kernel_size, height, width)
    patches = F.interpolate(patches, scale_factor=up, mode="nearest").reshape(
        batch, channels, kernel_size * kernel_size, out_height, out_width)
    return (patches * mask.reshape(batch, 1, kernel_size * kernel_size, out_height, out_width)).sum(2)


def _freqfusion_class():
    path = _source("FreqFusion") / "FreqFusion.py"
    name = "_pfd_official_freqfusion"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.carafe = _torch_carafe
        module.print = lambda *args, **kwargs: None
        sys.modules[name] = module
    return sys.modules[name].FreqFusion


class PlainFusionResidual(nn.Module):
    def __init__(self, channels=256, hidden=96, init=0.):
        super().__init__()
        self.residual = PlainResidual(channels, hidden=hidden, init=init)

    def forward(self, base, shallow, deep):
        return self.residual(base), None


class FreqFusionControl(nn.Module):
    def __init__(self, channels=256, compressed_channels=64, feature_resample=True,
                 lowpass_kernel=5, highpass_kernel=3):
        super().__init__()
        self.fusion = _freqfusion_class()(
            hr_channels=channels, lr_channels=channels, compressed_channels=compressed_channels,
            lowpass_kernel=lowpass_kernel, highpass_kernel=highpass_kernel,
            feature_resample=feature_resample, feature_resample_group=4,
            use_high_pass=True, use_low_pass=True, semi_conv=True,
            up_group=1, hamming_window=True, hr_residual=True)

    def forward(self, base, shallow, deep):
        high = F.pixel_unshuffle(shallow, 2)
        if high.shape != base.shape or high.shape[-2] != deep.shape[-2] * 2 or high.shape[-1] != deep.shape[-1] * 2:
            raise ValueError("FreqFusion control expects aligned P2/stride-8 and stride-16 features")
        _, refined_high, upsampled_deep = self.fusion(high, deep)
        return base + refined_high + upsampled_deep, None

import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pfd_runtime.config import load_config
from pfd_runtime.modeling import construct, criterion, initialize

torch.set_num_threads(2)


@pytest.mark.parametrize("name", ["wtconv", "freqfusion", "p2_decoding", "dual_capacity"])
def test_control_detection_loss_backward(name):
    if name in ("wtconv", "freqfusion"):
        source = "WTConv" if name == "wtconv" else "FreqFusion"
        if not (ROOT / "external_sources" / source).exists():
            pytest.skip("Run tools/download_control_sources.py to test optional external controls")
    config = load_config(ROOT / f"configs/pfd/mechanism/{name}.yaml")["model"]
    config.update(image_size=64, num_queries=20, num_denoising=10)
    model = construct(config).train()
    targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[.5, .5, .2, .2]])} for _ in range(2)]
    outputs = model(torch.randn(2, 3, 64, 64), targets)
    losses = criterion(config)(outputs, targets)
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    added = (model.decoder.input_proj[0] if name == "p2_decoding" else
             model.backbone.sswr if name == "wtconv" else model.encoder.rdpf)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in added.parameters())
    assert ("loss_support" in losses) == (name == "wtconv")
    model.eval()
    with torch.no_grad():
        predictions = model(torch.zeros(1, 3, 64, 64))
    assert predictions["pred_boxes"].shape == (1, 20, 4)


def test_p2_initialization_preserves_shared_scale_projection_identity():
    config = load_config(ROOT / "configs/pfd/mechanism/p2_decoding.yaml")["model"]
    config.update(image_size=64, num_queries=20, num_denoising=10)
    p2, metadata = initialize(config, 2)
    baseline_config = copy.deepcopy(config)
    baseline_config["variant"] = "baseline"
    baseline, _ = initialize(baseline_config, 2)
    for index in range(3):
        for key, value in baseline.decoder.input_proj[index].state_dict().items():
            torch.testing.assert_close(p2.decoder.input_proj[index + 1].state_dict()[key], value, rtol=0, atol=0)
    assert metadata["shared_key_mapping"]
    assert all(target.split(".")[2] == str(int(source.split(".")[2]) + 1)
               for target, source in metadata["shared_key_mapping"].items())
    assert p2.decoder.input_proj[0][0].in_channels == 64


def test_wtconv_fixed_filters_excluded_from_learnable_parameters():
    if not (ROOT / "external_sources/WTConv").exists():
        pytest.skip("WTConv optional source is not downloaded")
    config = load_config(ROOT / "configs/pfd/mechanism/wtconv.yaml")["model"]
    module = construct(config).backbone.sswr
    assert sum(p.numel() for p in module.parameters() if p.requires_grad) == 16768
    assert sum(p.numel() for p in module.parameters() if not p.requires_grad) == 4096


def test_dual_capacity_and_p2_are_independent_baseline_controls():
    base = construct({"architecture": "paper_v716", "variant": "baseline"})
    for name, delta in (("p2_decoding", 90912), ("dual_capacity", 64005)):
        configuration = load_config(ROOT / f"configs/pfd/mechanism/{name}.yaml")["model"]
        model = construct(configuration)
        count = lambda net: sum(p.numel() for p in net.parameters() if p.requires_grad)
        assert count(model) - count(base) == delta

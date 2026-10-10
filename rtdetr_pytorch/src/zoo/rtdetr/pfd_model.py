# Paper compatibility adaptations reference Ultralytics 8.0.201, AGPL-3.0.
# Modified for the PFD-DETR distribution; see NOTICE and LICENSE.
import random

from torch import nn
import torch.nn.functional as F

from src.core import register
from src.nn.backbone.pfd_resnet import PFDResNet
from .matcher import HungarianMatcher
from .pfd_criterion import PFDCriterion
from .pfd_decoder import PFDPaperTransformer
from .pfd_encoder import PFDHybridEncoder
from .rtdetr import RTDETR
from .rtdetr_decoder import RTDETRTransformer


@register
class PFDDETR(RTDETR):
    __inject__ = ["backbone", "encoder", "decoder"]

    def __init__(self, backbone, encoder, decoder, multi_scale=None):
        super().__init__(backbone, encoder, decoder, multi_scale)

    def forward(self, images, targets=None):
        if self.multi_scale and self.training:
            size = random.choice(self.multi_scale)
            images = F.interpolate(images, size=(size, size))
        features, support = self.encoder(self.backbone(images))
        outputs = self.decoder(features, targets)
        if support is not None and self.training:
            outputs["support_logits"] = support
        return outputs


def _configuration(config):
    config = dict(config or {})
    if config.get("architecture", "legacy_official") not in ("legacy_official", "paper_v716"):
        raise ValueError("architecture must be legacy_official or paper_v716")
    variant = str(config.get("variant", "full")).lower().replace("-", "_")
    if variant == "sswrsupport":
        variant = "sswr_support"
    if variant not in ("baseline", "sswr", "support", "sswr_support", "rdpf", "full", "p2_decoding", "dual_capacity"):
        raise ValueError(f"Unsupported model variant: {variant}")
    sswr = dict(config.get("sswr", {}))
    rdpf = dict(config.get("rdpf", {}))
    if variant in ("support", "sswr_support"):
        rdpf["mode"] = "aux_only"
    if variant == "dual_capacity":
        sswr["mode"] = "plain_residual"
        rdpf["mode"] = "plain_residual"
    return config, variant, sswr, rdpf


def build_model(config=None):
    config, variant, sswr, rdpf = _configuration(config)
    for key in ("target", "loss", "support_gain"):
        rdpf.pop(key, None)
    if rdpf.get("mode") == "no_support_loss":
        rdpf["mode"] = "full"
    if "windows" in sswr:
        sswr["windows"] = tuple(sswr["windows"])
    image_size = config.get("image_size", 640)
    spatial_size = [image_size, image_size] if isinstance(image_size, int) else list(image_size)
    if len(spatial_size) != 2 or any(size < 32 or size % 32 for size in spatial_size):
        raise ValueError("image_size must contain positive multiples of 32")
    backbone = PFDResNet(enable_sswr=variant in ("sswr", "sswr_support", "full", "dual_capacity"),
                         sswr_config=sswr, freeze_norm=config.get("freeze_norm", False),
                         pretrained=False)
    encoder = PFDHybridEncoder(enable_rdpf=variant in ("support", "sswr_support", "rdpf", "full", "dual_capacity"),
                                rdpf_config=rdpf, eval_spatial_size=spatial_size,
                                p2_decoding=variant == "p2_decoding")
    paper = config.get("architecture", "legacy_official") == "paper_v716"
    decoder_class = PFDPaperTransformer if paper else RTDETRTransformer
    decoder = decoder_class(num_classes=config.get("num_classes", 1),
                                feat_channels=[64, 256, 256, 256] if variant == "p2_decoding" else [256, 256, 256],
                                feat_strides=[4, 8, 16, 32] if variant == "p2_decoding" else [8, 16, 32],
                                hidden_dim=256, num_levels=4 if variant == "p2_decoding" else 3,
                                num_queries=config.get("num_queries", 300),
                                num_decoder_layers=config.get("decoder_layers", 3),
                                num_denoising=config.get("num_denoising", 100),
                                eval_spatial_size=spatial_size)
    model = PFDDETR(backbone, encoder, decoder, config.get("multi_scale"))
    if paper:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3
                layer.momentum = 0.03
    return model


def build_criterion(config=None):
    config, variant, _, rdpf = _configuration(config)
    gain = float(rdpf.get("support_gain", 0.25))
    if variant in ("baseline", "sswr", "p2_decoding", "dual_capacity") or rdpf.get("mode") in ("no_support_loss", "freqfusion", "plain_residual"):
        gain = 0.
    matcher = HungarianMatcher({"cost_class": 2., "cost_bbox": 5., "cost_giou": 2.},
                                use_focal_loss=True, alpha=0.25, gamma=2.)
    return PFDCriterion(matcher, {"loss_vfl": 1., "loss_bbox": 5., "loss_giou": 2.},
                         num_classes=config.get("num_classes", 1), support_gain=gain,
                         support_target=rdpf.get("target", "truncated_gaussian"),
                         support_loss_type=rdpf.get("loss", "balanced"),
                         paper_compatible=config.get("architecture", "legacy_official") == "paper_v716")

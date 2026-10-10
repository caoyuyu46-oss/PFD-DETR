import torch
from torch import nn
import torch.nn.functional as F

from src.core import register
from src.nn.backbone.pfd_modules import RDPF
from src.nn.backbone.pfd_controls import FreqFusionControl, PlainFusionResidual
from .hybrid_encoder import HybridEncoder


@register
class PFDHybridEncoder(HybridEncoder):
    def __init__(self, in_channels=(128, 256, 512), feat_strides=(8, 16, 32),
                 hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0.,
                 enc_act="gelu", use_encoder_idx=(2,), num_encoder_layers=1,
                 pe_temperature=10000, expansion=0.5, depth_mult=1., act="silu",
                 eval_spatial_size=None, enable_rdpf=True, rdpf_config=None, p2_decoding=False):
        super().__init__(list(in_channels), list(feat_strides), hidden_dim, nhead,
                         dim_feedforward, dropout, enc_act, list(use_encoder_idx),
                         num_encoder_layers, pe_temperature, expansion, depth_mult,
                         act, eval_spatial_size)
        self.enable_rdpf = enable_rdpf
        self.p2_decoding = p2_decoding
        self.rdpf_mode = dict(rdpf_config or {}).get("mode", "full")
        self.cam_fusion = nn.Identity()
        if enable_rdpf:
            configuration = dict(rdpf_config or {})
            if self.rdpf_mode == "plain_residual":
                self.rdpf = PlainFusionResidual(hidden_dim, hidden=configuration.get("plain_hidden", 96))
            elif self.rdpf_mode == "freqfusion":
                self.rdpf = FreqFusionControl(hidden_dim,
                    compressed_channels=configuration.get("compressed_channels", 64),
                    feature_resample=configuration.get("feature_resample", True),
                    lowpass_kernel=configuration.get("lowpass_kernel", 5),
                    highpass_kernel=configuration.get("highpass_kernel", 3))
            else:
                self.rdpf = RDPF(channels=hidden_dim, deep_channels=hidden_dim, **configuration)

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError("PFDHybridEncoder expects P2, P3, P4, P5")
        shallow, *encoder_feats = feats
        projected = [self.input_proj[i](feature) for i, feature in enumerate(encoder_feats)]
        if self.num_encoder_layers > 0:
            for i, index in enumerate(self.use_encoder_idx):
                h, w = projected[index].shape[-2:]
                sequence = projected[index].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    position = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(sequence.device)
                else:
                    position = getattr(self, f"pos_embed{index}", None)
                encoded = self.encoder[i](sequence, pos_embed=position)
                projected[index] = encoded.permute(0, 2, 1).reshape(
                    -1, self.hidden_dim, h, w).contiguous()
        inner = [projected[-1]]
        support = None
        for index in range(len(self.in_channels) - 1, 0, -1):
            layer = len(self.in_channels) - 1 - index
            high = self.lateral_convs[layer](inner[0])
            inner[0] = high
            upsampled = F.interpolate(high, scale_factor=2., mode="nearest")
            concatenated = torch.cat((upsampled, projected[index - 1]), dim=1)
            if layer == 0:
                concatenated = self.cam_fusion(concatenated)
            fused = self.fpn_blocks[layer](concatenated)
            if index == 1 and self.enable_rdpf:
                fused, support = self.rdpf(fused, shallow, high if self.rdpf_mode == "freqfusion" else upsampled)
            inner.insert(0, fused)
        outputs = [inner[0]]
        for index in range(len(self.in_channels) - 1):
            downsampled = self.downsample_convs[index](outputs[-1])
            outputs.append(self.pan_blocks[index](torch.cat((downsampled, inner[index + 1]), dim=1)))
        return ([shallow, *outputs] if self.p2_decoding else outputs), support

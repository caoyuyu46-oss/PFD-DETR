import torch.nn.functional as F

from src.core import register
from .presnet import PResNet
from .pfd_modules import SSWR, PlainResidual
from .pfd_controls import official_wtconv


@register
class PFDResNet(PResNet):
    def __init__(self, depth=18, variant="d", num_stages=4, return_idx=(0, 1, 2, 3),
                 act="relu", freeze_at=-1, freeze_norm=False, pretrained=False,
                 enable_sswr=True, sswr_config=None):
        if depth != 18 or num_stages != 4 or tuple(return_idx) != (0, 1, 2, 3):
            raise ValueError("PFDResNet requires ResNet-18 and returns P2, P3, P4, P5")
        super().__init__(depth, variant, num_stages, list(return_idx), act,
                         freeze_at, freeze_norm, pretrained)
        self.enable_sswr = enable_sswr
        if enable_sswr:
            config = dict(sswr_config or {})
            if config.get("mode") == "plain_residual":
                self.sswr = PlainResidual(128, hidden=config.get("plain_hidden", config.get("hidden", 41)),
                                          init=config.get("init", 0.))
            elif config.get("mode") == "wtconv":
                self.sswr = official_wtconv(128, kernel_size=config.get("kernel_size", 5),
                                           wt_levels=config.get("wt_levels", 1),
                                           wt_type=config.get("wt_type", "db1"),
                                           bias=config.get("bias", True))
            else:
                self.sswr = SSWR(128, **config)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=3, stride=2, padding=1)
        outputs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx == 1 and self.enable_sswr:
                x = self.sswr(x)
            outputs.append(x)
        return outputs

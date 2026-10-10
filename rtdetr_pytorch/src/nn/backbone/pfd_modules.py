import math

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )


def pad_even(x):
    h, w = x.shape[-2:]
    return F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")


def replicate_pool(x, window):
    p = window // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="replicate"), window, stride=1)


def down2(x):
    return F.avg_pool2d(pad_even(x), 2, stride=2)


def spatial_permutation(x, seed, stream=0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1000003 * int(stream))
    permutation = torch.randperm(x.shape[-2] * x.shape[-1], generator=generator).to(x.device)
    return x.flatten(2).index_select(2, permutation).reshape_as(x)


class Haar2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernels = torch.tensor([
            [[1., 1.], [1., 1.]],
            [[1., 1.], [-1., -1.]],
            [[1., -1.], [1., -1.]],
            [[1., -1.], [-1., 1.]],
        ])[:, None] / 2.
        self.channels = channels
        self.register_buffer("kernels", kernels.repeat(channels, 1, 1, 1))

    def analysis(self, x):
        h, w = x.shape[-2:]
        bands = F.conv2d(pad_even(x), self.kernels.to(dtype=x.dtype), stride=2,
                         groups=self.channels)
        return bands.reshape(x.shape[0], self.channels, 4, (h + 1) // 2, (w + 1) // 2), (h, w)

    def synthesis(self, bands, spatial_size):
        b, c, _, h, w = bands.shape
        x = F.conv_transpose2d(bands.reshape(b, c * 4, h, w),
                              self.kernels.to(dtype=bands.dtype), stride=2, groups=c)
        return x[..., :spatial_size[0], :spatial_size[1]]


class ScaleStructure(nn.Module):
    def __init__(self, channels, windows=(3, 5, 9), context=8,
                 scale_mode="learned", descriptor="tensor", eps=1e-6):
        super().__init__()
        self.windows = tuple(int(k) for k in windows)
        if not self.windows or any(k < 1 or k % 2 == 0 for k in self.windows):
            raise ValueError("windows must contain positive odd integers")
        if scale_mode not in ("learned", "uniform"):
            raise ValueError("scale_mode must be learned or uniform")
        if descriptor not in ("tensor", "none"):
            raise ValueError("descriptor must be tensor or none")
        self.scale_mode = scale_mode
        self.descriptor = descriptor
        self.eps = eps
        kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.
        self.register_buffer("sobel_x", kernel[None, None])
        self.register_buffer("sobel_y", kernel.t()[None, None])
        self.context = ConvBNAct(channels, context)
        n = len(self.windows)
        self.scale_logits = nn.Conv2d(context + 2 * n, n, 1)
        self.output_channels = context + 2 + n
        self.capture = False
        self.diagnostics = {}
        self.intervention = "none"
        self.intervention_seed = 0
        self.intervention_shift = (0, 1)

    def statistics(self, u):
        with torch.autocast(device_type=u.device.type, enabled=False):
            x = u.float()
            c = x.shape[1]
            padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
            gx = F.conv2d(padded, self.sobel_x.float().repeat(c, 1, 1, 1), groups=c)
            gy = F.conv2d(padded, self.sobel_y.float().repeat(c, 1, 1, 1), groups=c)
            xx = gx.square().mean(1, keepdim=True)
            yy = gy.square().mean(1, keepdim=True)
            xy = (gx * gy).mean(1, keepdim=True)
            maps = []
            for window in self.windows:
                jxx, jyy, jxy = (replicate_pool(t, window) for t in (xx, yy, xy))
                trace = jxx + jyy
                gap = ((jxx - jyy).square() + 4 * jxy.square() + self.eps ** 2).sqrt() - self.eps
                coherence = (gap / (trace + self.eps)).clamp(0., 1.)
                energy = trace / (trace + trace.mean((-2, -1), keepdim=True) + self.eps)
                maps.extend((down2(coherence), down2(energy)))
            descriptors = torch.cat(maps, dim=1)
            if self.descriptor == "none":
                descriptors = descriptors * 0.
        return descriptors.to(dtype=u.dtype)

    def forward(self, u, low):
        descriptors = self.statistics(u)
        if self.intervention == "structure_zero":
            descriptors = descriptors * 0.
        context = self.context(low)
        logits = self.scale_logits(torch.cat((context, descriptors), dim=1))
        if self.scale_mode == "uniform" or self.intervention == "scale_uniform":
            weights = logits * 0. + 1. / len(self.windows)
        else:
            weights = logits.softmax(dim=1)
        if self.intervention == "scale_shuffle":
            weights = spatial_permutation(weights, self.intervention_seed)
        elif self.intervention == "scale_shift":
            weights = weights.roll(shifts=self.intervention_shift, dims=(-2, -1))
        coherence = (weights * descriptors[:, 0::2]).sum(1, keepdim=True)
        energy = (weights * descriptors[:, 1::2]).sum(1, keepdim=True)
        if self.capture:
            self.diagnostics = {"scale_weights": weights.detach(),
                                "coherence": coherence.detach(), "energy": energy.detach(),
                                "all_descriptors": descriptors.detach()}
        return torch.cat((context, coherence, energy, weights), dim=1)


class SSWR(nn.Module):
    def __init__(self, channels=128, hidden=32, windows=(3, 5, 9), gate_floor=0.1,
                 mode="full", scale_mode="learned", descriptor="tensor", init=0.):
        super().__init__()
        if mode not in ("full", "uniform", "single", "no_descriptors"):
            raise ValueError(f"Unsupported SSWR mode: {mode}")
        if not 0. <= gate_floor <= 1.:
            raise ValueError("gate_floor must lie in [0, 1]")
        if mode == "single":
            windows = (5,)
        elif mode == "uniform":
            scale_mode = "uniform"
        elif mode == "no_descriptors":
            descriptor = "none"
        self.reduce = ConvBNAct(channels, hidden)
        self.wavelet = Haar2D(hidden)
        self.structure = ScaleStructure(hidden, windows, scale_mode=scale_mode,
                                        descriptor=descriptor)
        self.gate = nn.Conv2d(self.structure.output_channels, hidden * 3, 1)
        self.low_filter = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False)
        self.high_filter = nn.Conv2d(hidden * 3, hidden * 3, 3, padding=1,
                                     groups=hidden * 3, bias=False)
        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), float(init)))
        self.gate_floor = gate_floor
        self.intervention = "none"
        self.intervention_seed = 0
        self.intervention_shift = (0, 1)
        self.capture = False
        self.diagnostics = {}

    def forward(self, x):
        if self.intervention == "bypass":
            return x
        u = self.reduce(x)
        bands, spatial_size = self.wavelet.analysis(u)
        b, c, _, h, w = bands.shape
        self.structure.capture = self.capture
        structure_interventions = ("structure_zero", "scale_uniform", "scale_shuffle", "scale_shift")
        self.structure.intervention = self.intervention if self.intervention in structure_interventions else "none"
        self.structure.intervention_seed = self.intervention_seed
        self.structure.intervention_shift = self.intervention_shift
        condition = self.structure(u, bands[:, :, 0])
        gate = self.gate_floor + (1. - self.gate_floor) * self.gate(condition).sigmoid()
        if self.intervention == "gate_mean":
            gate = gate.mean((-2, -1), keepdim=True).expand_as(gate)
        elif self.intervention == "gate_one":
            gate = torch.ones_like(gate)
        elif self.intervention not in ("none", *structure_interventions):
            raise ValueError(f"Unsupported SSWR intervention: {self.intervention}")
        low = self.low_filter(bands[:, :, 0])
        high = self.high_filter(bands[:, :, 1:].reshape(b, c * 3, h, w)) * gate
        residual_bands = torch.cat((low.unsqueeze(2), high.reshape(b, c, 3, h, w)), dim=2)
        delta = self.expand(self.wavelet.synthesis(residual_bands, spatial_size))
        residual = self.alpha.tanh() * delta
        if self.capture:
            self.diagnostics = {**self.structure.diagnostics, "high_frequency_gate": gate.detach(),
                                "residual": residual.detach()}
        return x + residual


class PlainResidual(nn.Module):
    def __init__(self, channels=128, hidden=41, init=0.):
        super().__init__()
        self.reduce = ConvBNAct(channels, hidden)
        self.filter = ConvBNAct(hidden, hidden, 5, groups=hidden)
        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), float(init)))
        self.intervention = "none"
        self.intervention_seed = 0

    def forward(self, x):
        if self.intervention == "bypass":
            return x
        return x + self.alpha.tanh() * self.expand(self.filter(self.reduce(x)))


def protective_gate(shallow, deep, floor=0.1, mode="full"):
    if mode not in ("full", "ungated", "semantic", "ambiguity", "absolute", "aux_only"):
        raise ValueError(f"Unsupported RDPF gate mode: {mode}")
    u = 4. * deep * (1. - deep)
    positive_difference = (shallow - deep).relu()
    if mode == "semantic":
        mixing = torch.zeros_like(deep)
    elif mode == "ambiguity":
        mixing = u
    elif mode == "absolute":
        mixing = u + (1. - u) * (shallow - deep).abs()
    else:
        mixing = u + (1. - u) * positive_difference
    gate = floor + (1. - floor) * ((1. - mixing) * deep + mixing * shallow)
    if mode == "ungated":
        gate = torch.ones_like(gate)
    return gate, u, positive_difference, mixing


class RDPF(nn.Module):
    def __init__(self, channels=256, shallow_channels=64, deep_channels=256, hidden=64,
                 floor=0.1, mode="full", init=0.):
        super().__init__()
        if not 0. <= floor <= 1.:
            raise ValueError("floor must lie in [0, 1]")
        if mode not in ("full", "ungated", "semantic", "ambiguity", "absolute", "aux_only"):
            raise ValueError(f"Unsupported RDPF mode: {mode}")
        self.shallow = nn.Sequential(ConvBNAct(4 * shallow_channels, hidden),
                                     ConvBNAct(hidden, hidden, 3, groups=hidden))
        self.deep = ConvBNAct(deep_channels, hidden)
        self.shallow_head = nn.Conv2d(hidden, 1, 1)
        self.deep_head = nn.Conv2d(hidden, 1, 1)
        for head in (self.shallow_head, self.deep_head):
            nn.init.constant_(head.bias, math.log(0.01 / 0.99))
        self.expand = nn.Conv2d(hidden, channels, 1, bias=False)
        self.beta = nn.Parameter(torch.full((1, channels, 1, 1), float(init)))
        self.floor = floor
        self.mode = mode
        self.intervention = "none"
        self.intervention_seed = 0
        self.capture = False
        self.diagnostics = {}
        if mode == "aux_only":
            self.expand.requires_grad_(False)
            self.beta.requires_grad_(False)

    def forward(self, base, shallow, deep):
        z_shallow = self.shallow(F.pixel_unshuffle(pad_even(shallow), 2))
        if z_shallow.shape[-2:] != base.shape[-2:]:
            z_shallow = F.interpolate(z_shallow, size=base.shape[-2:], mode="bilinear",
                                      align_corners=False)
        if deep.shape[-2:] != base.shape[-2:]:
            deep = F.interpolate(deep, size=base.shape[-2:], mode="nearest")
        z_deep = self.deep(deep)
        shallow_logits = self.shallow_head(z_shallow)
        deep_logits = self.deep_head(z_deep)
        p_shallow, p_deep = shallow_logits.sigmoid(), deep_logits.sigmoid()
        support_interventions = ("shallow_support_shuffle", "deep_support_shuffle", "independent_both_shuffle")
        if self.intervention in ("shallow_support_shuffle", "independent_both_shuffle"):
            p_shallow = spatial_permutation(p_shallow, self.intervention_seed)
        if self.intervention in ("deep_support_shuffle", "independent_both_shuffle"):
            p_deep = spatial_permutation(p_deep, self.intervention_seed, stream=1)
        gate_mode = "ambiguity" if self.intervention == "ambiguity_only" else self.mode
        gate, u, difference, mixing = protective_gate(p_shallow, p_deep, self.floor, gate_mode)
        if self.intervention == "gate_mean":
            gate = gate.mean((-2, -1), keepdim=True).expand_as(gate)
        elif self.intervention == "gate_one":
            gate = torch.ones_like(gate)
        elif self.intervention not in ("none", "bypass", "ambiguity_only", *support_interventions):
            raise ValueError(f"Unsupported RDPF intervention: {self.intervention}")
        if self.mode == "aux_only" or self.intervention == "bypass":
            residual = torch.zeros_like(base)
            output = base
        else:
            residual = self.beta.tanh() * gate * self.expand(z_shallow)
            output = base + residual
        if self.capture:
            self.diagnostics = {"shallow_support": p_shallow.detach(), "deep_support": p_deep.detach(),
                                "u": u.detach(), "dpos": difference.detach(), "mixing": mixing.detach(),
                                "gate": gate.detach(), "residual": residual.detach()}
        return output, (shallow_logits, deep_logits)

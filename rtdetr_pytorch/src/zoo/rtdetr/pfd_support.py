import torch
import torch.nn.functional as F


@torch.no_grad()
def center_targets(targets, height, width, device, target="truncated_gaussian", sigma_min=1.):
    if target not in ("truncated_gaussian", "binary_box", "center_cell", "gaussian"):
        raise ValueError(f"Unsupported support target: {target}")
    if height < 1 or width < 1 or sigma_min <= 0.:
        raise ValueError("Grid dimensions and sigma_min must be positive")
    yy = torch.arange(height, device=device, dtype=torch.float32)[None, :, None] + 0.5
    xx = torch.arange(width, device=device, dtype=torch.float32)[None, None, :] + 0.5
    heatmap = torch.zeros((len(targets), 1, height, width), device=device, dtype=torch.float32)
    for batch_index, item in enumerate(targets):
        boxes = item["boxes"].detach().to(device=device, dtype=torch.float32)
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("Support targets require normalized cxcywh boxes of shape [N, 4]")
        if (not torch.isfinite(boxes).all() or (boxes < 0).any()
                or (boxes > 1).any() or (boxes[:, 2:] <= 0).any()):
            raise ValueError("Support target boxes must be finite normalized cxcywh with positive sizes")
        for box in boxes:
            center_x = (box[0] * width).floor().clamp(0, width - 1) + 0.5
            center_y = (box[1] * height).floor().clamp(0, height - 1) + 0.5
            box_width, box_height = box[2] * width, box[3] * height
            if target == "binary_box":
                response = ((xx - box[0] * width).abs() <= box_width / 2.) & (
                    (yy - box[1] * height).abs() <= box_height / 2.)
                response = response.float()
            elif target == "center_cell":
                response = ((xx == center_x) & (yy == center_y)).float()
            else:
                sigma_x = (box_width / 6.).clamp_min(sigma_min)
                sigma_y = (box_height / 6.).clamp_min(sigma_min)
                response = torch.exp(-0.5 * (((xx - center_x) / sigma_x).square()
                                            + ((yy - center_y) / sigma_y).square()))
                if target == "truncated_gaussian":
                    region = ((xx - center_x).abs() <= (box_width / 2.).clamp_min(0.5)) & (
                        (yy - center_y).abs() <= (box_height / 2.).clamp_min(0.5))
                    response = response * region
            heatmap[batch_index] = torch.maximum(heatmap[batch_index], response)
    return heatmap


def balanced_support_loss(logits, target, eps=1e-6):
    if logits.shape != target.shape:
        raise ValueError("Support logits and targets must have identical shapes")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        z, t = logits.float(), target.float()
        dims = tuple(range(1, z.ndim))
        positive_mass = t.sum(dims)
        negative_mass = (1. - t).sum(dims)
        positive = (t * F.softplus(-z)).sum(dims) / positive_mass.clamp_min(eps)
        negative = ((1. - t) * F.softplus(z)).sum(dims) / negative_mass.clamp_min(eps)
        active = (positive_mass > eps).float() + (negative_mass > eps).float()
        return ((positive + negative) / active.clamp_min(1.)).mean()


def support_loss(logits, target, loss="balanced"):
    if len(logits) != 2:
        raise ValueError("Expected shallow and deep support logits")
    if loss not in ("balanced", "mean_bce", "shallow_only", "deep_only"):
        raise ValueError(f"Unsupported support loss: {loss}")
    if loss == "shallow_only":
        selected = logits[:1]
    elif loss == "deep_only":
        selected = logits[1:]
    else:
        selected = logits
    if loss == "mean_bce":
        with torch.autocast(device_type=selected[0].device.type, enabled=False):
            values = [F.binary_cross_entropy_with_logits(z.float(), target.float()) for z in selected]
    else:
        values = [balanced_support_loss(z, target) for z in selected]
    return torch.stack(values).mean()

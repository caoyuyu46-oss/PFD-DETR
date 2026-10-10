# Ultralytics YOLO, AGPL-3.0 license.
# Paper-compatible loss branches adapted from Ultralytics 8.0.201 utils/loss.py.
# Modified for the PFD-DETR distribution; see NOTICE and LICENSES/AGPL-3.0.txt.
import torch
import torch.nn.functional as F

from src.core import register
from .box_ops import box_cxcywh_to_xyxy, box_iou
from .pfd_support import center_targets, support_loss
from .rtdetr_criterion import SetCriterion


@register
class PFDCriterion(SetCriterion):
    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(self, matcher, weight_dict, losses=("vfl", "boxes"), alpha=0.75,
                 gamma=2., eos_coef=1e-4, num_classes=1, support_gain=0.25,
                 support_target="truncated_gaussian", support_loss_type="balanced",
                 paper_compatible=False):
        super().__init__(matcher, weight_dict, list(losses), alpha, gamma, eos_coef, num_classes)
        if support_gain < 0.:
            raise ValueError("support_gain must be nonnegative")
        if support_target not in ("truncated_gaussian", "binary_box", "center_cell", "gaussian"):
            raise ValueError(f"Unsupported support target: {support_target}")
        if support_loss_type not in ("balanced", "mean_bce", "shallow_only", "deep_only"):
            raise ValueError(f"Unsupported support loss: {support_loss_type}")
        self.support_gain = support_gain
        self.support_target = support_target
        self.support_loss_type = support_loss_type
        self.paper_compatible = paper_compatible

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        if not self.paper_compatible:
            return super().loss_labels_vfl(outputs, targets, indices, num_boxes, log)
        logits = outputs["pred_logits"]
        index = self._get_src_permutation_idx(indices)
        labels = torch.zeros_like(logits)
        target_classes = torch.cat([target["labels"][matched] for target, (_, matched) in zip(targets, indices)])
        labels[index[0], index[1], target_classes] = 1.
        if len(target_classes):
            predicted_boxes = outputs["pred_boxes"][index]
            target_boxes = torch.cat([target["boxes"][matched] for target, (_, matched) in zip(targets, indices)])
            quality = box_iou(box_cxcywh_to_xyxy(predicted_boxes), box_cxcywh_to_xyxy(target_boxes))[0].diag().detach()
            scores = torch.zeros_like(logits)
            scores[index[0], index[1], target_classes] = quality
            weight = self.alpha * logits.sigmoid().pow(self.gamma) * (1. - labels) + scores
            loss = F.binary_cross_entropy_with_logits(logits.float(), scores.float(), reduction="none") * weight
        else:
            probability = logits.sigmoid()
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * probability.pow(1.5) * 0.75
        return {"loss_vfl": loss.sum() / num_boxes}

    def forward(self, outputs, targets):
        support = outputs.get("support_logits")
        detection = {key: value for key, value in outputs.items() if key != "support_logits"}
        losses = super().forward(detection, targets)
        if support is not None and self.support_gain > 0.:
            target = center_targets(targets, *support[0].shape[-2:], support[0].device,
                                    target=self.support_target)
            losses["loss_support"] = self.support_gain * support_loss(
                support, target, self.support_loss_type)
        return losses

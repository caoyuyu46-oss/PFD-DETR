# Ultralytics YOLO, AGPL-3.0 license.
# Adapted from Ultralytics 8.0.201 nn/modules/head.py and models/utils/ops.py.
# Modified for the PFD-DETR distribution; see NOTICE and LICENSES/AGPL-3.0.txt.
import math

import torch
from torch import nn

from src.core import register
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .rtdetr_decoder import RTDETRTransformer


def paper_denoising(targets, embedding, num_queries, num_denoising=100,
                    label_noise_ratio=0.5, box_noise_scale=1.0):
    sizes = [len(target["labels"]) for target in targets]
    maximum = max(sizes, default=0)
    if num_denoising <= 0 or maximum == 0:
        return None, None, None, None
    groups = max(num_denoising // maximum, 1)
    device = embedding.weight.device
    labels = torch.cat([target["labels"] for target in targets]).repeat(2 * groups)
    boxes = torch.cat([target["boxes"] for target in targets]).repeat(2 * groups, 1)
    count = sum(sizes)
    if label_noise_ratio > 0.:
        noisy = (torch.rand(labels.shape) < label_noise_ratio / 2.).nonzero().flatten()
        replacement = torch.randint(embedding.num_embeddings, noisy.shape,
                                    dtype=labels.dtype, device=device)
        labels[noisy.to(device)] = replacement
    if box_noise_scale > 0.:
        corners = box_cxcywh_to_xyxy(boxes)
        magnitude = boxes[:, 2:].repeat(1, 2) * (box_noise_scale / 2.)
        signs = torch.randint_like(boxes, 0, 2).mul(2.).sub(1.)
        noise = torch.rand_like(boxes)
        noise[count * groups:] += 1.
        corners = (corners + magnitude * signs * noise).clamp(0., 1.)
        boxes = torch.logit(box_xyxy_to_cxcywh(corners), eps=1e-6)
    denoising_count = 2 * groups * maximum
    content = embedding.weight.new_zeros(len(sizes), denoising_count, embedding.embedding_dim)
    references = boxes.new_zeros(len(sizes), denoising_count, 4)
    image_indices = torch.repeat_interleave(torch.arange(len(sizes), device=device),
                                           torch.tensor(sizes, device=device)).repeat(2 * groups)
    local_indices = torch.cat([torch.arange(size, device=device) for size in sizes])
    destinations = (local_indices[None] + maximum * torch.arange(2 * groups, device=device)[:, None]).flatten()
    content[image_indices, destinations] = embedding(labels)
    references[image_indices, destinations] = boxes
    attention = torch.zeros(denoising_count + num_queries, denoising_count + num_queries,
                            dtype=torch.bool, device=device)
    attention[denoising_count:, :denoising_count] = True
    block = torch.arange(denoising_count, device=device) // (2 * maximum)
    attention[:denoising_count, :denoising_count] = block[:, None] != block[None, :]
    positive = [(torch.arange(size, device=device)[None] + maximum * torch.arange(groups, device=device)[:, None]).flatten()
                for size in sizes]
    metadata = {"dn_positive_idx": positive, "dn_num_group": groups,
                "dn_num_split": [denoising_count, num_queries]}
    return content, references, attention, metadata


@register
class PFDPaperTransformer(RTDETRTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoising_class_embed = nn.Embedding(self.num_classes, self.hidden_dim)

    def _reset_parameters(self):
        super()._reset_parameters()
        bias = math.log(0.01 / 0.99) * self.num_classes / 80.
        nn.init.constant_(self.enc_score_head.bias, bias)
        for layer in self.dec_score_head:
            nn.init.constant_(layer.bias, bias)
        for layer in self.input_proj:
            nn.init.xavier_uniform_(layer.conv.weight)

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None,
                           denoising_bbox_unact=None):
        cached_shapes = ([[self.eval_spatial_size[0] // stride, self.eval_spatial_size[1] // stride]
                          for stride in self.feat_strides] if self.eval_spatial_size is not None else None)
        if not self.training and spatial_shapes == cached_shapes:
            anchors = self.anchors.to(device=memory.device, dtype=memory.dtype)
            valid = self.valid_mask.to(device=memory.device)
        else:
            anchors, valid = self._generate_anchors(spatial_shapes, dtype=memory.dtype,
                                                    device=memory.device)
        encoded = self.enc_output(memory * valid)
        logits = self.enc_score_head(encoded)
        indices = logits.amax(-1).topk(self.num_queries, dim=1).indices
        content = encoded.gather(1, indices[..., None].expand(-1, -1, self.hidden_dim))
        references = self.enc_bbox_head(content) + anchors.expand(len(memory), -1, -1).gather(
            1, indices[..., None].expand(-1, -1, 4))
        boxes = references.sigmoid()
        selected_logits = logits.gather(1, indices[..., None].expand(-1, -1, self.num_classes))
        if self.learnt_init_query:
            content = self.tgt_embed.weight[None].expand(len(memory), -1, -1)
        if self.training:
            references = references.detach()
            if not self.learnt_init_query:
                content = content.detach()
        if denoising_class is not None:
            content = torch.cat((denoising_class, content), dim=1)
            references = torch.cat((denoising_bbox_unact, references), dim=1)
        return content, references, boxes, selected_logits

    def forward(self, feats, targets=None):
        memory, shapes, offsets = self._get_encoder_input(feats)
        denoising = (None, None, None, None)
        if self.training and self.num_denoising > 0:
            if targets is None:
                raise ValueError("Training with denoising requires targets")
            denoising = paper_denoising(targets, self.denoising_class_embed, self.num_queries,
                                       self.num_denoising, self.label_noise_ratio, self.box_noise_scale)
        content, references, encoder_boxes, encoder_logits = self._get_decoder_input(
            memory, shapes, denoising[0], denoising[1])
        boxes, logits = self.decoder(content, references, memory, shapes, offsets,
                                     self.dec_bbox_head, self.dec_score_head, self.query_pos_head,
                                     attn_mask=denoising[2])
        metadata = denoising[3]
        if self.training and metadata is not None:
            dn_boxes, boxes = boxes.split(metadata["dn_num_split"], dim=2)
            dn_logits, logits = logits.split(metadata["dn_num_split"], dim=2)
        output = {"pred_logits": logits[-1], "pred_boxes": boxes[-1]}
        if self.training and self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(logits[:-1], boxes[:-1])
            output["aux_outputs"] += self._set_aux_loss([encoder_logits], [encoder_boxes])
            if metadata is not None:
                output["dn_aux_outputs"] = self._set_aux_loss(dn_logits, dn_boxes)
                output["dn_meta"] = metadata
        return output

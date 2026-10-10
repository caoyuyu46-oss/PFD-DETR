from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import load_config, resolve


class FruitDataset(Dataset):
    def __init__(self, data_config, split, size=640, augment=None, seed=0):
        self.config = load_config(resolve(data_config)) if isinstance(data_config, (str, Path)) else data_config
        self.root = resolve(self.config["path"])
        self.annotation_path = self.root / self.config[split]["annotations"]
        self.image_root = self.root / self.config[split]["images"]
        self.coco = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.images = sorted(self.coco["images"], key=lambda image: image["id"])
        image_ids = [image["id"] for image in self.images]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("Image identifiers must be unique")
        if any(image["width"] <= 0 or image["height"] <= 0 for image in self.images):
            raise ValueError("Image dimensions must be positive")
        categories = sorted(category["id"] for category in self.coco["categories"])
        if len(categories) != 1:
            raise ValueError("PFD-DETR configurations require exactly one foreground category")
        self.category_id = categories[0]
        self.annotations = {image["id"]: [] for image in self.images}
        image_lookup = {image["id"]: image for image in self.images}
        annotation_ids = set()
        for annotation in self.coco["annotations"]:
            if annotation["id"] in annotation_ids:
                raise ValueError("Annotation identifiers must be unique")
            annotation_ids.add(annotation["id"])
            if annotation["category_id"] != self.category_id:
                raise ValueError("An annotation references an unknown foreground category")
            if annotation["image_id"] not in image_lookup:
                raise ValueError("An annotation references an unknown image")
            if annotation.get("iscrowd", 0):
                raise ValueError("Training and operating-point evaluation require non-crowd instance boxes")
            box = np.asarray(annotation["bbox"], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
                raise ValueError("Annotations require finite xywh boxes with positive width and height")
            info = image_lookup[annotation["image_id"]]
            if box[0] >= info["width"] or box[1] >= info["height"] or box[0] + box[2] <= 0 or box[1] + box[3] <= 0:
                raise ValueError("An annotation box does not intersect its image")
            self.annotations[annotation["image_id"]].append(annotation)
        if not isinstance(size, int) or size < 32 or size % 32:
            raise ValueError("The square input size must be a positive multiple of 32")
        self.size, self.augment, self.seed, self.epoch = size, augment, seed, 0

    def __len__(self):
        return len(self.images)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        info = self.images[index]
        path = self.image_root / info["file_name"]
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        height, width = image.shape[:2]
        if (width, height) != (info["width"], info["height"]):
            raise ValueError(f"Image dimensions do not match annotations: {path}")
        records = self.annotations[info["id"]]
        boxes = np.array([record["bbox"] for record in records], dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        boxes *= np.array([self.size / width, self.size / height] * 2, np.float32)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if self.augment:
            key = f"{self.seed}:{self.epoch}:{info['id']}".encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
            image, boxes = augment_image(image, boxes, self.augment, rng)
        boxes /= self.size
        boxes[:, 2:] -= boxes[:, :2]
        boxes[:, :2] += boxes[:, 2:] / 2
        image = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float() / 255
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.zeros(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(info["id"]),
            "orig_size": torch.tensor([width, height]),
        }
        return image, target


def augment_image(image, boxes, settings, rng):
    size = image.shape[0]
    scale = rng.uniform(1 - settings["scale"], 1 + settings["scale"])
    tx, ty = rng.uniform(-settings["translate"], settings["translate"], 2) * size
    matrix = np.array([[scale, 0, tx + size * (1 - scale) / 2], [0, scale, ty + size * (1 - scale) / 2]], dtype=np.float32)
    image = cv2.warpAffine(image, matrix, (size, size), borderValue=(114, 114, 114))
    original = boxes.copy()
    boxes[:, [0, 2]] = scale * boxes[:, [0, 2]] + matrix[0, 2]
    boxes[:, [1, 3]] = scale * boxes[:, [1, 3]] + matrix[1, 2]
    boxes = boxes.clip(0, size)
    wh = boxes[:, 2:] - boxes[:, :2]
    area_ratio = wh.prod(1) / (np.maximum(original[:, 2:] - original[:, :2], 0).prod(1) * scale * scale + 1e-16)
    aspect = np.maximum(wh[:, 0] / (wh[:, 1] + 1e-16), wh[:, 1] / (wh[:, 0] + 1e-16))
    boxes = boxes[(wh > 2).all(1) & (area_ratio > 0.1) & (aspect < 100)]
    gains = rng.uniform(-1, 1, 3) * [settings["hsv_h"], settings["hsv_s"], settings["hsv_v"]] + 1
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    lut = np.arange(256, dtype=np.float32)
    hue = cv2.LUT(hue, ((lut * gains[0]) % 180).astype(np.uint8))
    saturation = cv2.LUT(saturation, np.clip(lut * gains[1], 0, 255).astype(np.uint8))
    value = cv2.LUT(value, np.clip(lut * gains[2], 0, 255).astype(np.uint8))
    image = cv2.cvtColor(cv2.merge((hue, saturation, value)), cv2.COLOR_HSV2BGR)
    if rng.random() < settings["fliplr"]:
        image = np.ascontiguousarray(image[:, ::-1])
        boxes[:, [0, 2]] = size - boxes[:, [2, 0]]
    return image, boxes


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

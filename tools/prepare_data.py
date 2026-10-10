#!/usr/bin/env python3
"""Validate YOLO detection labels and write COCO data for the PFD runtime.

This converter preserves existing train/val/test assignments. It does not make
an acquisition-disjoint split, add attribute labels, or reproduce a paper split.
Invalid annotations fail validation before any output dataset is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Sequence

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_boxes(path: Path, num_classes: int, tolerance: float = 1e-8) -> list[tuple[int, float, float, float, float]]:
                                                                               
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(x) for x in line.split()]
            if len(values) != 5 or not all(math.isfinite(x) for x in values):
                raise ValueError("expected five finite numeric values")
            category, x, y, width, height = values
            if category != int(category) or not 0 <= category < num_classes:
                raise ValueError("class index is outside the configured class list")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("invalid normalized center/width/height")
            if min(x - width / 2, y - height / 2) < -tolerance or max(x + width / 2, y + height / 2) > 1 + tolerance:
                raise ValueError("box crosses an image boundary")
            box = (int(category), x, y, width, height)
            if box in result:
                raise ValueError("duplicate annotation row")
            result.append(box)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_number}: {exc}") from exc
    return result


def collect_split(source: Path, split: str, names: Sequence[str]) -> list[dict]:
    image_dir, label_dir = source / "images" / split, source / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError(f"Missing images/{split} or labels/{split}")
    images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images in images/{split}")
    labels = {p.relative_to(label_dir).with_suffix("").as_posix(): p for p in label_dir.rglob("*.txt")}
    seen, records = set(), []
    for image_path in images:
        relative = image_path.relative_to(image_dir)
        key = relative.with_suffix("").as_posix()
        if key in seen:
            raise ValueError(f"Duplicate image stem in {split}: {key}")
        seen.add(key)
        if key not in labels:
            raise ValueError(f"Missing label in {split}: {key}.txt")
        with Image.open(image_path) as image:
            image.load()
            if image.getexif().get(274, 1) != 1:
                raise ValueError(f"Nontrivial EXIF orientation: {relative}; normalize image and labels together first")
            width, height = image.size
        records.append({"image_path": image_path, "label_path": labels[key], "file_name": relative.as_posix(),
                        "width": width, "height": height, "boxes": read_boxes(labels[key], len(names))})
    orphan_labels = set(labels) - seen
    if orphan_labels:
        raise ValueError(f"Orphan labels in {split}: {sorted(orphan_labels)[:5]}")
    return records


def make_coco(records: list[dict], names: Sequence[str], split: str) -> dict:
    images, annotations = [], []
    for image_id, row in enumerate(records, 1):
        images.append({"id": image_id, "file_name": row["file_name"], "width": row["width"], "height": row["height"]})
        for category, x, y, width, height in row["boxes"]:
                                                                                    
            x0 = max(0.0, (x - width / 2) * row["width"])
            y0 = max(0.0, (y - height / 2) * row["height"])
            x1 = min(float(row["width"]), (x + width / 2) * row["width"])
            y1 = min(float(row["height"]), (y + height / 2) * row["height"])
            bw, bh = x1 - x0, y1 - y0
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": category + 1,
                                "bbox": [x0, y0, bw, bh], "area": bw * bh, "iscrowd": 0})
    return {"info": {"description": f"YOLO-to-COCO conversion; source {split} assignment preserved"},
            "images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": name} for i, name in enumerate(names)]}


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def convert(source: Path, output: Path, names: Sequence[str], splits: Sequence[str], dataset_path: str) -> dict:
    source, output = source.resolve(), output.resolve()
    if not names or len(names) != len(set(names)):
        raise ValueError("Provide one distinct class name per YOLO class index")
    if len(splits) != len(set(splits)) or any(s not in {"train", "val", "test"} for s in splits):
        raise ValueError("Splits must be distinct names from train, val, test")
    rows = {split: collect_split(source, split, names) for split in splits}
                                                                                
    hashes = {}
    for split, records in rows.items():
        for row in records:
            value = sha256(row["image_path"])
            if value in hashes and hashes[value] != split:
                raise ValueError(f"Exact duplicate image crosses splits: {row['file_name']}")
            hashes[value] = split
    cocos = {split: make_coco(records, names, split) for split, records in rows.items()}
                                                                               
    for split, records in rows.items():
        for row in records:
            for kind, original, relative in [("images", row["image_path"], Path(row["file_name"])),
                                              ("labels", row["label_path"], Path(row["file_name"]).with_suffix(".txt"))]:
                target = output / kind / split / relative
                if target.exists() and target.resolve() != original.resolve() and sha256(target) != sha256(original):
                    raise ValueError(f"Output file differs from source: {target}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotations").mkdir(exist_ok=True)
    for split, records in rows.items():
        for row in records:
            for kind, original, relative in [("images", row["image_path"], Path(row["file_name"])),
                                              ("labels", row["label_path"], Path(row["file_name"]).with_suffix(".txt"))]:
                target = output / kind / split / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.resolve() != original.resolve() and not target.exists():
                    shutil.copy2(original, target)
        write_json(output / "annotations" / f"{split}.json", cocos[split])
                                                                               
    runtime_config = {"path": dataset_path, "names": list(names)}
    for split in splits:
        runtime_config[split] = {"images": f"images/{split}", "annotations": f"annotations/{split}.json"}
    write_json(output / "dataset.yaml", runtime_config)
    yolo_config = {"nc": len(names), "names": list(names)}
    yolo_config.update({split: f"images/{split}" for split in splits})
    write_json(output / "data.yaml", yolo_config)
    report = {"splits": {s: {"images": len(x["images"]), "annotations": len(x["annotations"])} for s, x in cocos.items()},
              "class_mapping": {str(i): i + 1 for i in range(len(names))},
              "validation": "All source image/label pairs validated; no source labels edited; exact cross-split image duplicates rejected.",
              "group_scope": "Original split labels preserved. No acquisition/tree/group metadata is inferred by this converter.",
              "file_name_scope": "COCO file_name is relative to images/<split>.",
              "path_scope": "dataset.yaml path is resolved by the consuming runtime; use --dataset-path for the deployment location."}
    write_json(output / "conversion_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing images/<split> and labels/<split>")
    parser.add_argument("--output", type=Path, required=True, help="Destination dataset directory; files are copied, never moved")
    parser.add_argument("--class-names", nargs="+", default=["Pear fruitlet"], help="Names in zero-based YOLO class order")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--dataset-path", default="data/sample", help="Portable dataset.yaml root path for the consuming project")
    args = parser.parse_args()
    try:
        result = convert(args.source, args.output, args.class_names, args.splits, args.dataset_path)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Data conversion failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

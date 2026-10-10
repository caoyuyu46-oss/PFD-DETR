from __future__ import annotations

import contextlib
import copy
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.stats import t as student_t


def coco_metrics(annotation, predictions, canvas=False, size=640):
    if not annotation["images"]:
        raise ValueError("Evaluation requires at least one image")
    annotation, predictions = copy.deepcopy(annotation), copy.deepcopy(predictions)
    if canvas:
        scale = {image["id"]: (size / image["width"], size / image["height"]) for image in annotation["images"]}
        for record in annotation["annotations"] + predictions:
            sx, sy = scale[record["image_id"]]
            x, y, w, h = record["bbox"]
            record["bbox"] = [x * sx, y * sy, w * sx, h * sy]
            if "area" in record:
                record["area"] = w * h * sx * sy
        for image in annotation["images"]:
            image["width"], image["height"] = size, size
    annotation.setdefault("info", {})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = annotation
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {**annotation, "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = [1, 10, 100]
        if canvas:
            evaluator.params.areaRng[1][1] = float(np.nextafter(32. ** 2, 0.))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    values = evaluator.stats
    names = ["AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "ARS", "ARM", "ARL"]
    return {name: None if value < 0 else float(value * 100) for name, value in zip(names, values)}


def iou_matrix(boxes, targets):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    targets = np.asarray(targets, dtype=float).reshape(-1, 4)
    a, b = boxes.copy(), targets.copy()
    a[:, 2:] += a[:, :2]
    b[:, 2:] += b[:, :2]
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.clip(high - low, 0, None).prod(-1)
    union = boxes[:, 2:].prod(1)[:, None] + targets[:, 2:].prod(1)[None, :] - intersection
    return intersection / np.maximum(union, np.finfo(float).eps)


def operating_metrics(annotation, predictions, threshold, iou=0.5):
    if not annotation["images"]:
        raise ValueError("Operating metrics require at least one image")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1 or not 0 < iou <= 1:
        raise ValueError("Invalid confidence threshold or IoU threshold")
    categories = {category["id"] for category in annotation["categories"]}
    if len(categories) != 1:
        raise ValueError("Operating metrics require one foreground category")
    gt = {image["id"]: [] for image in annotation["images"]}
    pred = {key: [] for key in gt}
    for record in annotation["annotations"]:
        if record["category_id"] not in categories or record.get("iscrowd", 0):
            raise ValueError("Operating metrics require valid non-crowd foreground annotations")
        gt[record["image_id"]].append(record)
    identifiers = set()
    for index, record in enumerate(predictions):
        identifier = str(record.get("prediction_id", index))
        if identifier in identifiers:
            raise ValueError("Prediction identifiers must be unique")
        identifiers.add(identifier)
        if record["category_id"] not in categories or record["image_id"] not in pred:
            raise ValueError("A prediction references an unknown category or image")
        if record["score"] >= threshold:
            pred[record["image_id"]].append({**record, "_prediction_id": identifier})
    rows = []
    for info in annotation["images"]:
        image_id = info["id"]
        detections = sorted(pred[image_id], key=lambda item: -item["score"])[:100]
        targets = gt[image_id]
        overlaps = iou_matrix([item["bbox"] for item in detections], [item["bbox"] for item in targets])
        matched = np.zeros(len(targets), dtype=bool)
        matched_predictions = np.zeros(len(detections), dtype=bool)
        for detection_index, overlap in enumerate(overlaps):
            remaining = np.flatnonzero(~matched)
            if remaining.size:
                best = remaining[np.argmax(overlap[remaining])]
                if overlap[best] >= iou:
                    matched[best] = True
                    matched_predictions[detection_index] = True
        tp = int(matched.sum())
        small = np.array([item["bbox"][2] * item["bbox"][3] * 640 ** 2 / (info["width"] * info["height"]) < 32 ** 2 for item in targets], dtype=bool)
        row = {"image_id": image_id, "tp": tp, "fp": len(detections) - tp, "fn": len(targets) - tp,
               "reference_count": len(targets), "predicted_count": len(detections),
               "small_tp": int(matched[small].sum()), "small_n": int(small.sum()),
               "unmatched_prediction_ids": [item["_prediction_id"] for item, used in zip(detections, matched_predictions) if not used]}
        if targets and all("occlusion" in item for item in targets):
            severe = np.array([item["occlusion"] == "O2" for item in targets], dtype=bool)
            row.update(severe_tp=int(matched[severe].sum()), severe_n=int(severe.sum()))
        rows.append(row)
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    errors = np.array([row["predicted_count"] - row["reference_count"] for row in rows], dtype=float)
    n = len(rows)
    small_tp, small_n = (sum(row[key] for row in rows) for key in ("small_tp", "small_n"))
    result = {"threshold": float(threshold), "TP": tp, "FP": fp, "FN": fn,
              "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
              "FP_per_image": fp / n, "FN_per_image": fn / n,
              "MAE": float(np.abs(errors).mean()), "RMSE": float(np.sqrt(np.mean(errors ** 2))), "Bias": float(errors.mean()),
              "small_recall": small_tp / small_n if small_n else None, "images": rows}
    if all("occlusion" in item for item in annotation["annotations"]):
        severe_tp = sum(row.get("severe_tp", 0) for row in rows)
        severe_n = sum(row.get("severe_n", 0) for row in rows)
        result.update(severe_recall=severe_tp / severe_n if severe_n else None,
                      severe_TP=severe_tp, severe_n=severe_n)
    return result


def select_thresholds(annotation, predictions):
    candidates = [operating_metrics(annotation, predictions, k / 20) for k in range(1, 20)]
    valid = [item for item in candidates if item["precision"] is not None and item["precision"] >= 0.9]
    detection = max(valid, key=lambda item: (item["recall"], item["threshold"])) if valid else None
    counting = min(candidates, key=lambda item: (item["MAE"], -item["threshold"]))
    return {"detection": detection["threshold"] if detection else None, "counting": counting["threshold"]}


def summary(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite seed values are required")
    mean, sd = values.mean(), values.std(ddof=1)
    half = student_t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))
    return {"n": len(values), "mean": float(mean), "sample_sd": float(sd), "ci95": [float(mean - half), float(mean + half)]}


def validate_evaluation_alignment(records):
    hashes = [record.get("evaluation_dataset_sha256") for record in records]
    if not hashes or any(not isinstance(value, str) or len(value) != 64 for value in hashes):
        raise ValueError("Evaluation records require evaluation_dataset_sha256 from the common evaluator")
    if len(set(hashes)) != 1:
        raise ValueError("Paired evaluation records refer to different annotation datasets")
    return {"evaluation_dataset_sha256": hashes[0], "dataset_alignment_verified": True}


def reviewed_leaf_metrics(operating_result, reviews):
    decisions = {}
    for row in reviews:
        identifier = str(row["prediction_id"])
        if identifier in decisions or not isinstance(row["is_leaf"], bool):
            raise ValueError("Leaf reviews require unique identifiers and boolean decisions")
        decisions[identifier] = row["is_leaf"]
    unmatched = [identifier for image in operating_result["images"] for identifier in image["unmatched_prediction_ids"]]
    missing = set(unmatched) - set(decisions)
    if missing:
        raise ValueError(f"Leaf review is incomplete for {len(missing)} retained false positives")
    count = sum(decisions[identifier] for identifier in unmatched)
    return {"Leaf_FP": count, "Leaf_FP_per_image": count / len(operating_result["images"]),
            "reviewed_retained_false_positives": len(unmatched)}

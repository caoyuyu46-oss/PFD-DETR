import argparse
import hashlib
import json
import math

from _bootstrap import ROOT
from pfd_runtime.config import digest, resolve, save_json
from pfd_runtime.metrics import coco_metrics, operating_metrics, reviewed_leaf_metrics, select_thresholds


def read(path):
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def validate_predictions(annotations, predictions):
    images = {row['id'] for row in annotations['images']}
    categories = {row['id'] for row in annotations['categories']}
    identifiers, retained = set(), []
    for index, row in enumerate(predictions):
        identifier = str(row.get('prediction_id', index))
        if identifier in identifiers:
            raise ValueError('Prediction identifiers must be unique')
        identifiers.add(identifier)
        box = row['bbox']
        if row['image_id'] not in images or row['category_id'] not in categories:
            raise ValueError('Unknown prediction image or category')
        if len(box) != 4 or not all(math.isfinite(x) for x in box) or min(box[2:]) <= 0:
            raise ValueError('COCO predictions require finite xywh boxes with positive width and height')
        if not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            raise ValueError('Prediction scores must be finite probabilities')
        if row['score'] >= .001:
            retained.append({**row, 'prediction_id': identifier})
    return retained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--validation-annotations", required=True)
    parser.add_argument("--validation-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--leaf-reviews")
    parser.add_argument("--evaluation-split", choices=("test", "development"), default="test")
    args = parser.parse_args()
    annotations = read(args.annotations)
    predictions = validate_predictions(annotations, read(args.predictions))
    validation = read(args.validation_annotations)
    thresholds = select_thresholds(validation, validate_predictions(validation, read(args.validation_predictions)))
    result = coco_metrics(annotations, predictions)
    result["APS640"] = coco_metrics(annotations, predictions, canvas=True)["APS"]
    result.update(seed=args.seed, configuration=args.configuration, thresholds=thresholds,
                  threshold_selection_split="validation", evaluation_split=args.evaluation_split,
                  predictions_sha256=hashlib.sha256(resolve(args.predictions).read_bytes()).hexdigest(),
                  annotations_sha256=hashlib.sha256(resolve(args.annotations).read_bytes()).hexdigest(),
                  evaluation_dataset_sha256=digest(annotations),
                  validation_dataset_sha256=digest(validation),
                  validation_predictions_sha256=hashlib.sha256(resolve(args.validation_predictions).read_bytes()).hexdigest(),
                  score_floor=.001, max_detections=100)
    for purpose, threshold in thresholds.items():
        result[purpose] = operating_metrics(annotations, predictions, threshold) if threshold is not None else None
    if args.leaf_reviews:
        reviews = read(args.leaf_reviews)
        if reviews["predictions_sha256"] != result["predictions_sha256"]:
            raise ValueError("Leaf reviews refer to a different prediction file")
        if result["detection"] is None:
            raise ValueError("The validation set has no feasible precision-constrained operating point")
        result["detection"].update(reviewed_leaf_metrics(result["detection"], reviews["reviews"]))
    save_json(resolve(args.output), result)


if __name__ == "__main__":
    main()

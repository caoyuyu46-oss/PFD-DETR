import argparse

from _bootstrap import ROOT
from pfd_runtime.config import digest, resolve, save_json
from pfd_runtime.data import FruitDataset
from pfd_runtime.engine import evaluate_predictions, predict
from pfd_runtime.metrics import select_thresholds
from pfd_runtime.modeling import load_trained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    model, config, checkpoint = load_trained(resolve(args.checkpoint), args.device)
    data = args.data or config["data"]
    size = config["model"].get("image_size", 640)
    validation = FruitDataset(data, "val", size)
    val_predictions = predict(model, validation, args.device, workers=args.workers)
    thresholds = select_thresholds(validation.coco, val_predictions)
    evaluation = validation if args.split == "val" else FruitDataset(data, "test", size)
    predictions = val_predictions if args.split == "val" else predict(model, evaluation, args.device, workers=args.workers)
    result = evaluate_predictions(evaluation, predictions, thresholds)
    result.update(seed=checkpoint["seed"], epoch=checkpoint["epoch"], configuration=config["experiment"],
                  split=args.split, evaluation_partition=evaluation.config.get("evaluation_split", args.split),
                  threshold_selection_split="val", smoke_steps=checkpoint.get("smoke_steps"))
    result["validation_dataset_sha256"] = digest(validation.coco)
    output = resolve(args.output)
    save_json(output / "thresholds.json", thresholds)
    save_json(output / "predictions.json", predictions)
    save_json(output / "metrics.json", result)
    print({key: result[key] for key in ("AP", "AP75", "APS640", "seed", "split")})


if __name__ == "__main__":
    main()

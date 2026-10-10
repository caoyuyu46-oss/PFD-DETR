import argparse
import numpy as np

from _bootstrap import ROOT
from pfd_runtime.config import digest, resolve, save_json
from pfd_runtime.data import FruitDataset
from pfd_runtime.engine import evaluate_predictions, predict
from pfd_runtime.metrics import summary
from pfd_runtime.modeling import load_trained

INTERVENTIONS = {
    "sswr": ["structure_zero", "scale_uniform", "scale_shuffle", "scale_shift", "bypass"],
    "rdpf": ["ambiguity_only", "shallow_support_shuffle", "deep_support_shuffle", "independent_both_shuffle", "gate_mean", "gate_one", "bypass"],
}


def mean_metric(values):
    if any(value is None for value in values):
        return None
    if not np.isfinite(values).all():
        raise ValueError("Intervention metrics must be finite or undefined")
    return float(np.mean(values))


def summarize_metric(values):
    return None if any(value is None for value in values) else summary(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--module", choices=list(INTERVENTIONS), required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args()
    if args.intervention not in INTERVENTIONS[args.module] or args.permutations < 1:
        raise ValueError("Unsupported intervention or permutation count")
    rows = []
    configuration = None
    for path in args.checkpoints:
        model, config, checkpoint = load_trained(resolve(path), args.device)
        if checkpoint.get("smoke_steps") is not None:
            raise ValueError("Frozen experiment summaries require completed training runs")
        signature = digest(config)
        if configuration is not None and signature != configuration:
            raise ValueError("Frozen summaries require the same resolved configuration for all seeds")
        configuration = signature
        model_config = config["model"]
        if (model_config.get("variant", "full") != "full"
                or model_config.get("sswr", {}).get("mode", "full") != "full"
                or model_config.get("rdpf", {}).get("mode", "full") != "full"):
            raise ValueError("Frozen interventions require a full PFD-DETR configuration")
        dataset = FruitDataset(config["data"], "test", config["model"].get("image_size", 640))
        module = getattr(model.backbone, "sswr", None) if args.module == "sswr" else getattr(model.encoder, "rdpf", None)
        if module is None or not hasattr(module, "intervention"):
            raise ValueError("The checkpoint does not contain the selected module")
        module.intervention = "none"
        intact = evaluate_predictions(dataset, predict(model, dataset, args.device, workers=args.workers))
        module.intervention = args.intervention
        repeats = args.permutations if "shuffle" in args.intervention else 1
        values = []
        for repetition in range(repeats):
            module.intervention_seed = checkpoint["seed"] * 1000 + repetition
            value = evaluate_predictions(dataset, predict(model, dataset, args.device, workers=args.workers))
            values.append({key: value[key] for key in ("AP", "APS640")})
        average = {key: mean_metric([value[key] for value in values]) for key in ("AP", "APS640")}
        rows.append({"seed": checkpoint["seed"], "intact": {key: intact[key] for key in average}, "intervention": average,
                     "delta": {key: average[key] - intact[key] if average[key] is not None and intact[key] is not None else None
                               for key in average}, "repetitions": values})
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("Checkpoint seeds must be distinct")
    result = {"module": args.module, "intervention": args.intervention, "configuration_sha256": configuration, "runs": rows}
    if len(rows) > 1:
        result["summary"] = {key: summarize_metric([row["delta"][key] for row in rows]) for key in ("AP", "APS640")}
        result["intact_summary"] = {key: summarize_metric([row["intact"][key] for row in rows]) for key in ("AP", "APS640")}
        result["intervention_summary"] = {key: summarize_metric([row["intervention"][key] for row in rows]) for key in ("AP", "APS640")}
    save_json(resolve(args.output), result)


if __name__ == "__main__":
    main()

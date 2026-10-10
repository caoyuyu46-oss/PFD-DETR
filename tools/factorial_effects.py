import argparse
import json

from _bootstrap import ROOT
from pfd_runtime.config import resolve, save_json
from pfd_runtime.metrics import summary, validate_evaluation_alignment


def load_runs(paths):
    records = [json.loads(resolve(path).read_text(encoding="utf-8")) for path in paths]
    if len(records) != 5 or {row["seed"] for row in records} != set(range(5)):
        raise ValueError("Each arm requires exactly the five distinct seeds 0 through 4")
    if any(row.get("smoke_steps") is not None for row in records):
        raise ValueError("Smoke results cannot be used as experiment evidence")
    return {row["seed"]: row for row in records}


def main():
    parser = argparse.ArgumentParser()
    for arm in ("baseline", "sswr", "support", "sswr-support", "rdpf", "full"):
        parser.add_argument("--" + arm, required=True, nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    arms = {name: load_runs(getattr(args, name)) for name in ("baseline", "sswr", "support", "sswr_support", "rdpf", "full")}
    contrasts = {
        "SSWR_given_baseline": {"sswr": 1, "baseline": -1},
        "SSWR_given_support": {"sswr_support": 1, "support": -1},
        "SSWR_given_RDPF": {"full": 1, "rdpf": -1},
        "RDPF_residual_given_SSWR_and_support": {"full": 1, "sswr_support": -1},
        "support_given_SSWR": {"sswr_support": 1, "sswr": -1},
        "support_given_baseline": {"support": 1, "baseline": -1},
        "RDPF_residual_given_support": {"rdpf": 1, "support": -1},
        "SSWR_RDPF_interaction": {"full": 1, "sswr": -1, "rdpf": -1, "baseline": 1},
        "SSWR_support_interaction": {"sswr_support": 1, "sswr": -1, "support": -1, "baseline": 1},
        "SSWR_RDPF_residual_interaction_given_support": {"full": 1, "sswr_support": -1, "rdpf": -1, "support": 1},
        "full_minus_baseline": {"full": 1, "baseline": -1},
    }
    alignment = validate_evaluation_alignment([record for arm in arms.values() for record in arm.values()])
    output = {"seeds": list(range(5)), "contrasts": {}, **alignment}
    for name, coefficients in contrasts.items():
        output["contrasts"][name] = {"coefficients": coefficients}
        for metric in ("AP", "AP75", "APS640"):
            values = [sum(weight * arms[arm][seed][metric] for arm, weight in coefficients.items()) for seed in range(5)]
            output["contrasts"][name][metric] = {"seed_values": values, **summary(values)}
    save_json(resolve(args.output), output)


if __name__ == "__main__":
    main()

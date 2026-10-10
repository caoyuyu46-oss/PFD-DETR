import argparse
import json

from _bootstrap import ROOT
from pfd_runtime.config import resolve, save_json
from pfd_runtime.metrics import summary, validate_evaluation_alignment


def read(paths):
    records = [json.loads(resolve(path).read_text(encoding="utf-8")) for path in paths]
    if any(record.get("smoke_steps") is not None for record in records):
        raise ValueError("Smoke-test records cannot be aggregated as experiment results")
    if len({record["seed"] for record in records}) != len(records):
        raise ValueError("Each input must contain a different seed")
    return {record["seed"]: record for record in records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--reference", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = read(args.metrics)
    result = {key: summary([record[key] for record in records.values()]) for key in ("AP", "AP75", "APS640")}
    result.update(validate_evaluation_alignment(list(records.values())))
    if args.reference:
        reference = read(args.reference)
        if set(reference) != set(records):
            raise ValueError("Paired summaries require the same seed identifiers")
        validate_evaluation_alignment([*records.values(), *reference.values()])
        result["paired_effect"] = {key: summary([records[seed][key] - reference[seed][key] for seed in sorted(records)]) for key in ("AP", "AP75", "APS640")}
    save_json(resolve(args.output), result)


if __name__ == "__main__":
    main()

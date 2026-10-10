from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Circular configuration inheritance: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    base = data.pop("_base_", None)
    return merge(load_config(path.parent / base, (*stack, path)), data) if base else data


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")

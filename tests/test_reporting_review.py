import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pfd_runtime.config import digest
from pfd_runtime.metrics import coco_metrics, operating_metrics, validate_evaluation_alignment


def test_canvas_small_area_boundary_matches_strict_paper_definition():
    annotations = {"images": [{"id": 1, "width": 640, "height": 640}],
                   "categories": [{"id": 1, "name": "fruit"}],
                   "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                                    "bbox": [0, 0, 16, 16], "area": 256},
                                   {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 0,
                                    "bbox": [100, 100, 32, 32], "area": 1024}]}
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [100, 100, 32, 32], "score": 0.99}]
    native = coco_metrics(annotations, predictions)
    canvas = coco_metrics(annotations, predictions, canvas=True)
    assert native["APS"] > 49.
    assert canvas["APS"] == 0.
    assert canvas["AP"] == native["AP"]
    assert operating_metrics(annotations, predictions, 0.5)["small_recall"] == 0.


def test_paired_statistics_reject_different_or_missing_datasets():
    first = digest({"images": [{"id": 1}], "annotations": [], "categories": []})
    second = digest({"images": [{"id": 2}], "annotations": [], "categories": []})
    records = [{"seed": 0, "evaluation_dataset_sha256": first},
               {"seed": 1, "evaluation_dataset_sha256": first}]
    assert validate_evaluation_alignment(records)["dataset_alignment_verified"] is True
    records[1]["evaluation_dataset_sha256"] = second
    with pytest.raises(ValueError, match="different annotation datasets"):
        validate_evaluation_alignment(records)
    del records[1]["evaluation_dataset_sha256"]
    with pytest.raises(ValueError, match="require evaluation_dataset_sha256"):
        validate_evaluation_alignment(records)

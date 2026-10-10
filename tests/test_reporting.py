import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from pfd_runtime.metrics import operating_metrics, reviewed_leaf_metrics
from pfd_runtime.config import digest


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_leaf_reviews_apply_only_to_retained_unmatched_predictions():
    annotations = {'images': [{'id': 1, 'width': 640, 'height': 640}],
                   'categories': [{'id': 1, 'name': 'fruit'}],
                   'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10]}]}
    predictions = [{'image_id': 1, 'category_id': 1, 'bbox': box, 'score': score, 'prediction_id': name}
                   for box, score, name in [([0, 0, 10, 10], .95, 'tp'), ([0, 0, 10, 10], .9, 'duplicate'),
                                            ([100, 100, 10, 10], .8, 'leaf'), ([200, 200, 10, 10], .1, 'discarded')]]
    result = operating_metrics(annotations, predictions, .5)
    assert result['images'][0]['unmatched_prediction_ids'] == ['duplicate', 'leaf']
    reviews = [{'prediction_id': 'duplicate', 'is_leaf': False}, {'prediction_id': 'leaf', 'is_leaf': True}]
    assert reviewed_leaf_metrics(result, reviews)['Leaf_FP_per_image'] == 1.
    with pytest.raises(ValueError, match='incomplete'):
        reviewed_leaf_metrics(result, reviews[:1])
    with pytest.raises(ValueError, match='unique'):
        reviewed_leaf_metrics(result, reviews + reviews)


def test_factorial_effects_use_paired_seeds_and_reject_smoke_records():
    module = tool('factorial_effects')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        offsets = {'baseline': 0., 'sswr': 2., 'support': 1., 'sswr_support': 3., 'rdpf': 4., 'full': 6.}
        argv = ['factorial_effects']
        all_paths = {}
        dataset_hash = digest({'images': [{'id': 1, 'width': 640, 'height': 640}],
                               'categories': [{'id': 1, 'name': 'fruit'}], 'annotations': []})
        for arm, offset in offsets.items():
            paths = []
            for seed in [4, 1, 3, 0, 2]:
                path = root / f'{arm}-{seed}.json'
                path.write_text(json.dumps({'seed': seed, 'AP': 50. + seed + offset,
                                            'AP75': 40. + seed + offset, 'APS640': 30. + seed + offset,
                                            'evaluation_dataset_sha256': dataset_hash}))
                paths.append(str(path))
            all_paths[arm] = paths
            argv += ['--' + arm.replace('_', '-'), *paths]
        output = root / 'effects.json'
        argv += ['--output', str(output)]
        with mock.patch.object(sys, 'argv', argv):
            module.main()
        result = json.loads(output.read_text())['contrasts']
        assert len(result) == 11
        assert result['SSWR_RDPF_interaction']['AP']['seed_values'] == [0.] * 5
        assert result['SSWR_RDPF_residual_interaction_given_support']['AP']['mean'] == 0.
        assert result['full_minus_baseline']['AP']['mean'] == 6.
        assert result['full_minus_baseline']['AP']['sample_sd'] == 0.
        path = Path(all_paths['full'][0])
        row = json.loads(path.read_text())
        row['smoke_steps'] = 1
        path.write_text(json.dumps(row))
        with pytest.raises(ValueError, match='Smoke'):
            module.load_runs(all_paths['full'])


def test_external_predictions_validate_geometry_scores_and_keep_review_ids():
    module = tool('evaluate_predictions')
    annotations = {'images': [{'id': 1}], 'categories': [{'id': 7}]}
    rows = [{'image_id': 1, 'category_id': 7, 'bbox': [0, 0, 10, 10], 'score': score}
            for score in [.0005, .8]]
    assert module.validate_predictions(annotations, rows)[0]['prediction_id'] == '1'
    rows[0]['score'] = float('nan')
    with pytest.raises(ValueError, match='probabilities'):
        module.validate_predictions(annotations, rows)
    rows[0]['score'] = .9
    rows[0]['bbox'][2] = -1
    with pytest.raises(ValueError, match='positive'):
        module.validate_predictions(annotations, rows)


def test_image_to_boxes_benchmark_preprocesses_every_timed_image():
    module = tool('benchmark')

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            assert tuple(images.shape) == (1, 3, 32, 32)
            assert torch.equal(images[:, 0], torch.ones_like(images[:, 0]))
            return {'pred_logits': torch.tensor([[[2.]]]), 'pred_boxes': torch.tensor([[[.5, .5, .2, .2]]])}

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'input.png'
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        cv2.imencode('.png', image)[1].tofile(path)
        model = Detector()
        argv = ['benchmark', '--checkpoint', 'unused', '--output', 'unused.json', '--device', 'cpu',
                '--scope', 'image-to-boxes', '--images', str(path), '--warmup', '1', '--iterations', '2', '--rounds', '2']
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(module, 'load_trained', return_value=(model, {'model': {'image_size': 32}}, {})), \
                mock.patch.object(module, 'save_json') as save:
            module.main()
        assert model.calls == 6
        assert save.call_args.args[1]['disk_decode_included'] is False
        assert save.call_args.args[1]['scope'] == 'image-to-boxes'
        assert save.call_args.args[1]['peak_allocated_bytes'] is None
        assert save.call_args.args[1]['peak_allocated_MiB'] is None

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pfd_runtime.data import FruitDataset
from pfd_runtime.engine import predict, train
from pfd_runtime.metrics import coco_metrics, operating_metrics, select_thresholds, summary
from pfd_runtime.modeling import initialize


torch.set_num_threads(2)


def annotation_fixture():
    return {"images": [{"id": 1, "width": 640, "height": 640},
                       {"id": 2, "width": 640, "height": 640}],
            "categories": [{"id": 7, "name": "fruit"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 7,
                             "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0, "occlusion": "O2"},
                            {"id": 2, "image_id": 1, "category_id": 7,
                             "bbox": [50, 50, 40, 40], "area": 1600, "iscrowd": 0, "occlusion": "O0"}]}


def detection(image_id, bbox, score):
    return {"image_id": image_id, "category_id": 7, "bbox": bbox, "score": score}


class MetricTests(unittest.TestCase):
    def test_one_to_one_duplicates_empty_images_and_bias_identity(self):
        annotations = annotation_fixture()
        predictions = [detection(1, [10, 10, 10, 10], 0.95),
                       detection(1, [10, 10, 10, 10], 0.90),
                       detection(1, [50, 50, 40, 40], 0.8),
                       detection(2, [100, 100, 20, 20], 0.85)]
        result = operating_metrics(annotations, predictions, 0.85)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 2, 1))
        self.assertEqual(result["precision"], 1 / 3)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["small_recall"], 1.)
        self.assertEqual(result["severe_recall"], 1.)
        self.assertEqual(result["MAE"], 0.5)
        self.assertEqual(result["Bias"], result["FP_per_image"] - result["FN_per_image"])

    def test_separate_operating_and_counting_thresholds(self):
        annotations = annotation_fixture()
        predictions = [detection(1, [10, 10, 10, 10], 0.91),
                       detection(1, [200, 200, 30, 30], 0.8)]
        selected = select_thresholds(annotations, predictions)
        self.assertEqual(selected, {"detection": 0.9, "counting": 0.8})

    def test_empty_predictions_and_infeasible_precision(self):
        annotations = annotation_fixture()
        empty = operating_metrics(annotations, [], 0.5)
        self.assertEqual((empty["TP"], empty["FP"], empty["FN"]), (0, 0, 2))
        self.assertIsNone(empty["precision"])
        self.assertEqual(select_thresholds(annotations, []), {"detection": None, "counting": 0.95})
        wrong = [detection(1, [200, 200, 30, 30], 0.99)]
        self.assertIsNone(select_thresholds(annotations, wrong)["detection"])
        metrics = coco_metrics(annotations, [])
        self.assertEqual(metrics["AP"], 0.)

    def test_max_detections_and_original_area_coordinates(self):
        annotations = annotation_fixture()
        predictions = [detection(2, [100, 100, 20, 20], 0.99) for _ in range(120)]
        result = operating_metrics(annotations, predictions, 0.5)
        self.assertEqual(result["FP"], 100)
        self.assertEqual(result["images"][1]["predicted_count"], 100)
        annotations["images"][0]["width"] = 1280
        annotations["images"][0]["height"] = 1280
        predictions = [detection(1, [10, 10, 10, 10], 0.99), detection(1, [50, 50, 40, 40], 0.98)]
        original = copy.deepcopy(annotations)
        metrics = coco_metrics(annotations, predictions, canvas=True)
        self.assertAlmostEqual(metrics["APS"], 100., places=6)
        self.assertEqual(annotations, original)

    def test_empty_dataset_and_wrong_category_are_rejected(self):
        annotations = annotation_fixture()
        annotations["images"] = []
        with self.assertRaises(ValueError):
            operating_metrics(annotations, [], 0.5)
        with self.assertRaises(ValueError):
            coco_metrics(annotations, [])
        wrong = detection(1, [10, 10, 10, 10], 0.9)
        wrong["category_id"] = 1
        with self.assertRaises(ValueError):
            operating_metrics(annotation_fixture(), [wrong], 0.5)

    def test_summary_uses_sample_sd_and_paired_delta_input(self):
        result = summary([1., 2., 3., 4., 5.])
        self.assertEqual(result["mean"], 3.)
        self.assertAlmostEqual(result["sample_sd"], np.std([1, 2, 3, 4, 5], ddof=1))
        self.assertAlmostEqual(result["ci95"][0], 1.0367568385, places=8)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        cv2.imencode(".png", image)[1].tofile(self.root / "fruit.png")
        self.annotation = {"images": [{"id": 17, "width": 80, "height": 40, "file_name": "fruit.png"}],
                           "categories": [{"id": 7, "name": "fruit"}],
                           "annotations": [{"id": 5, "image_id": 17, "category_id": 7,
                                            "bbox": [20, 10, 40, 20], "area": 800, "iscrowd": 0}]}
        self.data = {"path": str(self.root), "train": {"images": ".", "annotations": "annotations.json"}}
        self.write_annotation()

    def tearDown(self):
        self.temp.cleanup()

    def write_annotation(self):
        (self.root / "annotations.json").write_text(json.dumps(self.annotation), encoding="utf-8")

    def test_rgb_resize_normalized_box_and_category_map(self):
        dataset = FruitDataset(self.data, "train", 64)
        image, target = dataset[0]
        self.assertEqual(image.shape, (3, 64, 64))
        torch.testing.assert_close(image[0], torch.ones(64, 64))
        torch.testing.assert_close(image[2], torch.zeros(64, 64))
        torch.testing.assert_close(target["boxes"], torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
        self.assertEqual(target["labels"].tolist(), [0])
        self.assertEqual(target["orig_size"].tolist(), [80, 40])
        self.assertEqual(dataset.category_id, 7)

    def test_unknown_category_and_invalid_box_are_rejected(self):
        self.annotation["annotations"][0]["category_id"] = 99
        self.write_annotation()
        with self.assertRaises(ValueError):
            FruitDataset(self.data, "train", 64)
        self.annotation["annotations"][0]["category_id"] = 7
        self.annotation["annotations"][0]["bbox"][2] = 0
        self.write_annotation()
        with self.assertRaises(ValueError):
            FruitDataset(self.data, "train", 64)

    def test_partial_boundary_box_is_clipped_and_empty_image_supported(self):
        self.annotation["annotations"][0]["bbox"] = [-10, 10, 20, 20]
        self.write_annotation()
        target = FruitDataset(self.data, "train", 64)[0][1]
        torch.testing.assert_close(target["boxes"], torch.tensor([[0.0625, 0.5, 0.125, 0.5]]))
        self.annotation["annotations"] = []
        self.write_annotation()
        target = FruitDataset(self.data, "train", 64)[0][1]
        self.assertEqual(target["boxes"].shape, (0, 4))

    def test_augmentation_reproducible_by_image_epoch_and_seed(self):
        settings = {"scale": 0.2, "translate": 0.1, "hsv_h": 0.015,
                    "hsv_s": 0.7, "hsv_v": 0.4, "fliplr": 0.5}
        first = FruitDataset(self.data, "train", 64, settings, seed=2)
        second = FruitDataset(self.data, "train", 64, settings, seed=2)
        image_a, target_a = first[0]
        image_b, target_b = second[0]
        torch.testing.assert_close(image_a, image_b, rtol=0., atol=0.)
        torch.testing.assert_close(target_a["boxes"], target_b["boxes"], rtol=0., atol=0.)
        first.set_epoch(1)
        image_c, target_c = first[0]
        self.assertFalse(torch.equal(image_a, image_c) and torch.equal(target_a["boxes"], target_c["boxes"]))

    def test_prediction_restores_original_coordinates_and_category(self):
        dataset = FruitDataset(self.data, "train", 64)

        class Detector(torch.nn.Module):
            def forward(self, images):
                return {"pred_logits": torch.tensor([[[2.], [-20.]]]).expand(len(images), -1, -1),
                        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]]).expand(len(images), -1, -1)}

        model = Detector().train()
        predictions = predict(model, dataset, "cpu")
        self.assertTrue(model.training)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["category_id"], 7)
        self.assertEqual(predictions[0]["image_id"], 17)
        self.assertEqual(predictions[0]["bbox"], [20., 10., 40., 20.])


class InitializationTests(unittest.TestCase):
    def test_common_layers_and_rdpf_initialization_align_across_arms(self):
        config = {"image_size": 128, "num_queries": 20, "num_denoising": 10}
        baseline, baseline_meta = initialize({**config, "variant": "baseline"}, 3)
        full, full_meta = initialize({**config, "variant": "full"}, 3)
        rdpf, rdpf_meta = initialize({**config, "variant": "rdpf"}, 3)
        self.assertEqual(baseline_meta["shared_state_sha256"], full_meta["shared_state_sha256"])
        self.assertEqual(baseline_meta["shared_state_sha256"], rdpf_meta["shared_state_sha256"])
        baseline_state, full_state, rdpf_state = baseline.state_dict(), full.state_dict(), rdpf.state_dict()
        for key in baseline_state:
            self.assertTrue(torch.equal(baseline_state[key], full_state[key]), key)
            self.assertTrue(torch.equal(baseline_state[key], rdpf_state[key]), key)
        for key in (key for key in full_state if ".rdpf." in key):
            self.assertTrue(torch.equal(full_state[key], rdpf_state[key]), key)
        changed, changed_meta = initialize({**config, "variant": "baseline"}, 4)
        self.assertNotEqual(changed_meta["shared_state_sha256"], baseline_meta["shared_state_sha256"])

    @unittest.skipUnless((ROOT / "weights/rtdetr_r18vd_6x_coco.pth").exists(), "Official checkpoint is optional")
    def test_real_official_coco_checkpoint_shared_layers(self):
        path = ROOT / "weights/rtdetr_r18vd_6x_coco.pth"
        config = {"image_size": 128, "num_queries": 20}
        baseline, baseline_meta = initialize({**config, "variant": "baseline"}, 0, path)
        full, full_meta = initialize({**config, "variant": "full"}, 0, path)
        self.assertEqual(baseline_meta["shared_state_sha256"], full_meta["shared_state_sha256"])
        self.assertEqual(baseline_meta["pretrained"]["loaded_tensors"], 455)
        raw = torch.load(path, map_location="cpu")
        weights = raw.get("ema", raw.get("model", raw))
        weights = weights.get("module", weights)
        baseline_state, full_state = baseline.state_dict(), full.state_dict()
        loaded = []
        for key, value in weights.items():
            if key in baseline_state and value.shape == baseline_state[key].shape:
                self.assertTrue(torch.equal(baseline_state[key], value), key)
                self.assertTrue(torch.equal(full_state[key], value), key)
                loaded.append(key)
        self.assertEqual(len(loaded), 455)
        self.assertEqual(full_state["decoder.enc_score_head.weight"].shape[0], 1)


class ResumeTests(unittest.TestCase):
    def test_smoke_checkpoint_cannot_be_reclassified_as_formal_training(self):
        class Dataset:
            images = [1]

            def __init__(self, *args):
                pass

        config = {"model": {"variant": "baseline"}, "data": {}, "augmentation": {},
                  "initialization": {"type": "random"},
                  "training": {"lr": 1e-4, "betas": [0.9, 0.999], "weight_decay": 0.0001}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "smoke.pth"
            torch.save({"config": config, "seed": 0, "smoke_steps": 1}, checkpoint)
            with mock.patch("pfd_runtime.engine.FruitDataset", Dataset), \
                    mock.patch("pfd_runtime.engine.initialize", return_value=(torch.nn.Linear(1, 1), {})), \
                    mock.patch("pfd_runtime.engine.criterion", return_value=torch.nn.Identity()):
                with self.assertRaisesRegex(ValueError, "Smoke-test checkpoints"):
                    train(config, root / "resumed", seed=0, device="cpu", workers=0, resume=checkpoint)

    def test_optimizer_restore_casts_moment_state_to_parameter_device(self):
        source = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([source], lr=1e-4)
        source.square().sum().backward()
        optimizer.step()
        target = torch.nn.Parameter(torch.empty(2, device="meta"))
        restored = torch.optim.AdamW([target], lr=1e-4)
        restored.load_state_dict(optimizer.state_dict())
        self.assertEqual(restored.state[target]["exp_avg"].device.type, "meta")
        self.assertEqual(restored.state[target]["exp_avg_sq"].device.type, "meta")
        self.assertEqual(restored.state[target]["step"].device.type, "cpu")


class ToolTests(unittest.TestCase):
    def test_benchmark_uses_selected_cuda_device_and_warms_each_round(self):
        spec = importlib.util.spec_from_file_location("pfd_benchmark_test", ROOT / "tools/benchmark.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, images):
                self.calls += 1

        model = Detector()
        images = torch.zeros(1, 3, 32, 32)
        argv = ["benchmark", "--checkpoint", "unused.pth", "--output", "unused.json",
                "--device", "cuda:1", "--warmup", "3", "--iterations", "2", "--rounds", "2"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(module, "load_trained", return_value=(model, {"model": {"image_size": 32}}, {})), \
                mock.patch.object(module, "save_json") as save, \
                mock.patch.object(module.torch, "zeros", return_value=images), \
                mock.patch.object(module.torch.cuda, "synchronize") as sync, \
                mock.patch.object(module.torch.cuda, "reset_peak_memory_stats") as reset_peak, \
                mock.patch.object(module.torch.cuda, "max_memory_allocated", side_effect=[2 * 1024 ** 2, 3 * 1024 ** 2]) as peak, \
                mock.patch.object(module.torch.cuda, "get_device_name", return_value="selected device") as name:
            module.main()
        self.assertEqual(model.calls, 2 * (3 + 2))
        self.assertTrue(all(call.args == ("cuda:1",) for call in sync.call_args_list))
        name.assert_called_once_with("cuda:1")
        self.assertEqual(save.call_args.args[1]["hardware"], "selected device")
        self.assertEqual(save.call_args.args[1]["peak_allocated_MiB"], 3.)
        self.assertEqual(reset_peak.call_count, 2)
        self.assertEqual(peak.call_count, 2)
        self.assertTrue(all(call.args == ("cuda:1",) for call in reset_peak.call_args_list + peak.call_args_list))


if __name__ == "__main__":
    unittest.main(verbosity=2)

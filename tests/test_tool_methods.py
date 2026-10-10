import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "rtdetr_pytorch"))

from src.nn.backbone.pfd_modules import SSWR


torch.set_num_threads(2)


def tool(name):
    spec = importlib.util.spec_from_file_location("test_" + name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenInterventionTests(unittest.TestCase):
    def setUp(self):
        self.module = tool("frozen_interventions")
        self.model = SimpleNamespace(backbone=SimpleNamespace(sswr=SSWR().eval()))
        self.config = {"data": "test_data", "model": {"variant": "full", "image_size": 64}}

    def load(self, path, device):
        seed = int(path.stem[-1])
        return self.model, copy.deepcopy(self.config), {"seed": seed, "smoke_steps": None}

    def predict(self, model, dataset, device, workers):
        module = model.backbone.sswr
        with torch.no_grad():
            module(torch.zeros(1, 128, 8, 8))
        value = 100. if module.intervention == "none" else 90. + module.intervention_seed % 1000
        return {"AP": value, "APS640": None}

    def test_intact_mode_permutation_mean_and_seed_summary(self):
        argv = ["frozen", "--checkpoints", "seed0.pth", "seed1.pth", "--module", "sswr",
                "--intervention", "scale_shuffle", "--permutations", "3", "--output", "unused.json"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(self.module, "load_trained", side_effect=self.load), \
                mock.patch.object(self.module, "FruitDataset", return_value=object()), \
                mock.patch.object(self.module, "predict", side_effect=self.predict) as prediction, \
                mock.patch.object(self.module, "evaluate_predictions", side_effect=lambda dataset, predictions: predictions), \
                mock.patch.object(self.module, "save_json") as save:
            self.module.main()
        self.assertEqual(prediction.call_count, 8)
        result = save.call_args.args[1]
        self.assertEqual([row["intervention"]["AP"] for row in result["runs"]], [91., 91.])
        self.assertEqual(result["summary"]["AP"]["mean"], -9.)
        self.assertEqual(result["intervention_summary"]["AP"]["sample_sd"], 0.)
        self.assertIsNone(result["summary"]["APS640"])
        self.assertIsNone(result["runs"][0]["delta"]["APS640"])

    def test_mixed_configurations_and_missing_full_model_are_rejected(self):
        argv = ["frozen", "--checkpoints", "seed0.pth", "seed1.pth", "--module", "sswr",
                "--intervention", "bypass", "--output", "unused.json"]

        def load(path, device):
            model, config, checkpoint = self.load(path, device)
            config["data"] += str(checkpoint["seed"])
            return model, config, checkpoint

        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(self.module, "load_trained", side_effect=load), \
                mock.patch.object(self.module, "FruitDataset", return_value=object()), \
                mock.patch.object(self.module, "predict", side_effect=self.predict), \
                mock.patch.object(self.module, "evaluate_predictions", side_effect=lambda dataset, predictions: predictions):
            with self.assertRaisesRegex(ValueError, "same resolved configuration"):
                self.module.main()
        self.config["model"]["variant"] = "baseline"
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(self.module, "load_trained", side_effect=self.load):
            with self.assertRaisesRegex(ValueError, "full PFD-DETR"):
                self.module.main()


class HeatmapTests(unittest.TestCase):
    def test_default_six_queries_and_historical_box_normalization(self):
        module = tool("heatmap")

        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Conv2d(3, 4, 1)
                self.encoder = torch.nn.Module()
                self.encoder.cam_fusion = torch.nn.Identity()
                self.projection.weight.data.fill_(0.1)
                self.projection.bias.data.zero_()

            def forward(self, images):
                activation = self.encoder.cam_fusion(self.projection(images))
                signal = activation.mean((1, 2, 3)).reshape(-1, 1, 1)
                logits = signal + torch.linspace(2., -2., 300).reshape(1, 300, 1)
                boxes = torch.tensor([0.5, 0.5, 1., 1.]).reshape(1, 1, 4).expand(1, 300, 4)
                return {"pred_logits": logits, "pred_boxes": boxes}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((32, 40, 3), np.uint8)
            image[:, :, 2] = np.arange(40)[None, :] * 6
            image[:, :, 1] = np.arange(32)[:, None] * 7
            cv2.imencode(".png", image)[1].tofile(root / "image.png")
            argv = ["heatmap", "--checkpoint", "unused.pth", "--image", str(root / "image.png"),
                    "--output", str(root / "cam.png"), "--device", "cpu"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(module, "load_trained", return_value=(Detector(), {"model": {"image_size": 32}}, {"seed": 0})):
                module.main()
            metadata = json.loads((root / "cam.json").read_text())
            with np.load(root / "cam.npz") as archive:
                arrays = {key: archive[key] for key in archive.files}
            self.assertEqual(metadata["selected_queries"], 6)
            self.assertEqual(metadata["target"], "sum_of_selected_class_scores")
            self.assertTrue(np.isfinite(arrays["raw_cam"]).all())
            self.assertGreater(arrays["box_normalized_cam"].max(), 0.)
            self.assertTrue((arrays["box_normalized_cam"][-1, :] == 0.).all())
            self.assertTrue((arrays["box_normalized_cam"][:, -1] == 0.).all())
            self.assertEqual(cv2.imread(str(root / "cam.png")).shape, (32, 40, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)

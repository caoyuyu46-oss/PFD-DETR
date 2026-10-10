import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rtdetr_pytorch"))

from pfd_runtime.modeling import initialize
from pfd_runtime.engine import build_optimizer
from src.zoo.rtdetr.pfd_decoder import paper_denoising
from src.zoo.rtdetr.pfd_model import build_model, build_criterion

torch.set_num_threads(2)


class PaperArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.config = {"architecture": "paper_v716", "image_size": 128,
                       "num_queries": 20, "num_denoising": 10}

    def test_exact_parameter_counts_and_legacy_compatibility(self):
        count = lambda model: sum(parameter.numel() for parameter in model.parameters())
        baseline = build_model({**self.config, "variant": "baseline"})
        full = build_model({**self.config, "variant": "full"})
        legacy = build_model({**self.config, "architecture": "legacy_official", "variant": "baseline"})
        self.assertEqual(count(baseline), 20082772)
        self.assertEqual(count(full), 20144979)
        self.assertEqual(count(full) - count(baseline), 62207)
        self.assertEqual(count(legacy) - count(baseline), 256)
        self.assertEqual(baseline.decoder.denoising_class_embed.weight.shape, (1, 256))
        self.assertEqual(legacy.decoder.denoising_class_embed.weight.shape, (2, 256))
        for module in baseline.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertEqual((module.eps, module.momentum), (1e-3, 0.03))
        implicit = build_model({"variant": "baseline", "image_size": 128, "num_queries": 20})
        implicit.load_state_dict(legacy.state_dict(), strict=True)

    def test_paired_initialization_and_zero_residual_equivalence(self):
        baseline, base_meta = initialize({**self.config, "variant": "baseline"}, 19)
        full, full_meta = initialize({**self.config, "variant": "full"}, 19)
        self.assertEqual(base_meta["shared_state_sha256"], full_meta["shared_state_sha256"])
        for key, tensor in baseline.state_dict().items():
            torch.testing.assert_close(tensor, full.state_dict()[key], rtol=0., atol=0.)
        baseline.eval()
        full.eval()
        with torch.no_grad():
            images = torch.randn(2, 3, 128, 128)
            first, second = baseline(images), full(images)
        for key in ("pred_boxes", "pred_logits"):
            torch.testing.assert_close(first[key], second[key], rtol=0., atol=0.)

    def test_six_arms_training_with_unequal_and_empty_targets(self):
        targets = [{"labels": torch.zeros(2, dtype=torch.long), "boxes": torch.tensor([[.3, .4, .1, .2], [.7, .6, .2, .1]])},
                   {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}]
        for variant in ("baseline", "sswr", "support", "sswr_support", "rdpf", "full"):
            with self.subTest(variant=variant):
                config = {**self.config, "variant": variant}
                model = build_model(config).train()
                outputs = model(torch.randn(2, 3, 128, 128), targets)
                loss = sum(build_criterion(config)(outputs, targets).values())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(model.decoder.denoising_class_embed.weight.grad)
                self.assertTrue(torch.isfinite(model.decoder.denoising_class_embed.weight.grad).all())
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_paper_loss_retains_varifocal_modulator_gradient(self):
        target = [{"labels": torch.zeros(1, dtype=torch.long), "boxes": torch.tensor([[.5, .5, .2, .2]])}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        gradients = []
        for architecture in ("legacy_official", "paper_v716"):
            logits = torch.zeros(1, 2, 1, requires_grad=True)
            outputs = {"pred_logits": logits, "pred_boxes": torch.tensor([[[.5, .5, .2, .2], [.1, .1, .1, .1]]])}
            criterion = build_criterion({"variant": "baseline", "architecture": architecture})
            criterion.loss_labels_vfl(outputs, target, indices, 1)["loss_vfl"].backward()
            gradients.append(logits.grad[0, 1, 0].item())
        self.assertGreater(gradients[1], 2 * gradients[0])

    def test_denoising_padding_and_no_target_batch(self):
        embedding = nn.Embedding(1, 256)
        targets = [{"labels": torch.zeros(1, dtype=torch.long), "boxes": torch.tensor([[.5, .5, .2, .2]])},
                   {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}]
        content, references, attention, metadata = paper_denoising(targets, embedding, 20, 10)
        self.assertEqual(content.shape, (2, 20, 256))
        self.assertEqual(content[1].count_nonzero().item(), 0)
        self.assertEqual(references[1].count_nonzero().item(), 0)
        self.assertTrue(attention[20:, :20].all())
        self.assertEqual(len(metadata["dn_positive_idx"][0]), 10)
        self.assertEqual(len(metadata["dn_positive_idx"][1]), 0)
        self.assertEqual(paper_denoising(targets[1:], embedding, 20, 10), (None, None, None, None))
        config = {**self.config, "variant": "full"}
        model = build_model(config).train()
        empty = targets[1:] * 2
        loss = sum(build_criterion(config)(model(torch.randn(2, 3, 128, 128), empty), empty).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_paper_optimizer_decay_affects_only_trainable_non_norm_weights(self):
        model = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3), nn.Linear(3, 1))
        model[2].weight.requires_grad_(False)
        for parameter in model.parameters():
            nn.init.ones_(parameter)
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        settings = {"lr": 0.1, "betas": [0.937, 0.999], "weight_decay": 0.2}
        optimizer = build_optimizer(model, settings, "paper_v716")
        optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual({id(parameter) for parameter in optimized}, {id(parameter) for parameter in expected})
        self.assertEqual(len(optimized), len(expected))
        optimizer.step()
        torch.testing.assert_close(model[0].weight, torch.full_like(model[0].weight, 0.98))
        for parameter in (model[0].bias, model[1].weight, model[1].bias, model[2].bias, model[2].weight):
            torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0., atol=0.)
        legacy = build_optimizer(model, settings)
        self.assertEqual(len(legacy.param_groups), 1)
        self.assertEqual(legacy.param_groups[0]["weight_decay"], 0.2)


if __name__ == "__main__":
    unittest.main()

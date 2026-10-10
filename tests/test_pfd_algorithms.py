import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rtdetr_pytorch"))

from src.nn.backbone.pfd_modules import Haar2D, SSWR, RDPF, PlainResidual, ScaleStructure
from src.nn.backbone.pfd_modules import protective_gate, spatial_permutation
from src.nn.backbone.presnet import PResNet
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.pfd_model import build_model, build_criterion
from src.zoo.rtdetr.pfd_support import center_targets, balanced_support_loss, support_loss


torch.set_num_threads(2)


def targets():
    return [{"boxes": torch.tensor([[0.45, 0.55, 0.2, 0.3], [0.75, 0.25, 0.1, 0.1]]),
             "labels": torch.zeros(2, dtype=torch.long)},
            {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}]


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_haar_reconstruction_even_and_odd_with_gradients(self):
        module = Haar2D(3)
        for h, w in ((12, 10), (13, 11), (1, 3)):
            x = torch.randn(2, 3, h, w, requires_grad=True)
            bands, size = module.analysis(x)
            restored = module.synthesis(bands, size)
            torch.testing.assert_close(restored, x, rtol=1e-6, atol=5e-7)
            restored.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0., atol=5e-7)

    def test_parameter_counts(self):
        count = lambda module: sum(p.numel() for p in module.parameters())
        self.assertEqual(count(SSWR()), 11709)
        self.assertEqual(count(SSWR(mode="single")), 11483)
        self.assertEqual(count(RDPF()), 50498)
        self.assertEqual(count(PlainResidual()), 11813)

    def test_structure_constant_input_and_range(self):
        module = ScaleStructure(4)
        torch.testing.assert_close(module.statistics(torch.ones(2, 4, 9, 11)),
                                   torch.zeros(2, 6, 5, 6), rtol=0., atol=0.)
        x = torch.randn(2, 4, 9, 11, requires_grad=True)
        stats = module.statistics(x)
        self.assertTrue((stats >= 0).all() and (stats <= 1).all())
        stats.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_sswr_identity_and_second_step_branch_gradient(self):
        module = SSWR().train()
        x = torch.randn(2, 128, 13, 15, requires_grad=True)
        y = module(x)
        torch.testing.assert_close(x, y, rtol=0., atol=0.)
        y.square().mean().backward()
        self.assertGreater(module.alpha.grad.abs().sum().item(), 0.)
        self.assertEqual(module.expand.weight.grad.abs().sum().item(), 0.)
        with torch.no_grad():
            module.alpha.add_(-0.1 * module.alpha.grad)
        module.zero_grad(set_to_none=True)
        module(x).square().mean().backward()
        self.assertGreater(module.expand.weight.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(module.structure.scale_logits.weight.grad).all())

    def test_protective_gate_relations(self):
        shallow = torch.tensor([0., 0.2, 0.9, 1.])
        deep = torch.tensor([0.5, 0.8, 0.1, 0.5])
        gate, u, difference, mixing = protective_gate(shallow, deep)
        torch.testing.assert_close(gate[[0, 3]], 0.1 + 0.9 * shallow[[0, 3]])
        self.assertEqual(difference[1].item(), 0.)
        self.assertEqual(mixing[1].item(), u[1].item())
        self.assertTrue((gate >= 0.1).all() and (gate <= 1.).all())
        semantic = protective_gate(shallow, deep, mode="semantic")[0]
        torch.testing.assert_close(semantic, 0.1 + 0.9 * deep)
        torch.testing.assert_close(protective_gate(shallow, deep, mode="ungated")[0],
                                   torch.ones_like(shallow))
        absolute = protective_gate(shallow, deep, mode="absolute")[3]
        self.assertGreater(absolute[1].item(), mixing[1].item())

    def test_support_target_peaks_overlap_and_empty_image(self):
        items = targets()
        result = center_targets(items, 16, 20, "cpu")
        self.assertEqual(result.shape, (2, 1, 16, 20))
        self.assertEqual(result[0, 0, 8, 9].item(), 1.)
        self.assertEqual(result[0, 0, 4, 15].item(), 1.)
        self.assertEqual(result[1].sum().item(), 0.)
        self.assertEqual(result[0, 0, 0, 0].item(), 0.)
        duplicate = [{"boxes": items[0]["boxes"].repeat(2, 1)}]
        torch.testing.assert_close(center_targets(duplicate, 16, 20, "cpu"), result[:1])
        tiny = [{"boxes": torch.tensor([[0., 1., 0.001, 0.001]])}]
        self.assertEqual(center_targets(tiny, 16, 20, "cpu").sum().item(), 1.)
        for mode in ("binary_box", "center_cell", "gaussian"):
            alternative = center_targets(items, 16, 20, "cpu", target=mode)
            self.assertTrue(torch.isfinite(alternative).all())
        invalid = [{"boxes": torch.tensor([[0.5, 0.5, 0., 0.1]])}]
        with self.assertRaises(ValueError):
            center_targets(invalid, 16, 20, "cpu")

    def test_balanced_loss_empty_full_and_extreme_logits(self):
        logits = torch.tensor([[[[-3., 2.], [7., -4.]]]], requires_grad=True)
        for target, expected in ((torch.zeros_like(logits), F.softplus(logits).mean()),
                                 (torch.ones_like(logits), F.softplus(-logits).mean())):
            torch.testing.assert_close(balanced_support_loss(logits, target), expected)
        extremes = torch.tensor([[[[-1000., 1000.]]]], requires_grad=True)
        loss = balanced_support_loss(extremes, torch.tensor([[[[0.2, 0.8]]]]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(extremes.grad).all())
        target = torch.rand_like(logits)
        pair = (logits, -logits)
        torch.testing.assert_close(support_loss(pair, target, "balanced"),
                                   (balanced_support_loss(pair[0], target)
                                    + balanced_support_loss(pair[1], target)) / 2.)
        torch.testing.assert_close(support_loss(pair, target, "shallow_only"),
                                   balanced_support_loss(pair[0], target))

    def test_rdpf_alignment_identity_and_support_gradient(self):
        module = RDPF().train()
        base = torch.randn(2, 256, 7, 9, requires_grad=True)
        shallow = torch.randn(2, 64, 15, 17)
        deep = torch.randn(2, 256, 4, 5)
        output, logits = module(base, shallow, deep)
        torch.testing.assert_close(output, base, rtol=0., atol=0.)
        self.assertEqual(logits[0].shape, (2, 1, 7, 9))
        loss = output.square().mean() + support_loss(logits, torch.rand_like(logits[0]))
        loss.backward()
        self.assertGreater(module.beta.grad.abs().sum().item(), 0.)
        self.assertGreater(module.shallow_head.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(module.deep_head.weight.grad.abs().sum().item(), 0.)

    def test_interventions_local_rng_and_vector_integrity(self):
        x = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).float()
        state = torch.random.get_rng_state()
        shuffled = spatial_permutation(x, 5)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        torch.testing.assert_close(shuffled, spatial_permutation(x, 5), rtol=0., atol=0.)
        self.assertFalse(torch.equal(shuffled, spatial_permutation(x, 6)))
        torch.testing.assert_close(shuffled[:, 1] - shuffled[:, 0], torch.full((2, 5, 7), 35.))
        torch.testing.assert_close(shuffled.flatten(2).sort().values, x.flatten(2).sort().values)
        module = SSWR().eval()
        module.capture = True
        u = torch.randn(2, 128, 12, 14)
        module(u)
        weights = module.diagnostics["scale_weights"].clone()
        module.intervention = "scale_shift"
        module(u)
        torch.testing.assert_close(module.diagnostics["scale_weights"], weights.roll((0, 1), (-2, -1)))
        module.intervention = "scale_uniform"
        module(u)
        torch.testing.assert_close(module.diagnostics["scale_weights"], torch.full_like(weights, 1. / 3.))
        module.intervention = "structure_zero"
        module(u)
        self.assertEqual(module.diagnostics["all_descriptors"].abs().sum().item(), 0.)

    def test_rdpf_frozen_interventions(self):
        module = RDPF().eval()
        module.capture = True
        module.beta.data.fill_(0.2)
        inputs = (torch.randn(2, 256, 8, 10), torch.randn(2, 64, 16, 20),
                  torch.randn(2, 256, 8, 10))
        original, _ = module(*inputs)
        shallow = module.diagnostics["shallow_support"].clone()
        deep = module.diagnostics["deep_support"].clone()
        gate = module.diagnostics["gate"].clone()
        state = torch.random.get_rng_state()
        for intervention in ("shallow_support_shuffle", "deep_support_shuffle", "independent_both_shuffle",
                             "ambiguity_only", "gate_mean", "gate_one", "bypass"):
            module.intervention = intervention
            module.intervention_seed = 23
            result, _ = module(*inputs)
            self.assertTrue(torch.isfinite(result).all())
            if intervention == "bypass":
                torch.testing.assert_close(result, inputs[0], rtol=0., atol=0.)
            else:
                self.assertFalse(torch.equal(result, original))
            if intervention in ("shallow_support_shuffle", "independent_both_shuffle"):
                torch.testing.assert_close(module.diagnostics["shallow_support"], spatial_permutation(shallow, 23))
            if intervention in ("deep_support_shuffle", "independent_both_shuffle"):
                torch.testing.assert_close(module.diagnostics["deep_support"], spatial_permutation(deep, 23, 1))
            if intervention == "gate_mean":
                torch.testing.assert_close(module.diagnostics["gate"], gate.mean((-2, -1), keepdim=True).expand_as(gate))
            if intervention == "gate_one":
                torch.testing.assert_close(module.diagnostics["gate"], torch.ones_like(gate))
            if intervention == "ambiguity_only":
                torch.testing.assert_close(module.diagnostics["mixing"], module.diagnostics["u"])
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.config = {"image_size": 128, "num_queries": 20, "num_denoising": 10}

    def test_official_baseline_forward_equivalence_and_cam_position(self):
        model = build_model({**self.config, "variant": "baseline"}).eval()
        backbone = PResNet(18, return_idx=[0, 1, 2, 3], freeze_norm=False, pretrained=False).eval()
        backbone.load_state_dict(model.backbone.state_dict(), strict=True)
        encoder = HybridEncoder(in_channels=[128, 256, 512], expansion=0.5,
                                 eval_spatial_size=[128, 128]).eval()
        encoder.load_state_dict(model.encoder.state_dict(), strict=True)
        images = torch.randn(2, 3, 128, 128)
        seen = []
        hook = model.encoder.cam_fusion.register_forward_hook(lambda m, i, o: seen.append(o.shape))
        with torch.no_grad():
            features = model.backbone(images)
            reference_features = backbone(images)
            for actual, expected in zip(features, reference_features):
                torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
            actual, support = model.encoder(features)
            reference = encoder(reference_features[1:])
            self.assertIsNone(support)
            for result, expected in zip(actual, reference):
                torch.testing.assert_close(result, expected, rtol=0., atol=0.)
        hook.remove()
        self.assertEqual(seen, [torch.Size([2, 512, 8, 8])])

    def test_zero_initialized_full_model_preserves_baseline_outputs(self):
        baseline = build_model({**self.config, "variant": "baseline"}).eval()
        full = build_model({**self.config, "variant": "full"}).eval()
        status = full.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(status.unexpected_keys)
        self.assertTrue(all(".sswr." in key or ".rdpf." in key for key in status.missing_keys))
        images = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            reference, actual = baseline(images), full(images)
        for key in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(actual[key], reference[key], rtol=0., atol=0.)

    def test_six_arms_forward_backward_and_optimizer(self):
        images, annotations = torch.randn(2, 3, 128, 128), targets()
        for variant in ("baseline", "sswr", "support", "sswr_support", "rdpf", "full"):
            with self.subTest(variant=variant):
                config = {**self.config, "variant": variant}
                model, criterion = build_model(config).train(), build_criterion(config)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                output = model(images, annotations)
                self.assertEqual(output["pred_boxes"].shape, (2, 20, 4))
                losses = criterion(output, annotations)
                self.assertEqual("loss_support" in losses, variant not in ("baseline", "sswr"))
                self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                total = sum(losses.values())
                total.backward()
                missing = [name for name, param in model.named_parameters()
                           if param.requires_grad and param.grad is None]
                self.assertEqual(missing, [])
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_internal_variants_forward_backward(self):
        inputs = (torch.randn(2, 256, 8, 8), torch.randn(2, 64, 16, 16), torch.randn(2, 256, 8, 8))
        for mode in ("full", "uniform", "single", "no_descriptors"):
            module = SSWR(mode=mode)
            module(torch.randn(2, 128, 8, 8)).mean().backward()
            self.assertTrue(torch.isfinite(module.alpha.grad).all())
        for mode in ("full", "ungated", "semantic", "ambiguity", "absolute", "aux_only"):
            module = RDPF(mode=mode)
            output, logits = module(*inputs)
            (output.square().mean() + support_loss(logits, torch.rand_like(logits[0]))).backward()
            self.assertTrue(torch.isfinite(module.shallow_head.weight.grad).all())
        for loss in ("balanced", "mean_bce", "shallow_only", "deep_only"):
            logits = [torch.randn(2, 1, 8, 8, requires_grad=True) for _ in range(2)]
            value = support_loss(logits, torch.rand(2, 1, 8, 8), loss)
            value.backward()
            self.assertTrue(torch.isfinite(value))
        criterion = build_criterion({"variant": "full", "rdpf": {"mode": "no_support_loss"}})
        self.assertEqual(criterion.support_gain, 0.)

    def test_all_empty_detection_batch(self):
        config = {**self.config, "variant": "full"}
        model, criterion = build_model(config).train(), build_criterion(config)
        empty = [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)} for _ in range(2)]
        output = model(torch.randn(2, 3, 128, 128), empty)
        losses = criterion(output, empty)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        sum(losses.values()).backward()
        self.assertGreater(model.encoder.rdpf.shallow_head.bias.grad.abs().item(), 0.)

    def test_main_configuration_640_forward_and_exact_increment(self):
        baseline = build_model({"variant": "baseline"}).eval()
        full = build_model({"variant": "full"}).eval()
        count = lambda model: sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(count(full) - count(baseline), 62207)
        full.backbone.sswr.alpha.data.fill_(0.1)
        full.encoder.rdpf.beta.data.fill_(0.1)
        seen = []
        hook = full.encoder.cam_fusion.register_forward_hook(lambda m, i, o: seen.append(o.shape))
        with torch.no_grad():
            output = full(torch.randn(1, 3, 640, 640))
        hook.remove()
        self.assertEqual(output["pred_logits"].shape, (1, 300, 1))
        self.assertEqual(output["pred_boxes"].shape, (1, 300, 4))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))
        self.assertEqual(seen, [torch.Size([1, 512, 40, 40])])


if __name__ == "__main__":
    unittest.main(verbosity=2)

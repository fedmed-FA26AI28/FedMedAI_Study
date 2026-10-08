"""Offline checks for simplified head objectives and capped local fitting."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.coverage import (
    CALIBRATION_METHODS, HEAD_METHODS, head_proximal_penalty, head_reference,
    relative_head_row_weights, validate_coverage_options,
)
from client.train import train_local


NEW_METHODS = ("calibrated_uniform_head", "relative_coverage_calibrated")


class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 4)
        self.fc = nn.Linear(4, 3)

    def forward(self, images):
        return self.fc(torch.tanh(self.encoder(images)))


class RecordingDataset(Dataset):
    def __init__(self, labels=None):
        self.labels = np.asarray([0, 0, 0, 0, 1, 1, 0] if labels is None else labels)
        self.images = torch.arange(2 * len(self.labels), dtype=torch.float32).reshape(-1, 2) / 7
        self.seen = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        self.seen.append(int(index))
        return self.images[index], int(self.labels[index])


class CoverageExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _fit(self, algorithm="calibrated_uniform_head", *, labels=None, epochs=2,
             shuffle=False, batch_sampler=None, **options):
        torch.manual_seed(43)
        model = SmallClassifier()
        dataset = RecordingDataset(labels)
        generator = torch.Generator().manual_seed(71)
        if batch_sampler is None:
            loader = DataLoader(dataset, batch_size=3, shuffle=shuffle, generator=generator)
        else:
            loader = DataLoader(dataset, batch_sampler=batch_sampler, generator=generator)
        diagnostics = {}
        settings = {"head_mu": 1.7, "logit_tau": 1.0, "prior_smoothing": 1.0}
        settings.update(options)
        result = train_local(model, loader, epochs, 0.02, 0.01, torch.device("cpu"),
                             algorithm=algorithm, diagnostics=diagnostics, **settings)
        return model, result, diagnostics, dataset.seen, generator.get_state()

    def _assert_same_fit(self, first, second):
        for expected, actual in zip(first[0].parameters(), second[0].parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_relative_weights_match_numeric_example_and_scale_invariance(self):
        counts = np.asarray([99., 9., 0.])
        expected = np.asarray([12., 36., 45.]) / 31
        weights = relative_head_row_weights(counts)
        np.testing.assert_allclose(weights, expected, rtol=1e-14, atol=0)
        self.assertAlmostEqual(weights.mean(), 1.0, places=15)
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[1], weights[0])
        for scale in (2, 1e100, 1e-100):
            with self.subTest(scale=scale):
                np.testing.assert_allclose(relative_head_row_weights(scale * counts),
                                           weights, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(relative_head_row_weights([8, 8, 8]), np.ones(3))
        np.testing.assert_array_equal(relative_head_row_weights([9]), np.ones(1))

    def test_relative_weights_remain_finite_when_naive_count_sum_overflows(self):
        weights = relative_head_row_weights([1e308, 1e308, 0])
        np.testing.assert_allclose(weights, [2 / 3, 2 / 3, 5 / 3])
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights > 0).all())

    def test_relative_weights_reject_empty_zero_and_invalid_counts(self):
        for counts in ([], [0], [0, 0, 0], [-1, 2], [1, np.nan], [np.inf, 2],
                       [[1, 2]], [1, "invalid"]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                relative_head_row_weights(counts)

    def test_relative_head_penalty_gradient_has_expected_rows_and_no_encoder_term(self):
        torch.manual_seed(5)
        model = SmallClassifier().double()
        reference = head_reference(model)
        with torch.no_grad():
            model.fc.weight.add_(2)
            model.fc.bias.add_(torch.tensor([1., -1., 3.], dtype=torch.float64))
            model.encoder.weight.add_(100)
        weights = relative_head_row_weights([99, 9, 0])
        penalty = head_proximal_penalty(model, reference, weights, 2.0)
        gradients = torch.autograd.grad(penalty, tuple(model.parameters()), allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        expected = torch.tensor([12., 36., 45.], dtype=torch.float64) / 31
        torch.testing.assert_close(gradients[2], (4 * expected)[:, None].expand_as(model.fc.weight))
        torch.testing.assert_close(gradients[3], 2 * expected * torch.tensor([1., -1., 3.]))
        self.assertAlmostEqual(penalty.item(), float((expected * torch.tensor([17., 17., 25.])).sum()))

    def test_new_methods_belong_to_both_head_and_calibration_groups(self):
        for name in NEW_METHODS:
            with self.subTest(name=name):
                self.assertIn(name, HEAD_METHODS)
                self.assertIn(name, CALIBRATION_METHODS)
                for field, values in {
                        "head_mu": (-1, float("nan"), float("inf"), True),
                        "logit_tau": (-1, float("nan"), float("inf"), True),
                        "prior_smoothing": (0, -1, float("nan"), float("inf"), True),
                }.items():
                    for value in values:
                        with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                            validate_coverage_options(name, **{field: value})

    def test_new_methods_ignore_kappa_in_validation_and_actual_training(self):
        for name in NEW_METHODS:
            expected = self._fit(name)
            for kappa in (0, -1, float("nan"), float("inf"), None):
                with self.subTest(name=name, kappa=kappa):
                    validate_coverage_options(name, coverage_kappa=kappa)
                    self._assert_same_fit(expected, self._fit(name, coverage_kappa=kappa))
        # The validation contracts of already executed methods remain unchanged.
        for name in ("head_prox", "coverage_prox", "coverage_calibrated",
                     "logit_calibration", "calibrated_fedprox"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_coverage_options(name, coverage_kappa=0)

    def test_uniform_method_exactly_matches_previous_fp32_uniform_control(self):
        expected = self._fit("coverage_calibrated", coverage_kappa=1e12)
        self._assert_same_fit(expected, self._fit("calibrated_uniform_head"))

    def test_balanced_relative_weights_recover_exact_uniform_training(self):
        labels = [0, 1, 2, 0, 1, 2]
        self._assert_same_fit(self._fit("calibrated_uniform_head", labels=labels),
                              self._fit("relative_coverage_calibrated", labels=labels))

    def test_zero_head_coefficient_recovers_calibration_for_fixed_and_custom_prior(self):
        for tau, smoothing in ((1.0, 1.0), (0.7, 2.5)):
            expected = self._fit("logit_calibration", head_mu=0,
                                 logit_tau=tau, prior_smoothing=smoothing)
            for name in NEW_METHODS:
                with self.subTest(name=name, tau=tau, smoothing=smoothing):
                    self._assert_same_fit(expected, self._fit(name, head_mu=0,
                                          logit_tau=tau, prior_smoothing=smoothing))

    def test_train_objective_gradients_match_explicit_calibration_and_head_formula(self):
        for name in NEW_METHODS:
            with self.subTest(name=name):
                torch.manual_seed(37)
                model = SmallClassifier().double()
                initial = copy.deepcopy(model)
                data = RecordingDataset()
                data.images = data.images.double()
                loader = DataLoader(data, batch_size=3)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
                before = []
                gradients = []

                def record_step(optim, args, kwargs):
                    before.append(copy.deepcopy(model))
                    gradients.append([parameter.grad.clone() for parameter in model.parameters()])

                handle = optimizer.register_step_pre_hook(record_step)
                try:
                    train_local(model, loader, 2, 0.04, 0, torch.device("cpu"),
                                optimizer=optimizer, algorithm=name, head_mu=1.7, max_steps=2)
                finally:
                    handle.remove()
                self.assertEqual(len(before), 2)
                # The second step has nonzero displacement from the round reference.
                second = before[1]
                logits = second(data.images[3:6])
                labels = torch.tensor(data.labels[3:6])
                ce = F.cross_entropy(logits + torch.log(torch.tensor([6., 3., 1.], dtype=torch.float64)), labels)
                weights = torch.ones(3, dtype=torch.float64)
                if name == "relative_coverage_calibrated":
                    weights = 1 / (1 + 3 * torch.tensor([5., 2., 0.], dtype=torch.float64) / 7)
                    weights = weights / weights.mean()
                distances = (second.fc.weight - initial.fc.weight.detach()).square().sum(1)
                distances = distances + (second.fc.bias - initial.fc.bias.detach()).square()
                objective = ce + 1.7 / 2 * (weights * distances).sum()
                expected = torch.autograd.grad(objective, tuple(second.parameters()))
                for actual, gradient in zip(gradients[1], expected):
                    torch.testing.assert_close(actual, gradient, rtol=1e-12, atol=1e-12)

    def test_max_steps_caps_across_epochs_without_fetching_an_extra_batch(self):
        prefix_batches = [[0, 1, 2], [3, 4, 5], [6], [0, 1, 2]]
        for name in ("fedavg",) + NEW_METHODS:
            with self.subTest(name=name):
                capped = self._fit(name, epochs=3, max_steps=4)
                expected = self._fit(name, epochs=1, batch_sampler=prefix_batches)
                self._assert_same_fit(expected, capped)
                self.assertEqual(capped[2]["optimizer_steps"], 4)
                self.assertEqual(capped[2]["train_samples_seen"], 10)
                self.assertEqual(capped[3], [0, 1, 2, 3, 4, 5, 6, 0, 1, 2])

    def test_default_and_oversized_step_cap_have_exact_same_updates(self):
        for name in ("fedavg", "fedprox") + NEW_METHODS:
            with self.subTest(name=name):
                options = {"proximal_mu": 0.3} if name == "fedprox" else {}
                expected = self._fit(name, **options)
                self._assert_same_fit(expected, self._fit(name, max_steps=None, **options))
                self._assert_same_fit(expected, self._fit(name, max_steps=6, **options))
                self._assert_same_fit(expected, self._fit(name, max_steps=100, **options))
                self.assertEqual(expected[2]["optimizer_steps"], 6)
                self.assertEqual(expected[2]["train_samples_seen"], 14)

    def test_capped_shuffled_fit_is_reproducible(self):
        first = self._fit("relative_coverage_calibrated", epochs=3, max_steps=4, shuffle=True)
        second = self._fit("relative_coverage_calibrated", epochs=3, max_steps=4, shuffle=True)
        self._assert_same_fit(first, second)
        self.assertEqual(first[3], second[3])
        torch.testing.assert_close(first[4], second[4], rtol=0, atol=0)

    def test_partial_last_batch_reports_actual_seen_samples(self):
        _, _, diagnostics, seen, _ = self._fit(max_steps=3, epochs=10)
        self.assertEqual(diagnostics["optimizer_steps"], 3)
        self.assertEqual(diagnostics["train_samples_seen"], 7)
        self.assertEqual(seen, list(range(7)))

    def test_capped_metrics_use_seen_samples_but_prior_uses_selected_train_dataset(self):
        torch.manual_seed(11)
        model = SmallClassifier()
        data = RecordingDataset()
        loader = DataLoader(data, batch_size=3)
        with torch.no_grad():
            logits = model(data.images[:3])
            labels = torch.tensor(data.labels[:3])
            raw = F.cross_entropy(logits, labels).item()
            # Full selected training counts are [5, 2, 0], although the cap
            # exposes only three class-zero examples to the optimizer.
            calibrated = F.cross_entropy(logits + torch.log(torch.tensor([6., 3., 1.])), labels).item()
        diagnostics = {}
        loss, accuracy = train_local(model, loader, 10, 0.0, 0.0, torch.device("cpu"),
                                     algorithm="calibrated_uniform_head", max_steps=1,
                                     diagnostics=diagnostics)
        self.assertEqual(loss, raw)
        self.assertEqual(accuracy, float((logits.argmax(1) == labels).float().mean()))
        self.assertEqual(diagnostics["train_objective_loss"], calibrated)
        self.assertEqual(diagnostics["train_head_penalty"], 0)
        self.assertEqual(diagnostics["optimizer_steps"], 1)
        self.assertEqual(diagnostics["train_samples_seen"], 3)

    def test_invalid_max_steps_rejected_before_any_optimizer_update(self):
        data = RecordingDataset()
        loader = DataLoader(data, batch_size=3)
        model = SmallClassifier()
        initial = [parameter.detach().clone() for parameter in model.parameters()]
        for cap in (0, -1, 1.5, 1.0, True, False, "2", float("nan"), float("inf")):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "max_steps"):
                train_local(model, loader, 2, 0.02, 0.01, torch.device("cpu"), max_steps=cap)
        self.assertEqual(data.seen, [])
        for actual, expected in zip(model.parameters(), initial):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # NumPy integers are valid protocol inputs as well as Python integers.
        self.assertEqual(self._fit(max_steps=np.int64(2))[2]["optimizer_steps"], 2)


if __name__ == "__main__":
    unittest.main()

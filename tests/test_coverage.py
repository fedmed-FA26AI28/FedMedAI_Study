"""Offline tests for train-count classifier stabilization and calibration."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import get_strategy
from algorithms.coverage import (
    CALIBRATION_METHODS, COVERAGE_METHODS, HEAD_METHODS, KAPPA_FREE_METHODS, head_proximal_penalty,
    head_reference, head_row_weights, logit_adjustment, training_class_counts,
)
from client.client import FedMedClient
from client.evaluate import evaluate_classification
from client.train import train_local
from experiments.artifacts import load_config, seed_everything
from experiments.simulation import run_federated_simulation
from models.cnn import get_parameters


class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 2)
        self.fc = nn.Linear(2, 3)

    def forward(self, images):
        return self.fc(self.encoder(images))


class MetadataOnlyDataset(Dataset):
    def __init__(self, labels):
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        raise AssertionError("Class counting must not load images or transforms")


class SyntheticCoverageDataset(Dataset):
    def __init__(self, size, seed):
        self.labels = np.arange(size) % 8
        self.images = torch.randn(size, 3, 8, 8, generator=torch.Generator().manual_seed(seed))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], int(self.labels[index])


class CoverageMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_counts_resolve_nested_subsets_without_loading_or_shuffling(self):
        data = MetadataOnlyDataset([[0], [1], [2], [0], [1], [2]])
        first = Subset(data, [4, 2, 0, 3])
        selected = Subset(first, torch.tensor([3, 2, 3]))
        loader = DataLoader(selected, shuffle=True, generator=torch.Generator().manual_seed(27))
        state = loader.generator.get_state().clone()
        self.assertEqual(training_class_counts(loader.dataset, 4).tolist(), [3, 0, 0, 0])
        torch.testing.assert_close(loader.generator.get_state(), state, rtol=0, atol=0)
        self.assertEqual(first.indices, [4, 2, 0, 3])
        self.assertEqual(selected.indices.tolist(), [3, 2, 3])

    def test_tensor_targets_absent_classes_and_invalid_metadata(self):
        data = TensorDataset(torch.zeros(4, 1), torch.tensor([[1], [1], [2], [1]]))
        self.assertEqual(training_class_counts(Subset(data, [0, 2]), 4).tolist(), [0, 1, 1, 0])
        self.assertEqual(training_class_counts(Subset(data, []), 4).tolist(), [0, 0, 0, 0])
        for labels in ([0, -1], [0, 4], [0, 0.5], [0, np.nan]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                training_class_counts(MetadataOnlyDataset(labels), 4)
        with self.assertRaisesRegex(ValueError, "stored"):
            training_class_counts([0, 1], 2)

    def test_penalty_gradient_is_exactly_row_weighted_and_excludes_encoder(self):
        model = SmallClassifier().double()
        reference = head_reference(model)
        with torch.no_grad():
            model.fc.weight.add_(2)
            model.fc.bias.add_(3)
            model.encoder.weight.add_(100)
        weights = head_row_weights([0, 32, 320], 32)
        np.testing.assert_allclose(weights, [1, 0.5, 1 / 11])
        penalty = head_proximal_penalty(model, reference, weights, 2.0)
        gradients = torch.autograd.grad(penalty, tuple(model.parameters()), allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        expected = torch.as_tensor(weights, dtype=torch.float64)
        torch.testing.assert_close(gradients[2], (4 * expected)[:, None].expand_as(model.fc.weight))
        torch.testing.assert_close(gradients[3], 6 * expected)
        self.assertAlmostEqual(penalty.item(), 17 * weights.sum())
        # The reference remains detached and unchanged after mutations to the model.
        self.assertFalse(reference[0].requires_grad)
        torch.testing.assert_close(model.fc.weight - reference[0], torch.full_like(model.fc.weight, 2))

    def test_uniform_head_penalty_and_zero_weight_row_have_expected_gradients(self):
        model = SmallClassifier()
        reference = head_reference(model)
        with torch.no_grad():
            model.fc.weight.add_(1)
            model.fc.bias.add_(1)
        uniform = head_row_weights([0, 32, 320], 32, uniform=True)
        np.testing.assert_array_equal(uniform, np.ones(3))
        loss = head_proximal_penalty(model, reference, [1, 0.5, 0], 4)
        weight_grad, bias_grad = torch.autograd.grad(loss, (model.fc.weight, model.fc.bias))
        torch.testing.assert_close(weight_grad, torch.tensor([[4., 4.], [2., 2.], [0., 0.]]))
        torch.testing.assert_close(bias_grad, torch.tensor([4., 2., 0.]))

    def test_calibration_gradient_formula_and_absent_class_finiteness(self):
        logits = torch.tensor([[0.2, 0.3, -0.4], [1.0, -0.2, 0.1]],
                              dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 2])
        counts = [0, 100, 1]
        offset = logit_adjustment(counts, 32, 1.0, device="cpu", dtype=logits.dtype)
        torch.testing.assert_close(offset, torch.log(torch.tensor([32., 132., 33.], dtype=logits.dtype)))
        gradient, = torch.autograd.grad(F.cross_entropy(logits + offset, labels), (logits,))
        expected = ((logits + offset).softmax(dim=1) - F.one_hot(labels, 3)) / 2
        torch.testing.assert_close(gradient, expected)
        self.assertTrue(torch.isfinite(gradient).all())

    def _local_fit(self, algorithm=None, head_mu=1.0, logit_tau=1.0, proximal_mu=0.0,
                   coverage_kappa=2.0, prior_smoothing=1.0):
        seed_everything(19)
        model = SmallClassifier()
        data = TensorDataset(torch.tensor([[1.], [2.], [3.], [4.]]), torch.tensor([0, 0, 1, 0]))
        loader = DataLoader(data, batch_size=1)
        diagnostics = {}
        result = train_local(model, loader, 2, 0.03, 0.0, torch.device("cpu"),
                             proximal_mu, None, algorithm=algorithm, head_mu=head_mu,
                             coverage_kappa=coverage_kappa, logit_tau=logit_tau,
                             prior_smoothing=prior_smoothing, diagnostics=diagnostics)
        return get_parameters(model), result, diagnostics

    def test_zero_coefficients_recover_exact_fedavg_updates_and_metrics(self):
        expected, expected_metrics, _ = self._local_fit()
        for algorithm in sorted(COVERAGE_METHODS):
            with self.subTest(algorithm=algorithm):
                actual, metrics, diagnostics = self._local_fit(algorithm, head_mu=0, logit_tau=0)
                for first, second in zip(actual, expected):
                    np.testing.assert_array_equal(first, second)
                self.assertEqual(metrics, expected_metrics)
                self.assertEqual(diagnostics["train_head_penalty"], 0)
                self.assertEqual(diagnostics["train_objective_loss"], metrics[0])

    def test_positive_head_penalty_changes_local_updates(self):
        plain, _, _ = self._local_fit()
        for algorithm in ("head_prox", "coverage_prox"):
            with self.subTest(algorithm=algorithm):
                penalized, _, diagnostics = self._local_fit(algorithm, head_mu=10)
                self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(plain, penalized)))
                self.assertGreater(diagnostics["train_head_penalty"], 0)
                self.assertEqual(len(diagnostics["head_displacement_norms"]), 3)
                self.assertGreater(diagnostics["head_displacement_l2"], 0)

    def test_calibrated_fedprox_reduces_to_scope_matched_controls(self):
        prox, _, _ = self._local_fit("fedprox", proximal_mu=3.0)
        calibrated_zero, _, _ = self._local_fit("calibrated_fedprox", proximal_mu=3.0, logit_tau=0)
        calibration, _, _ = self._local_fit("logit_calibration", logit_tau=1.0)
        no_prox, _, _ = self._local_fit("calibrated_fedprox", proximal_mu=0, logit_tau=1.0)
        full_prox, _, diagnostics = self._local_fit("calibrated_fedprox", proximal_mu=3.0, logit_tau=1.0)
        for expected, actual in zip(prox, calibrated_zero):
            np.testing.assert_array_equal(expected, actual)
        for expected, actual in zip(calibration, no_prox):
            np.testing.assert_array_equal(expected, actual)
        # Full-model stabilization changes the encoder as well as the classifier.
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(full_prox[:2], calibration[:2])))
        self.assertEqual(diagnostics["train_head_penalty"], 0)
        self.assertGreater(diagnostics["train_proximal_penalty"], 0)

    def test_prior_smoothing_and_coverage_kappa_control_independent_mechanisms(self):
        # Calibration-only updates are independent of the coverage weight kappa.
        first, _, _ = self._local_fit("logit_calibration", coverage_kappa=1)
        second, _, _ = self._local_fit("logit_calibration", coverage_kappa=100)
        for expected, actual in zip(first, second):
            np.testing.assert_array_equal(expected, actual)
        # Head-only stabilization is independent of the logit pseudocount epsilon.
        first, _, _ = self._local_fit("coverage_prox", head_mu=10, prior_smoothing=1)
        second, _, _ = self._local_fit("coverage_prox", head_mu=10, prior_smoothing=100)
        for expected, actual in zip(first, second):
            np.testing.assert_array_equal(expected, actual)
        # The combined method responds to each coefficient independently.
        combined, _, _ = self._local_fit("coverage_calibrated", head_mu=10)
        different_prior, _, _ = self._local_fit("coverage_calibrated", head_mu=10, prior_smoothing=100)
        different_weights, _, _ = self._local_fit("coverage_calibrated", head_mu=10, coverage_kappa=100)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(combined, different_prior)))
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(combined, different_weights)))

    def test_calibrated_training_reports_raw_ce_and_keeps_inference_raw(self):
        seed_everything(11)
        model = SmallClassifier()
        dataset = TensorDataset(torch.tensor([[1.], [2.], [3.]]), torch.tensor([0, 0, 0]))
        loader = DataLoader(dataset, batch_size=3)
        with torch.no_grad():
            logits = model(dataset.tensors[0])
            raw = F.cross_entropy(logits, dataset.tensors[1]).item()
            offset = logit_adjustment([3, 0, 0], 2, 1.0, device="cpu", dtype=logits.dtype)
            calibrated = F.cross_entropy(logits + offset, dataset.tensors[1]).item()
        diagnostics = {}
        loss, accuracy = train_local(model, loader, 1, 0.0, 0.0, torch.device("cpu"),
                                     algorithm="coverage_calibrated", coverage_kappa=2,
                                     prior_smoothing=2,
                                     diagnostics=diagnostics)
        self.assertEqual(loss, raw)
        self.assertEqual(accuracy, (logits.argmax(1) == dataset.tensors[1]).float().mean().item())
        self.assertEqual(diagnostics["train_objective_loss"], calibrated)
        self.assertNotEqual(calibrated, raw)
        report = evaluate_classification(model, loader, torch.device("cpu"), 3)
        self.assertAlmostEqual(report["loss"], raw, places=6)
        self.assertEqual(report["accuracy"], accuracy)

    def test_client_counts_use_only_train_metadata_and_do_not_transmit_counts(self):
        train = TensorDataset(torch.tensor([[1.], [2.], [3.]]), torch.tensor([0, 0, 1]))
        client = FedMedClient(0, SmallClassifier(), DataLoader(train, batch_size=3),
                              DataLoader(MetadataOnlyDataset([99, 99])), num_classes=3,
                              device=torch.device("cpu"), monitoring={"enabled": False})
        _, n, metrics = client.fit(get_parameters(client.model), {
            "round": 1, "algorithm": "coverage_calibrated", "lr": 0.0,
            "head_mu": 2.0, "coverage_kappa": 3.0, "logit_tau": 0.5,
            "prior_smoothing": 2.0,
            "proximal_mu": 100.0,
        })
        record = json.loads(metrics["telemetry_json"])
        self.assertEqual(n, 3)
        self.assertEqual(record["algorithm"], "coverage_calibrated")
        self.assertEqual(record["proximal_mu"], 0)
        self.assertEqual(record["head_mu"], 2)
        self.assertEqual(record["coverage_kappa"], 3)
        self.assertEqual(record["logit_tau"], 0.5)
        self.assertEqual(record["prior_smoothing"], 2)
        self.assertNotIn("class_counts", record)
        self.assertEqual(len(record["head_displacement_norms"]), 3)


class CoveragePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="coverage_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "config.yaml"
        self.base = {
            "federation": {"num_clients": 2, "num_rounds": 2, "seed": 23},
            "training": {"batch_size": 4, "learning_rate": 0.003},
            "dataset": {"download": False, "max_train_samples": 8,
                        "max_val_samples": 4, "max_test_samples": 9},
            "runtime": {"backend": "sequential"},
            "monitoring": {"enabled": False},
            "paths": {name + "_dir": str(self.root / name)
                      for name in ("results", "checkpoints", "runs")},
        }
        self.path.write_text(yaml.safe_dump(self.base), encoding="utf-8")
        self.loads = []

    def _fake_data(self, dataset_name="BloodMNIST", split="train", *args, **kwargs):
        self.loads.append(split)
        return SyntheticCoverageDataset(96 if split == "train" else 32,
                                        {"train": 1, "val": 2, "test": 3}[split])

    def test_invalid_active_coefficients_rejected_before_artifact_creation(self):
        cases = []
        for name in HEAD_METHODS:
            cases.extend((name, "head_mu", value) for value in (-1, float("nan"), float("inf"), True))
        for name in COVERAGE_METHODS - KAPPA_FREE_METHODS:
            cases.extend((name, "coverage_kappa", value) for value in (0, -1, float("nan"), float("inf"), True))
        for name in CALIBRATION_METHODS:
            cases.extend((name, "logit_tau", value) for value in (-1, float("nan"), float("inf"), True))
            cases.extend((name, "prior_smoothing", value) for value in (0, -1, float("nan"), float("inf"), True))
        cases.extend(("calibrated_fedprox", "proximal_mu", value)
                     for value in (-1, float("nan"), float("inf"), True))
        for name, field, value in cases:
            with self.subTest(name=name, field=field, value=value):
                config = {**self.base, "algorithm": {"name": name, field: value}}
                self.path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.assertRaises(ValueError):
                    run_federated_simulation(str(self.path))
        self.assertFalse((self.root / "results").exists())

    def test_named_methods_use_fedavg_aggregation_and_planned_slots_stay_unavailable(self):
        for name in ("fednova", "proposed"):
            with self.assertRaises(NotImplementedError):
                get_strategy(name)
        for name in COVERAGE_METHODS:
            config = load_config(self.path, {"algorithm": {"name": name}})
            self.assertEqual(config["algorithm"]["head_mu"], 1)
            self.assertEqual(config["algorithm"]["coverage_kappa"], 32)
            self.assertEqual(config["algorithm"]["logit_tau"], 1)
            self.assertEqual(config["algorithm"]["prior_smoothing"], 1)

    def test_full_synthetic_pipeline_preserves_selection_and_paired_samples(self):
        partitions, test_ids = set(), []
        for name in sorted(COVERAGE_METHODS):
            with self.subTest(name=name):
                self.loads.clear()
                with patch("datasets.partition.load_medmnist_split", side_effect=self._fake_data), \
                        patch("datasets.medmnist_code.load_medmnist_split", side_effect=self._fake_data):
                    result = run_federated_simulation(str(self.path), algorithm_override=name)
                self.assertEqual(self.loads.count("test"), 1)
                self.assertEqual(self.loads[-1], "test")
                self.assertEqual(result["selection_split"], "val")
                self.assertEqual(result["completed_rounds"], 2)
                self.assertEqual(result["algorithm"], name)
                path = Path(result["results_dir"])
                manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "completed")
                records = [json.loads(line) for line in (path / "client_metrics.jsonl").read_text().splitlines()]
                fits = [record for record in records if record["phase"] == "fit"]
                self.assertEqual(len(fits), 4)
                for record in fits:
                    self.assertEqual(record["algorithm"], name)
                    self.assertTrue(np.isfinite(record["train_objective_loss"]))
                    self.assertTrue(np.isfinite(record["train_head_penalty"]))
                    self.assertEqual(len(record["head_displacement_norms"]), 8)
                    self.assertTrue(np.isfinite(record["encoder_displacement_l2"]))
                    self.assertNotIn("class_counts", record)
                    self.assertEqual(record["proximal_mu"], 0.01 if name == "calibrated_fedprox" else 0)
                with (path / "client_metrics.csv").open(newline="", encoding="utf-8") as handle:
                    flat = list(csv.DictReader(handle))
                self.assertEqual(len(flat), 8)
                self.assertEqual([r["algorithm"] for r in flat if r["phase"] == "fit"], [name] * 4)
                with (path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                    rounds = list(csv.DictReader(handle))
                expected_round = max(rounds, key=lambda row: float(row["global_val_f1_macro"]))
                self.assertEqual(result["best_round"], int(expected_round["round"]))
                partitions.add(result["partition_sha256"])
                test_ids.append(json.loads((path / "evaluation/test_sample_ids.json").read_text())["sample_ids"])
        self.assertEqual(len(partitions), 1)
        self.assertTrue(all(ids == test_ids[0] for ids in test_ids))


if __name__ == "__main__":
    unittest.main()

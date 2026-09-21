"""Scientific protocol and artifact regression tests (offline, unittest only)."""

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
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedmedai.centralized import run_centralized
from fedmedai.evaluate import evaluate_classification
from fedmedai.experiment import RunArtifacts, load_config, seed_everything
from fedmedai.model import get_weights, set_weights
from fedmedai.monitoring.metrics import classification_metrics, client_fairness, straggler_metrics
from fedmedai.monitoring.resource import ResourceMonitor
from fedmedai.partition import create_client_dataloaders, partition_dirichlet_non_iid, partition_iid
from fedmedai.simulation import run_federated_simulation
from fedmedai.train import train_local


class SyntheticMedicalDataset(Dataset):
    def __init__(self, size=80, seed=0):
        self.labels = np.arange(size) % 8
        self.images = torch.randn(size, 3, 12, 12, generator=torch.Generator().manual_seed(seed))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], int(self.labels[index])


class MetricsTests(unittest.TestCase):
    def test_known_confusion_matrix_and_absent_class(self):
        report = classification_metrics([[2, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertAlmostEqual(report["accuracy"], 0.75)
        self.assertAlmostEqual(report["precision_macro"], 0.5)
        self.assertAlmostEqual(report["recall_macro"], 5 / 9)
        self.assertAlmostEqual(report["f1_macro"], (0.8 + 2 / 3) / 3)
        self.assertEqual(report["per_class"][2]["support"], 0)
        with self.assertRaises(ValueError):
            classification_metrics(np.zeros((3, 3)))

    def test_fairness_is_unweighted_and_straggler_is_explicit(self):
        metrics = client_fairness([{"accuracy": 1.0, "f1_macro": 0.8},
                                   {"accuracy": 0.0, "f1_macro": 0.0}])
        self.assertEqual(metrics["mean_client_accuracy"], 0.5)
        self.assertEqual(metrics["client_accuracy_variance"], 0.25)
        times = straggler_metrics([2, 4, 12])
        self.assertEqual(times["straggler_overhead_seconds"], 8)
        self.assertEqual(times["estimated_wait_seconds_sum"], 18)
        self.assertIsNone(client_fairness([])["worst_client_accuracy"])

    def test_error_sample_ids_are_original_subset_ids(self):
        data = TensorDataset(torch.tensor([[4., 0.], [0., 4.], [4., 0.], [0., 4.]]),
                             torch.tensor([0, 1, 1, 0]))
        loader = DataLoader(Subset(data, [3, 2]), batch_size=2, shuffle=False)
        report = evaluate_classification(nn.Identity(), loader, torch.device("cpu"), 2,
                                         max_failure_cases=1)
        self.assertEqual(report["confusion_matrix"], [[0, 1], [1, 0]])
        self.assertEqual(report["failure_cases"][0]["sample_id"], 3)
        self.assertEqual(len(report["failure_cases"]), 1)

    def test_energy_integrates_valid_intervals_without_bridging_gaps(self):
        monitor = ResourceMonitor(torch.device("cpu"), enabled=False)
        monitor.duration = 4
        monitor.samples = [
            {"time": t, "power_watts": p, "cpu_utilization_percent": 0,
             "process_rss_bytes": 1, "system_ram_used_bytes": 2,
             "gpu_utilization_percent": None, "temperature_c": None}
            for t, p in [(0, 10), (1, 20), (2, None), (4, 20)]]
        result = monitor._summarize()
        self.assertEqual(result["energy_joules_observed"], 15)
        self.assertEqual(result["energy_observed_seconds"], 1)
        self.assertIsNone(result["gpu_utilization_percent_mean"])
        with ResourceMonitor(torch.device("cpu"), enabled=False) as disabled:
            pass
        self.assertIsNone(disabled.result["energy_joules_observed"])


class TrainingPartitionTests(unittest.TestCase):
    def test_non_iid_is_reproducible_disjoint_complete_and_never_falls_back(self):
        data = SyntheticMedicalDataset(240)
        # Non-contiguous labels must not silently disappear.
        data.labels = np.where(data.labels % 2, 7, 2)
        a = partition_dirichlet_non_iid(data, 3, 0.3, 42, 2)
        self.assertEqual(a, partition_dirichlet_non_iid(data, 3, 0.3, 42, 2))
        flattened = [index for indices in a.values() for index in indices]
        self.assertEqual(sorted(flattened), list(range(len(data))))
        with self.assertRaises(ValueError):
            partition_dirichlet_non_iid(data, 3, 0.1, 42, 100)
        with self.assertRaises(ValueError):
            partition_dirichlet_non_iid(data, 3, 0.1, 42, 1, max_attempts=0)
        with self.assertRaises(ValueError):
            partition_iid(data, 300)

    def test_partitions_and_class_distributions_match_selected_samples(self):
        data = SyntheticMedicalDataset(240)
        with patch("fedmedai.partition.load_medmnist_split", return_value=data):
            train, val, metadata = create_client_dataloaders(
                partition_type="dirichlet", alpha=0.3, max_train_samples=12, max_val_samples=8)
        for cid in train:
            info = metadata["clients"][cid]
            self.assertEqual(sum(info["class_distribution"].values()), len(train[cid].dataset))
            self.assertEqual(sum(info["val_class_distribution"].values()), len(val[cid].dataset))
            self.assertEqual(info["sample_ids"]["train_sample_ids"], train[cid].dataset.indices)
        self.assertEqual(metadata["allocation_policy"], "shared_class_proportions_train_val")

    def test_fedprox_zero_matches_fedavg_and_positive_mu_changes_updates(self):
        loader = DataLoader(TensorDataset(torch.tensor([[1.], [2.], [3.], [4.]]),
                                          torch.tensor([0, 1, 0, 1])), batch_size=1)
        def trained(mu):
            seed_everything(7)
            model = nn.Linear(1, 2)
            train_local(model, loader, 2, 0.1, 0.0, torch.device("cpu"), proximal_mu=mu)
            return get_weights(model)
        base, zero, prox = trained(0), trained(0), trained(10)
        for x, y in zip(base, zero):
            np.testing.assert_array_equal(x, y)
        self.assertTrue(any(not np.allclose(x, y) for x, y in zip(base, prox)))

    def test_parameter_exchange_does_not_alias_model_or_accept_bad_shapes(self):
        model = nn.Linear(2, 2)
        weights = get_weights(model)
        weights[0][:] = 100
        self.assertFalse(torch.all(model.weight == 100))
        with self.assertRaises(ValueError):
            set_weights(model, weights[:-1])
        with self.assertRaises(ValueError):
            set_weights(model, [np.zeros((3, 3)), weights[1]])


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fedmedai_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_path = self.root / "config.yaml"
        self.config = {
            "federation": {"num_clients": 2, "num_rounds": 2, "seed": 11},
            "training": {"batch_size": 4, "centralized_epochs": 2},
            "dataset": {"download": False, "max_train_samples": 8,
                        "max_val_samples": 4, "max_test_samples": 9},
            "runtime": {"backend": "sequential"},
            "monitoring": {"enabled": False},
            "evaluation": {"target_accuracy": 1.0, "max_failure_cases": 3},
            "paths": {name + "_dir": str(self.root / name)
                      for name in ("results", "checkpoints", "runs")},
        }
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        self.calls = []

        def load_fake(dataset_name="BloodMNIST", split="train", *args, **kwargs):
            self.calls.append(split)
            return SyntheticMedicalDataset(80 if split == "train" else 32, seed={"train": 0, "val": 1, "test": 2}[split])
        for name in ("fedmedai.partition.load_medmnist_split", "fedmedai.dataset.load_medmnist_split"):
            patcher = patch(name, side_effect=load_fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_end_to_end_schema_selection_and_reproducibility(self):
        result = run_federated_simulation(str(self.config_path))
        self.assertEqual(self.calls.count("test"), 1)
        self.assertEqual(self.calls[-1], "test")
        self.assertEqual(result["selection_split"], "val")
        self.assertEqual(result["test_samples"], 9)
        self.assertEqual(result["completed_rounds"], 2)
        path = Path(result["results_dir"])
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        with (path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(float(row["global_val_accuracy"]), float(row["pooled_client_val_accuracy"]))
            self.assertAlmostEqual(float(row["global_val_f1_macro"]), float(row["pooled_client_val_f1_macro"]))
            self.assertGreater(int(row["round_model_payload_bytes"]), 0)
            self.assertEqual(row["client_train_energy_joules_observed_sum"], "")
            self.assertGreater(float(row["round_latency_seconds"]), float(row["global_validation_seconds"]))
        clients = [json.loads(line) for line in (path / "client_metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(clients), 8)
        self.assertIsNone(clients[0]["upload_seconds"])
        self.assertEqual(sum(sum(row) for row in result["test_evaluation"]["confusion_matrix"]), 9)
        again = run_federated_simulation(str(self.config_path))
        self.assertNotEqual(result["run_id"], again["run_id"])
        self.assertTrue(path.exists())
        self.assertEqual(result["partition_sha256"], again["partition_sha256"])
        self.assertEqual(result["test_evaluation"], again["test_evaluation"])
        self.assertEqual(result["best_round"], again["best_round"])

    def test_centralized_uses_same_training_partition_and_test_protocol(self):
        federated = run_federated_simulation(str(self.config_path), algorithm_override="fedprox")
        self.calls.clear()
        centralized = run_centralized(str(self.config_path))
        self.assertEqual(centralized["mode"], "centralized")
        self.assertEqual(centralized["total_model_payload_bytes"], 0)
        self.assertEqual(centralized["partition_sha256"], federated["partition_sha256"])
        self.assertEqual(centralized["train_samples"], federated["train_samples"])
        self.assertEqual(self.calls.count("test"), 1)
        self.assertEqual(centralized["selection_split"], "val")

    def test_zero_accuracy_still_saves_first_checkpoint(self):
        with patch("fedmedai.server.evaluate_classification") as evaluate:
            report = classification_metrics(np.roll(np.eye(8, dtype=int), 1, axis=1))
            report["loss"] = 2.0
            evaluate.return_value = report
            result = run_federated_simulation(str(self.config_path))
        self.assertEqual(result["best_round"], 1)
        self.assertEqual(result["best_validation_metrics"]["accuracy"], 0)

    def test_failed_runs_are_marked_and_never_reuse_old_best(self):
        with patch("fedmedai.simulation.prepare_data", side_effect=RuntimeError("bad partition")):
            with self.assertRaisesRegex(RuntimeError, "bad partition"):
                run_federated_simulation(str(self.config_path))
        manifests = list((self.root / "results").rglob("run_manifest.json"))
        self.assertEqual(len(manifests), 1)
        self.assertEqual(json.loads(manifests[0].read_text())["status"], "failed")
        self.assertFalse(list((self.root / "results").rglob("final_test_metrics.json")))

    def test_invalid_config_rejected_before_training(self):
        with self.assertRaises(ValueError):
            load_config(self.config_path, {"federation": {"partition_type": "typo"}})
        with self.assertRaises(ValueError):
            load_config(self.config_path, {"dataset": {"max_test_samples": 0}})
        with self.assertRaises(ValueError):
            RunArtifacts(load_config(self.config_path), "../escape")


if __name__ == "__main__":
    unittest.main()

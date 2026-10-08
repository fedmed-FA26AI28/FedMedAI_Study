"""Offline checks for root packages, deployment utilities and provenance."""
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

from algorithms import get_strategy
from algorithms.fedavg import FedMedFedAvg
from algorithms.fedprox import FedMedFedProx
from datasets.partition import visualize_partition
from experiments.artifacts import collect_source_hashes, load_config
from models.cnn import CNN, count_parameters
from monitoring.metrics import FLMetricsRecorder
from monitoring.network import ClientPingLogger, ping
from scripts.distribute_data import transfer_command


class LayoutTests(unittest.TestCase):
    def test_registry_selects_real_strategies_and_rejects_extension_slots(self):
        with patch.object(FedMedFedAvg, "__init__", return_value=None), \
                patch.object(FedMedFedProx, "__init__", return_value=None):
            for name, expected in (("fedavg", FedMedFedAvg), ("fedprox", FedMedFedProx)):
                artifacts = SimpleNamespace(config={"algorithm": {"name": name}})
                self.assertIs(type(get_strategy(name, None, None, artifacts)), expected)
            with self.assertRaises(ValueError):
                get_strategy("fedavg", None, None, artifacts)
        for name in ("fednova", "proposed"):
            with self.assertRaises(NotImplementedError):
                get_strategy(name)
        with self.assertRaises(ValueError):
            get_strategy("unknown")

    def test_provenance_covers_all_live_packages(self):
        hashes = collect_source_hashes()
        for path in ("models/cnn.py", "algorithms/fedprox.py", "server/server.py",
                     "client/train.py", "datasets/partition.py", "monitoring/network.py",
                     "experiments/artifacts.py", "scripts/distribute_data.py"):
            self.assertIn(path, hashes)
        self.assertFalse(any("source_snapshot" in path for path in hashes))
        self.assertEqual(count_parameters(CNN()), 20104)

    def test_hardware_presets_resolve_and_explicit_values_win(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jetson.yaml").write_text(yaml.safe_dump({
                "profiles": {"nano": {"batch_size": 16, "workers": 0}},
                "client_registry": []}), encoding="utf-8")
            config = root / "experiment.yaml"
            config.write_text("hardware:\n  profile: nano\n", encoding="utf-8")
            self.assertEqual(load_config(config)["training"]["batch_size"], 16)
            config.write_text("hardware:\n  profile: nano\ntraining:\n  batch_size: 8\n", encoding="utf-8")
            self.assertEqual(load_config(config)["training"]["batch_size"], 8)
            config.write_text("hardware:\n  profile: missing\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(config)

    def test_tcp_probe_and_logger_use_loopback_and_preserve_failed_null(self):
        with socket.socket() as listener, tempfile.TemporaryDirectory() as directory:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            result = ping("127.0.0.1", port)
            self.assertTrue(result["reachable"])
            self.assertGreaterEqual(result["latency_ms"], 0)
            path = Path(directory) / "ping.csv"
            ClientPingLogger(path).log_round(3, [{"id": 0, "host": "127.0.0.1", "port": port}])
            self.assertIn("latency_ms", path.read_text(encoding="utf-8"))
        with patch("monitoring.network.socket.create_connection", side_effect=OSError("unreachable")):
            failed = ping("127.0.0.1", port)
        self.assertFalse(failed["reachable"])
        self.assertIsNone(failed["latency_ms"])

    def test_scp_commands_validate_destinations_without_transferring(self):
        with tempfile.TemporaryDirectory() as directory:
            client = {"host": "jetson.local", "ssh_user": "student", "data_dir": "/home/student/data"}
            command = transfer_command(directory, client)
            self.assertEqual(command[-1], "student@jetson.local:/home/student/data")
            for bad in ("/data;touch_bad", "/data/../etc", "relative"):
                with self.assertRaises(ValueError):
                    transfer_command(directory, {**client, "data_dir": bad})

    def test_metric_exports_and_partition_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = FLMetricsRecorder()
            recorder.record(1, accuracy=0.5, power_watts=None)
            recorder.record(2, accuracy=0.75)
            recorder.export(root)
            self.assertIsNone(json.loads((root / "metrics.json").read_text())[0]["power_watts"])
            recorder.plot("accuracy", root / "accuracy.png")
            metadata = {"clients": {0: {"class_distribution": {"0": 2, "1": 3}},
                                    1: {"class_distribution": {"0": 4, "1": 1}}}}
            visualize_partition(metadata, root / "partition.png")
            self.assertGreater((root / "partition.png").stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()

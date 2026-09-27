"""Offline codec, real serialized payload and aggregation regression checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import yaml
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedmedai.client import FedMedClient
from fedmedai.communication import decode_uplink, encode_uplink
from fedmedai.experiment import RunArtifacts, load_config
from fedmedai.model import get_weights
from fedmedai.server import FedMedFedAvg, model_payload_bytes
from fedmedai.simulation import run_sequential


class UplinkCodecTests(unittest.TestCase):
    def test_roundtrip_error_bounds_and_integer_buffers_are_exact(self):
        weights = [np.linspace(-3, 3, 4096, dtype=np.float32).reshape(64, 64),
                   np.array(0.75, dtype=np.float32), np.zeros(7, dtype=np.float32),
                   np.array(2 ** 60 + 1, dtype=np.int64), np.array([True, False]),
                   np.array([], dtype=np.float32)]
        for codec in ("none", "fp16", "int8"):
            with self.subTest(codec=codec):
                encoded = encode_uplink(weights, codec)
                wire = ndarrays_to_parameters(encoded)
                restored = decode_uplink(parameters_to_ndarrays(wire), weights, codec)
                for original, actual in zip(weights, restored):
                    self.assertEqual(original.shape, actual.shape)
                    self.assertEqual(original.dtype, actual.dtype)
                    if original.dtype.kind != "f" or codec == "none":
                        np.testing.assert_array_equal(actual, original)
                    elif codec == "fp16":
                        np.testing.assert_allclose(actual, original, rtol=5e-4, atol=1e-7)
                    else:
                        bound = float(np.max(np.abs(original))) / 254 if original.size else 0
                        np.testing.assert_allclose(actual, original, rtol=0, atol=bound + 1e-6)
                np.testing.assert_array_equal(restored[2], weights[2])
                if codec != "none":
                    self.assertLess(model_payload_bytes(wire),
                                    model_payload_bytes(ndarrays_to_parameters(weights)))
                if codec == "int8":
                    self.assertEqual(len(encoded), len(weights) + 1)
                    self.assertEqual(encoded[-1].shape, (4,))

    def test_none_is_bit_exact_and_codecs_do_not_alias(self):
        for codec in ("none", "fp16", "int8"):
            original = [np.array([-0.0, 0.1, 1.0], dtype=np.float32)]
            before = original[0].tobytes()
            encoded = encode_uplink(original, codec)
            restored = decode_uplink(encoded, original, codec)
            if codec == "none":
                self.assertEqual(restored[0].tobytes(), before)
            encoded[0][0] = 1
            restored[0][0] = 2
            self.assertEqual(original[0].tobytes(), before)

    def test_reject_invalid_input_overflow_and_wire_schema(self):
        reference = [np.array([1.0, -2.0], dtype=np.float32), np.array(3, dtype=np.int64)]
        with self.assertRaises(ValueError):
            encode_uplink(reference, "unknown")
        for value in (np.nan, np.inf, -np.inf):
            for codec in ("none", "fp16", "int8"):
                with self.assertRaises(ValueError):
                    encode_uplink([np.array([value], dtype=np.float32)], codec)
        with self.assertRaises(ValueError):
            encode_uplink([np.array([70000.0], dtype=np.float32)], "fp16")
        with self.assertRaises(ValueError):
            encode_uplink([np.array([1 + 1j])], "none")
        for codec in ("none", "fp16", "int8"):
            wire = encode_uplink(reference, codec)
            with self.assertRaises(ValueError):
                decode_uplink(wire[:-1], reference, codec)
            for bad in (np.zeros((1, 2), dtype=wire[0].dtype),
                        wire[0].astype(np.float64)):
                with self.assertRaises(ValueError):
                    decode_uplink([bad, *wire[1:]], reference, codec)
            with self.assertRaises(ValueError):
                decode_uplink([wire[0], np.array(3, dtype=np.int32), *wire[2:]], reference, codec)

    def test_reject_invalid_scales_range_and_dequantization_overflow(self):
        reference = [np.array([1.0], dtype=np.float32)]
        wire = encode_uplink(reference, "int8")
        for scale in (0, -1, np.nan, np.inf, 1e300):
            with self.assertRaises(ValueError):
                decode_uplink([wire[0], np.array([scale], dtype=np.float64)], reference, "int8")
        for scales in (np.array(1.0), np.array([1.0, 1.0]), np.array([1.0], dtype=np.float32)):
            with self.assertRaises(ValueError):
                decode_uplink([wire[0], scales], reference, "int8")
        with self.assertRaises(ValueError):
            decode_uplink([np.array([-128], dtype=np.int8), wire[-1]], reference, "int8")


class UplinkStrategyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fedmedai_codec_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "config.yaml"
        self.path.write_text(yaml.safe_dump({
            "federation": {"num_clients": 2, "num_rounds": 1},
            "runtime": {"backend": "sequential"},
            "monitoring": {"enabled": False},
            "paths": {name + "_dir": str(self.root / name)
                      for name in ("results", "checkpoints", "runs")}}), encoding="utf-8")

    def strategy(self, codec):
        config = load_config(self.path, {"communication": {"uplink_codec": codec}})
        model = nn.Linear(2, 8)
        loader = DataLoader(TensorDataset(torch.arange(16).reshape(8, 2).float() / 16,
                                         torch.arange(8)), batch_size=4)
        artifacts = RunArtifacts(config, "codec_test")
        strategy = FedMedFedAvg(model, loader, artifacts, Mock(), torch.device("cpu"))
        return strategy, loader, artifacts

    def test_config_defaults_and_invalid_codec(self):
        self.assertEqual(load_config(self.path)["communication"]["uplink_codec"], "none")
        with self.assertRaises(ValueError):
            load_config(self.path, {"communication": {"uplink_codec": "zip"}})

    def test_strategy_dequantizes_before_weighted_average_and_counts_received_bytes(self):
        for codec in ("none", "fp16", "int8"):
            with self.subTest(codec=codec):
                strategy, _, _ = self.strategy(codec)
                template = get_weights(strategy.global_model)
                states = [[np.full_like(t, value) for t in template] for value in (0.1234, -0.5678)]
                results = []
                expected = []
                for cid, (state, count) in enumerate(zip(states, (1, 3))):
                    encoded = encode_uplink(state, codec)
                    expected.append(decode_uplink(encoded, template, codec))
                    record = {"client_id": cid, "round": 1, "phase": "fit", "num_samples": count,
                              "uplink_codec": codec, "train_loss": 0.5, "train_accuracy": 0.25,
                              "resources": {"duration_seconds": 0.1, "energy_joules_observed": None}}
                    results.append((SimpleNamespace(cid=str(cid)), FitRes(
                        Status(Code.OK, ""), ndarrays_to_parameters(encoded), count,
                        {"client_id": cid, "uplink_codec": codec, "telemetry_json": json.dumps(record)})))
                before = [list(result.parameters.tensors) for _, result in results]
                byte_count = sum(model_payload_bytes(result.parameters) for _, result in results)
                strategy.round_started = 0.0
                strategy.round_history[1] = {"fit_download_model_bytes": 100, "evaluate_download_model_bytes": 100}
                aggregated, _ = strategy.aggregate_fit(1, list(reversed(results)), [])
                for actual, left, right in zip(parameters_to_ndarrays(aggregated), *expected):
                    np.testing.assert_allclose(actual, (left + 3 * right) / 4, atol=1e-7)
                self.assertEqual(strategy.round_history[1]["fit_upload_model_bytes"], byte_count)
                self.assertEqual(strategy.summary()["total_fit_upload_model_bytes"], byte_count)
                self.assertEqual(before, [result.parameters.tensors for _, result in results])
                results[0][1].metrics["uplink_codec"] = "invalid"
                with self.assertRaises(ValueError):
                    strategy.aggregate_fit(1, results, [])

    def test_real_clients_complete_round_with_codec_and_unchanged_downlinks(self):
        torch.set_num_threads(1)
        download_totals = []
        for codec in ("none", "fp16", "int8"):
            with self.subTest(codec=codec):
                strategy, loader, artifacts = self.strategy(codec)
                clients = {cid: FedMedClient(cid, nn.Linear(2, 8), loader, loader,
                                             device=torch.device("cpu"), monitoring={"enabled": False})
                           for cid in range(2)}
                self.assertEqual(strategy.fit_config(1)["uplink_codec"], codec)
                run_sequential(strategy, clients, 1)
                summary = strategy.summary()
                self.assertEqual(summary["completed_rounds"], 1)
                self.assertEqual(summary["uplink_codec"], codec)
                self.assertEqual(summary["total_model_payload_bytes"], sum(summary[key] for key in (
                    "total_fit_upload_model_bytes", "total_fit_download_model_bytes",
                    "total_evaluate_download_model_bytes")))
                download_totals.append(summary["total_fit_download_model_bytes"])
                manifest = json.loads((artifacts.results / "run_manifest.json").read_text())
                self.assertEqual(manifest["uplink_codec"], codec)
                records = [json.loads(line) for line in
                           (artifacts.results / "client_metrics.jsonl").read_text().splitlines()]
                self.assertEqual([r["uplink_codec"] for r in records if r["phase"] == "fit"], [codec] * 2)
                self.assertTrue((artifacts.checkpoints / "best_model.pt").exists())
        self.assertEqual(len(set(download_totals)), 1)


if __name__ == "__main__":
    unittest.main()

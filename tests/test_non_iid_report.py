"""Reports must reject contradictory provenance and distinguish codec effects."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "non_iid_report", Path(__file__).resolve().parents[1] / "scripts/report_non_iid_study.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class NonIIDReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        names = ("baseline.lock.json", "protocol.md", "runner_snapshot.py")
        for name in names:
            (self.root / name).write_bytes(b"snapshot")
        self.write("snapshot_hashes.json", {name: hashlib.sha256(b"snapshot").hexdigest() for name in names})
        self.execution = {"status": "completed", "exit_code": 0, "supervisor_exit_code": 0}
        self.keys = {"final_test_accuracy", "final_test_f1_macro", "final_test_loss", "test_evaluation",
                     "best_round", "total_model_payload_bytes", "partition_sha256", "checkpoint_tensors"}
        self.keys.update(f"round_{r}_metrics" for r in range(1, 9))
        self.replay = {"exact_match": True, "checks": dict.fromkeys(self.keys, True)}
        self.lock = {"baseline_config": {"federation": {"num_rounds": 8}}}

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def rejected(self):
        self.write("execution.json", self.execution)
        self.write("baseline_replay.json", self.replay)
        with self.assertRaises(ValueError):
            report.validate_evidence(self.root, {"status": "completed"}, [], self.lock)

    def test_timeout_supervisor_cannot_claim_success(self):
        self.execution["supervisor_exit_code"] = 124
        self.rejected()

    def test_missing_round_or_false_check_rejected(self):
        self.replay["checks"].pop("round_8_metrics")
        self.rejected()
        self.replay["checks"]["round_8_metrics"] = False
        self.rejected()

    def test_changed_snapshot_rejected(self):
        (self.root / "protocol.md").write_text("changed")
        self.rejected()

    def test_replay_cannot_be_original_baseline_or_unrelated_reference(self):
        baseline = {"arm": "fedavg", "result": {"alpha": 0.3, "results_dir": str(self.root / "baseline")}}
        path = baseline["result"]["results_dir"]
        self.write("execution.json", self.execution)
        for reference in (path, str(self.root / "unrelated")):
            self.replay.update(reference=reference, new_baseline=path)
            self.write("baseline_replay.json", self.replay)
            with self.assertRaises(ValueError):
                report.validate_evidence(self.root, {"status": "completed", "runs": [path], "replay": path},
                                         [baseline], self.lock)

    def test_confusion_matrix_recomputation_detects_mislabeled_score(self):
        cm = [[0] * 8 for _ in range(8)]
        cm[0][0], cm[0][1] = 3, 1
        cell = {"result": {"test_evaluation": {"confusion_matrix": cm, "accuracy": 0.75,
                                               "f1_macro": (6 / 7) / 8},
                           "final_test_accuracy": 0.75, "final_test_f1_macro": (6 / 7) / 8}}
        report.check_metrics(cell)
        cell["result"]["final_test_accuracy"] = 0.8
        with self.assertRaises(ValueError):
            report.check_metrics(cell)

    def test_codec_effect_is_relative_to_fedprox(self):
        cells = []
        for alpha in report.ALPHAS:
            for arm, accuracy, payload in (("fedavg", .5, 300), ("fedprox", .8, 300), ("fedprox_fp16", .79, 250)):
                cells.append({"arm": arm, "result": {"alpha": alpha, "final_test_accuracy": accuracy,
                              "final_test_f1_macro": accuracy, "total_model_payload_bytes": payload}})
        for row in report.direct_codec_comparisons(cells):
            self.assertAlmostEqual(row["accuracy_delta_pp"], -1)
            self.assertAlmostEqual(row["total_saving_percent"], 100 / 6)


if __name__ == "__main__":
    unittest.main()

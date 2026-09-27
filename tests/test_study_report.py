"""The report must reject stale or contradictory verification evidence."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts/report_communication_study.py"
spec = importlib.util.spec_from_file_location("study_report", path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        protocol = b"fixed protocol"
        (self.root / "protocol.md").write_bytes(protocol)
        self.write("protocol_sha256.json", {"sha256": hashlib.sha256(protocol).hexdigest()})
        self.execution = {"status": "completed", "exit_code": 0}
        self.replay = {"exact_match": True, "original": "original", "replay": "repeat",
                      "checks": {key: True for key in (
                          "final_test_accuracy", "final_test_f1_macro", "final_test_loss", "best_round",
                          "total_model_payload_bytes", "test_evaluation", "checkpoint_tensors")}}
        self.index = {"runs": ["original"], "replay": "repeat"}

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def check_rejected(self):
        self.write("execution.json", self.execution)
        self.write("reproducibility.json", self.replay)
        with self.assertRaises(ValueError):
            report.validate_evidence(self.root, self.index, [])

    def test_false_individual_check_cannot_claim_exact_match(self):
        self.replay["checks"]["checkpoint_tensors"] = False
        self.check_rejected()

    def test_nonzero_exit_code_rejected(self):
        self.execution["exit_code"] = 1
        self.check_rejected()

    def test_unrelated_replay_rejected(self):
        self.replay["replay"] = "different_suite"
        self.check_rejected()

    def test_changed_protocol_rejected(self):
        (self.root / "protocol.md").write_text("changed design")
        self.check_rejected()


if __name__ == "__main__":
    unittest.main()

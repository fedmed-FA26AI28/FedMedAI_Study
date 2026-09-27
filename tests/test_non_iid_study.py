"""Reject baseline drift, incorrect treatments, IID and unpaired comparisons."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedmedai.non_iid_study import ALPHAS, ARMS, compare_cells, make_config, read_lock, validate_config, verify_source

LOCK = read_lock(Path(__file__).resolve().parents[1] / "research/non_iid_baseline.lock.json")


def fixture():
    cells = []
    for alpha in ALPHAS:
        for arm in ARMS:
            cells.append({"manifest": {"config": make_config(LOCK, alpha, arm), "source_sha256": {"same": "source"}},
                          "test_ids": [0, 1], "result": {"alpha": alpha, "partition_sha256": str(alpha),
                          "final_test_accuracy": 0.8, "final_test_f1_macro": 0.7,
                          "total_fit_upload_model_bytes": 50 if arm.endswith("fp16") else 100,
                          "total_fit_download_model_bytes": 100,
                          "total_evaluate_download_model_bytes": 100,
                          "total_model_payload_bytes": 250 if arm.endswith("fp16") else 300}})
    return cells


class NonIIDStudyTests(unittest.TestCase):
    def test_project_configs_resolve_to_the_frozen_current_baseline(self):
        from fedmedai.experiment import load_config
        root = Path(__file__).resolve().parents[1]
        for name in ("config.yaml", "communication_study.yaml", "non_iid_study.yaml"):
            with self.subTest(config=name):
                self.assertEqual(load_config(root / "configs" / name), LOCK["baseline_config"])

    def test_baseline_code_and_config_are_frozen(self):
        verify_source(LOCK)
        baseline = make_config(LOCK, 0.3, "fedavg")
        expected = copy.deepcopy(LOCK["baseline_config"])
        baseline.pop("paths")
        expected.pop("paths")
        self.assertEqual(baseline, expected)

    def test_no_iid_or_other_seed_or_baseline_compression(self):
        for mutation in (lambda c: c["federation"].update(partition_type="iid"),
                         lambda c: c["federation"].update(seed=1),
                         lambda c: c["communication"].update(uplink_codec="fp16"),
                         lambda c: c["training"].update(learning_rate=0.01)):
            config = make_config(LOCK, 0.1, "fedavg")
            mutation(config)
            with self.assertRaises(ValueError):
                validate_config(config, LOCK)

    def test_real_fedprox_requires_fixed_nonzero_mu(self):
        for arm in ("fedprox", "fedprox_fp16"):
            config = make_config(LOCK, 1.0, arm)
            self.assertEqual(validate_config(config, LOCK), arm)
            config["algorithm"]["proximal_mu"] = 0
            with self.assertRaises(ValueError):
                validate_config(config, LOCK)

    def test_requires_exact_nine_matched_cells(self):
        cells = fixture()
        result = compare_cells(cells, LOCK)
        self.assertFalse(result["fedprox_consistent_non_iid_gain"])
        self.assertTrue(result["fedprox_fp16_communication_pass_all_alphas"])
        for bad in (cells[:-1], cells + [cells[0]]):
            with self.assertRaises(ValueError):
                compare_cells(bad, LOCK)
        cells[1]["result"]["partition_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            compare_cells(cells, LOCK)

    def test_one_failed_alpha_cannot_be_hidden_by_average(self):
        cells = fixture()
        cells[2]["result"]["final_test_accuracy"] = 0.78
        result = compare_cells(cells, LOCK)
        self.assertFalse(result["fedprox_fp16_communication_pass_all_alphas"])

    def test_downlink_change_is_not_attributed_to_upload_codec(self):
        cells = fixture()
        cells[2]["result"]["total_fit_download_model_bytes"] = 50
        with self.assertRaises(ValueError):
            compare_cells(cells, LOCK)


if __name__ == "__main__":
    unittest.main()

"""Offline checks that study conclusions require valid, fully paired evidence."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedmedai.study import analyze_runs, assert_matched, load_run, validate_study_config


def fixture():
    runs = []
    for partition in ("iid", "dirichlet"):
        for seed in (0, 1, 2):
            for codec, upload in (("none", 100), ("fp16", 50), ("int8", 25)):
                runs.append({"result": {"partition_type": partition, "random_seed": seed,
                    "uplink_codec": codec, "partition_sha256": f"{partition}_{seed}",
                    "final_test_accuracy": 0.8, "final_test_f1_macro": 0.75,
                    "total_fit_upload_model_bytes": upload, "total_model_payload_bytes": 200 + upload,
                    "total_fit_download_model_bytes": 100, "total_evaluate_download_model_bytes": 100},
                    "test_ids": [0, 1, 2], "manifest": {"source_sha256": {"model": "abc"},
                    "config": {"seed": seed, "partition": partition, "lr": 0.001,
                               "communication": {"uplink_codec": codec}, "paths": {}}}})
    return runs


class StudyTests(unittest.TestCase):
    def test_primary_arm_cannot_be_omitted(self):
        runs = [run for run in fixture() if run["result"]["uplink_codec"] != "fp16"]
        with self.assertRaises(ValueError):
            analyze_runs(runs, codecs=("none", "int8"))

    def test_upload_savings_are_distinct_from_total_savings(self):
        analysis = analyze_runs(fixture())
        fp16 = analysis["comparisons"][0]
        self.assertEqual(fp16["upload_saving_percent"]["mean"], 50)
        self.assertAlmostEqual(fp16["total_saving_percent"]["mean"], 100 / 6)
        self.assertTrue(analysis["primary_fp16_pass"])

    def test_incomplete_duplicate_and_extra_cells_are_rejected(self):
        runs = fixture()
        for bad in (runs[:-1], runs + [runs[0]]):
            with self.assertRaises(ValueError):
                analyze_runs(bad)

    def test_partition_test_ids_source_and_hyperparameter_confound_rejected(self):
        original = fixture()
        for mutation in (lambda r: r["result"].update(partition_sha256="different"),
                         lambda r: r.update(test_ids=[0, 1, 3]),
                         lambda r: r["manifest"].update(source_sha256={"model": "new"}),
                         lambda r: r["manifest"]["config"].update(lr=0.1),
                         lambda r: r["result"].update(total_fit_download_model_bytes=200)):
            runs = copy.deepcopy(original)
            mutation(runs[1])
            with self.assertRaises(ValueError):
                assert_matched(runs[0], runs[1])

    def test_accuracy_regression_fails_primary_even_when_int8_passes(self):
        runs = fixture()
        for run in runs:
            if run["result"]["uplink_codec"] == "fp16":
                run["result"]["final_test_accuracy"] = 0.78
        self.assertFalse(analyze_runs(runs)["primary_fp16_pass"])

    def test_single_seed_regression_is_preserved(self):
        runs = fixture()
        runs[1]["result"]["final_test_accuracy"] = 0.78
        comparison = analyze_runs(runs)["comparisons"][0]
        self.assertAlmostEqual(comparison["pairs"][0]["accuracy_delta_pp"], -2)
        self.assertGreater(comparison["accuracy_delta_pp"]["sample_sd"], 0)


class StudyArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Synthetic legacy-schema fixture, independent of the current research YAML.
        # This tests backward-compatible artifact validation, not old research outcomes.
        self.config = {
            "federation": {"num_clients": 3, "num_rounds": 5, "dirichlet_alpha": 0.3,
                           "partition_type": "iid", "seed": 0},
            "algorithm": {"name": "fedavg"}, "communication": {"uplink_codec": "none"},
            "model": {"name": "bloodcnn", "pretrained": False},
            "training": {"batch_size": 64, "local_epochs": 1, "learning_rate": 0.001,
                         "weight_decay": 0.01},
            "dataset": {"name": "BloodMNIST", "image_size": 28, "download": False,
                        "max_train_samples": None, "max_val_samples": None, "max_test_samples": None},
            "runtime": {"backend": "sequential", "deterministic": True, "torch_num_threads": 1},
            "monitoring": {"enabled": False},
            "evaluation": {"selection_metric": "f1_macro", "target_accuracy": 0.8,
                           "max_failure_cases": 4000},
        }
        self.files = {
            "run_manifest.json": {"run_id": "test", "status": "completed", "config": self.config},
            "final_test_metrics.json": {"run_id": "test", "completed_rounds": 5,
                "random_seed": 0, "partition_type": "iid", "uplink_codec": "none",
                "algorithm": "fedavg", "model": "bloodcnn", "dataset": "BloodMNIST",
                "selection_split": "val", "test_checkpoint_round": 1, "best_round": 1,
                "partition_sha256": "paired", "train_samples": 11959, "test_samples": 3421,
                "total_fit_upload_model_bytes": 100, "total_fit_download_model_bytes": 100,
                "total_evaluate_download_model_bytes": 100, "total_model_payload_bytes": 300,
                "test_evaluation": {"confusion_matrix": [[3421]]}},
            "partition_metadata.json": {"partition_sha256": "paired", "source_split_sizes": {"train": 11959, "val": 1712}},
            "evaluation/test_sample_ids.json": {"split": "test", "selection_policy": "official_full_split", "sample_ids": list(range(3421))},
        }
        for number in range(1, 6):
            self.files[f"rounds/round_{number:04d}.json"] = {
                "round_model_payload_bytes": 60, "fit_upload_model_bytes": 20,
                "fit_download_model_bytes": 20, "evaluate_download_model_bytes": 20,
                "fit_failures": 0, "evaluate_failures": 0, "val_samples": 1712,
                "fit_clients": 3, "evaluate_clients": 3}

    def write_files(self):
        for name, value in self.files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))

    def test_valid_full_split_artifacts_load(self):
        self.write_files()
        self.assertEqual(load_run(self.root)["result"]["test_samples"], 3421)

    def test_mislabeled_results_are_rejected(self):
        result = self.files["final_test_metrics.json"]
        for field, wrong in (("random_seed", 1), ("partition_type", "dirichlet"),
                             ("uplink_codec", "fp16"), ("run_id", "other")):
            saved = result[field]
            result[field] = wrong
            self.write_files()
            with self.assertRaises(ValueError):
                load_run(self.root)
            result[field] = saved

    def test_duplicate_test_ids_and_incomplete_split_rejected(self):
        self.files["evaluation/test_sample_ids.json"]["sample_ids"][-1] = 0
        self.write_files()
        with self.assertRaises(ValueError):
            load_run(self.root)
        self.files["evaluation/test_sample_ids.json"]["sample_ids"] = list(range(3420))
        self.files["final_test_metrics.json"]["test_samples"] = 3420
        self.files["final_test_metrics.json"]["test_evaluation"]["confusion_matrix"] = [[3420]]
        self.write_files()
        with self.assertRaises(ValueError):
            load_run(self.root)

    def test_reallocated_bytes_cannot_hide_behind_matching_total(self):
        result = self.files["final_test_metrics.json"]
        result["total_fit_upload_model_bytes"] -= 10
        result["total_fit_download_model_bytes"] += 10
        self.write_files()
        with self.assertRaises(ValueError):
            load_run(self.root)

    def test_fixed_protocol_rejects_design_changes(self):
        validate_study_config(self.config)
        for section, key, wrong in (("model", "name", "resnet18"), ("algorithm", "name", "fedprox"),
                                     ("federation", "num_rounds", 4), ("federation", "dirichlet_alpha", 0.5)):
            altered = copy.deepcopy(self.config)
            altered[section][key] = wrong
            with self.assertRaises(ValueError):
                validate_study_config(altered)


if __name__ == "__main__":
    unittest.main()

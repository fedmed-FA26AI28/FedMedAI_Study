"""Paired communication-study analysis with explicit integrity checks."""

import json
from pathlib import Path
from statistics import mean, stdev


def validate_study_config(config):
    """The published protocol is fixed; a different design needs a new protocol."""
    required = {
        "federation": {"num_clients": 3, "num_rounds": 5, "dirichlet_alpha": 0.3},
        "algorithm": {"name": "fedavg"},
        "model": {"name": "bloodcnn", "pretrained": False},
        "training": {"batch_size": 64, "local_epochs": 1, "learning_rate": 0.001, "weight_decay": 0.01},
        "dataset": {"name": "BloodMNIST", "download": False, "max_train_samples": None,
                    "max_val_samples": None, "max_test_samples": None},
        "runtime": {"backend": "sequential", "deterministic": True, "torch_num_threads": 1},
        "monitoring": {"enabled": False},
        "evaluation": {"selection_metric": "f1_macro", "target_accuracy": 0.8, "max_failure_cases": 4000},
    }
    for section, fields in required.items():
        for key, expected in fields.items():
            if key not in config.get(section, {}) or config[section][key] != expected:
                raise ValueError(f"Configuration differs from fixed protocol: {section}.{key}")
    if config["dataset"].get("image_size", 28) != 28:
        raise ValueError("Fixed study requires 28x28 images")
    fed = config["federation"]
    if fed["partition_type"] not in ("iid", "dirichlet") or fed["seed"] not in (0, 1, 2):
        raise ValueError("Unexpected partition or seed")
    if config["communication"]["uplink_codec"] not in ("none", "fp16", "int8"):
        raise ValueError("Unexpected study codec")


def descriptive(values):
    return {"n": len(values), "mean": mean(values),
            "sample_sd": stdev(values) if len(values) > 1 else None}


def load_run(result_dir, config_validator=validate_study_config):
    root = Path(result_dir)
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    result, manifest = read("final_test_metrics.json"), read("run_manifest.json")
    config = manifest["config"]
    config_validator(config)
    bindings = {"random_seed": config["federation"]["seed"],
                "partition_type": config["federation"]["partition_type"],
                "uplink_codec": config["communication"]["uplink_codec"],
                "algorithm": config["algorithm"]["name"], "model": config["model"]["name"],
                "dataset": config["dataset"]["name"], "run_id": manifest["run_id"]}
    if any(result[key] != value for key, value in bindings.items()):
        raise ValueError("Result labels disagree with manifest configuration")
    if manifest["status"] != "completed" or result["completed_rounds"] != config["federation"]["num_rounds"]:
        raise ValueError("Incomplete study run")
    if result["selection_split"] != "val" or result["test_checkpoint_round"] != result["best_round"]:
        raise ValueError("Invalid checkpoint selection")
    parts = read("partition_metadata.json")
    if parts["source_split_sizes"] != {"train": 11959, "val": 1712} or result["test_samples"] != 3421:
        raise ValueError("Unexpected official BloodMNIST split sizes")
    if result["partition_sha256"] != parts["partition_sha256"]:
        raise ValueError("Partition provenance mismatch")
    if any(config["dataset"][key] is not None for key in (
            "max_train_samples", "max_val_samples", "max_test_samples")):
        raise ValueError("Study requires uncapped official splits")
    if result["train_samples"] != parts["source_split_sizes"]["train"]:
        raise ValueError("Incomplete training split")
    components = ("total_fit_upload_model_bytes", "total_fit_download_model_bytes",
                  "total_evaluate_download_model_bytes")
    if result["total_model_payload_bytes"] != sum(result[key] for key in components):
        raise ValueError("Payload accounting does not reconcile")
    round_rows = [read(f"rounds/round_{r:04d}.json")
                  for r in range(1, result["completed_rounds"] + 1)]
    if sum(row["round_model_payload_bytes"] for row in round_rows) != result["total_model_payload_bytes"]:
        raise ValueError("Round accounting does not reconcile")
    for key in components:
        if result[key] != sum(row[key.removeprefix("total_")] for row in round_rows):
            raise ValueError("Payload components disagree with round records")
    for row in round_rows:
        if row["fit_clients"] != 3 or row["evaluate_clients"] != 3:
            raise ValueError("Incomplete client participation")
        if row["fit_failures"] or row["evaluate_failures"]:
            raise ValueError("Client failure in study run")
        if row["val_samples"] != parts["source_split_sizes"]["val"]:
            raise ValueError("Incomplete validation split")
    ids = read("evaluation/test_sample_ids.json")
    if ids["split"] != "test" or sorted(ids["sample_ids"]) != list(range(3421)):
        raise ValueError("Test IDs are not the complete official split")
    if ids["selection_policy"] != "official_full_split" or len(ids["sample_ids"]) != result["test_samples"]:
        raise ValueError("Invalid test split provenance")
    if sum(map(sum, result["test_evaluation"]["confusion_matrix"])) != result["test_samples"]:
        raise ValueError("Test confusion matrix count mismatch")
    return {"result": result, "manifest": manifest, "test_ids": ids["sample_ids"]}


def assert_matched(baseline, candidate):
    left, right = baseline["result"], candidate["result"]
    if left["partition_sha256"] != right["partition_sha256"] or baseline["test_ids"] != candidate["test_ids"]:
        raise ValueError("Unpaired partitions or test samples")
    if baseline["manifest"]["source_sha256"] != candidate["manifest"]["source_sha256"]:
        raise ValueError("Source changed during paired study")
    configs = []
    for run in (baseline, candidate):
        config = json.loads(json.dumps(run["manifest"]["config"]))
        config.pop("communication")
        config.pop("paths")
        configs.append(config)
    if configs[0] != configs[1]:
        raise ValueError("Confounded comparison: settings differ beyond codec")
    for key in ("total_fit_download_model_bytes", "total_evaluate_download_model_bytes"):
        if left[key] != right[key]:
            raise ValueError("Unexpected downlink change")


def analyze_runs(runs, expected_seeds=(0, 1, 2), codecs=("none", "fp16", "int8")):
    """Reject incomplete/duplicate pairs; never infer noninferiority from n=3."""
    if not {"none", "fp16"}.issubset(codecs) or not expected_seeds:
        raise ValueError("Study requires baseline, primary FP16 arm, and nonempty seeds")
    indexed = {}
    for run in runs:
        result = run["result"]
        key = (result["partition_type"], result["random_seed"], result["uplink_codec"])
        if key in indexed:
            raise ValueError("Duplicate study cell")
        indexed[key] = run
    expected = {(partition, seed, codec) for partition in ("iid", "dirichlet")
                for seed in expected_seeds for codec in codecs}
    if set(indexed) != expected:
        raise ValueError("Incomplete or unexpected study cells")
    comparisons = []
    for partition in ("iid", "dirichlet"):
        for codec in codecs:
            if codec == "none":
                continue
            pairs = []
            for seed in expected_seeds:
                base, candidate = indexed[partition, seed, "none"], indexed[partition, seed, codec]
                assert_matched(base, candidate)
                a, b = base["result"], candidate["result"]
                pairs.append({"seed": seed,
                              "accuracy_delta_pp": 100 * (b["final_test_accuracy"] - a["final_test_accuracy"]),
                              "f1_delta_pp": 100 * (b["final_test_f1_macro"] - a["final_test_f1_macro"]),
                              "upload_saving_percent": 100 * (1 - b["total_fit_upload_model_bytes"] / a["total_fit_upload_model_bytes"]),
                              "total_saving_percent": 100 * (1 - b["total_model_payload_bytes"] / a["total_model_payload_bytes"])})
            stats = {key: descriptive([pair[key] for pair in pairs]) for key in pairs[0] if key != "seed"}
            comparisons.append({"partition": partition, "codec": codec, "pairs": pairs, **stats,
                                "descriptive_threshold_pass": stats["accuracy_delta_pp"]["mean"] >= -1
                                and stats["f1_delta_pp"]["mean"] >= -1
                                and stats["total_saving_percent"]["mean"] >= 15})
    return {"comparisons": comparisons,
            "primary_fp16_pass": all(row["descriptive_threshold_pass"] for row in comparisons if row["codec"] == "fp16"),
            "inference": "Descriptive repeated-seed study; no significance, equivalence or clinical claim."}

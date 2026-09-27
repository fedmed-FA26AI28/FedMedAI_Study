"""Frozen FedAvg control versus actual FedProx under one-seed Dirichlet skew."""

import copy
import hashlib
import json
from pathlib import Path

from fedmedai.study import load_run

ARMS = {"fedavg": ("fedavg", "none"), "fedprox": ("fedprox", "none"),
        "fedprox_fp16": ("fedprox", "fp16")}
ALPHAS = (0.1, 0.3, 1.0)


def read_lock(path):
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if lock["alphas"] != list(ALPHAS) or lock["seed"] != 2026 or lock["proximal_mu"] != 0.01:
        raise ValueError("Unexpected fixed design in baseline lock")
    return lock


def verify_source(lock):
    root = Path(__file__).parent
    for name, expected in lock["frozen_source_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen baseline source changed: {name}")


def make_config(lock, alpha, arm):
    if alpha not in ALPHAS or arm not in ARMS:
        raise ValueError("Only the declared Dirichlet conditions and arms are allowed")
    config = copy.deepcopy(lock["baseline_config"])
    config["federation"].update(partition_type="dirichlet", dirichlet_alpha=alpha, seed=lock["seed"])
    config["algorithm"].update(name=ARMS[arm][0], proximal_mu=lock["proximal_mu"])
    config["communication"]["uplink_codec"] = ARMS[arm][1]
    config["paths"] = {key: f"{name}/non_iid_study" for key, name in (
        ("results_dir", "results"), ("checkpoints_dir", "checkpoints"), ("runs_dir", "runs"))}
    return config


def validate_config(config, lock):
    fed = config["federation"]
    pair = (config["algorithm"]["name"], config["communication"]["uplink_codec"])
    arm = next((key for key, value in ARMS.items() if value == pair), None)
    if arm is None or fed["partition_type"] != "dirichlet" or fed["seed"] != lock["seed"]:
        raise ValueError("Unexpected algorithm, codec, partition or seed")
    expected = make_config(lock, fed["dirichlet_alpha"], arm)
    actual = copy.deepcopy(config)
    expected.pop("paths")
    actual.pop("paths")
    if expected != actual:
        raise ValueError("Configuration changed beyond the declared treatment/alpha")
    return arm


def load_cell(path, lock):
    cell = load_run(path, config_validator=lambda config: validate_config(config, lock))
    config, result = cell["manifest"]["config"], cell["result"]
    arm = validate_config(config, lock)
    if result["alpha"] != config["federation"]["dirichlet_alpha"]:
        raise ValueError("Mislabeled alpha")
    for name, value in lock["frozen_source_sha256"].items():
        if cell["manifest"]["source_sha256"].get(name) != value:
            raise ValueError("Executed source changed the frozen baseline")
    root = Path(path)
    metadata = json.loads((root / "partition_metadata.json").read_text())
    recorded_hash = metadata.pop("partition_sha256")
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != recorded_hash:
        raise ValueError("Partition hash does not match metadata")
    for split, count in (("train", 11959), ("val", 1712)):
        ids = [index for client in metadata["clients"].values()
               for index in client["sample_ids"][f"{split}_sample_ids"]]
        if sorted(ids) != list(range(count)):
            raise ValueError("Partitions do not cover the complete official split exactly once")
    records = [json.loads(line) for line in (root / "client_metrics.jsonl").read_text().splitlines()]
    expected_mu = 0.0 if arm == "fedavg" else lock["proximal_mu"]
    for round_number in range(1, config["federation"]["num_rounds"] + 1):
        for phase in ("fit", "evaluate"):
            rows = [row for row in records if row["round"] == round_number and row["phase"] == phase]
            if sorted(row["client_id"] for row in rows) != [0, 1, 2]:
                raise ValueError("Missing or duplicated client telemetry")
            if phase == "fit" and any(row["proximal_mu"] != expected_mu or
                                      row["uplink_codec"] != ARMS[arm][1] for row in rows):
                raise ValueError("Client did not execute the declared FedAvg/FedProx treatment")
    reports = [json.loads((root / f"evaluation/global_val_round_{r:04d}.json").read_text())
               for r in range(1, config["federation"]["num_rounds"] + 1)]
    best = max(range(len(reports)), key=lambda i: reports[i]["f1_macro"]) + 1
    if result["best_round"] != best:
        raise ValueError("Checkpoint was not selected by validation macro-F1")
    cell["arm"] = arm
    return cell


def compare_cells(cells, lock):
    indexed = {}
    for cell in cells:
        config = cell["manifest"]["config"]
        arm = validate_config(config, lock)
        key = (cell["result"]["alpha"], arm)
        if key in indexed:
            raise ValueError("Duplicate alpha/arm cell")
        indexed[key] = cell
    if set(indexed) != {(alpha, arm) for alpha in ALPHAS for arm in ARMS}:
        raise ValueError("Incomplete or unexpected matrix; require exactly nine cells")
    rows = []
    source = cells[0]["manifest"]["source_sha256"]
    for alpha in ALPHAS:
        baseline = indexed[alpha, "fedavg"]
        for arm in ("fedprox", "fedprox_fp16"):
            candidate = indexed[alpha, arm]
            if (candidate["test_ids"] != baseline["test_ids"] or candidate["result"]["partition_sha256"]
                    != baseline["result"]["partition_sha256"]):
                raise ValueError("Unpaired samples")
            for cell in (baseline, candidate):
                if cell["manifest"]["source_sha256"] != source:
                    raise ValueError("Source changed within the matrix")
            a, b = baseline["result"], candidate["result"]
            for key in ("total_fit_download_model_bytes", "total_evaluate_download_model_bytes"):
                if a[key] != b[key]:
                    raise ValueError("Unexpected downlink change in paired study")
            da = 100 * (b["final_test_accuracy"] - a["final_test_accuracy"])
            df = 100 * (b["final_test_f1_macro"] - a["final_test_f1_macro"])
            saving = 100 * (1 - b["total_model_payload_bytes"] / a["total_model_payload_bytes"])
            rows.append({"alpha": alpha, "arm": arm, "accuracy_delta_pp": da, "f1_delta_pp": df,
                         "upload_saving_percent": 100 * (1 - b["total_fit_upload_model_bytes"] / a["total_fit_upload_model_bytes"]),
                         "total_saving_percent": saving,
                         "non_iid_practical_gain": df >= 1 and da >= 0,
                         "communication_quality_pass": saving >= 15 and da >= -1 and df >= -1})
    return {"comparisons": rows,
            "fedprox_consistent_non_iid_gain": all(row["non_iid_practical_gain"] for row in rows if row["arm"] == "fedprox"),
            "fedprox_fp16_communication_pass_all_alphas": all(row["communication_quality_pass"] for row in rows if row["arm"] == "fedprox_fp16"),
            "replication_note": "One shared seed only; alphas are conditions, not independent seed replicates."}

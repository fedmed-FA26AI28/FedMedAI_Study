"""Resolved configuration, reproducibility, and isolated experiment artifacts."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

from fedmedai.dataset import dataset_info
from fedmedai.communication import validate_codec

SCHEMA_VERSION = 2
MODEL_METRICS = ("loss", "accuracy", "precision_macro", "recall_macro", "f1_macro",
                 "precision_weighted", "recall_weighted", "f1_weighted")
RESOURCE_FIELDS = (
    "cpu_utilization_percent_mean", "process_rss_bytes_peak", "system_ram_used_bytes_peak",
    "gpu_utilization_percent_mean", "gpu_memory_allocated_bytes_peak", "temperature_c_max",
    "power_watts_mean", "energy_joules_observed", "energy_observed_seconds", "resource_sample_count")
ROUND_FIELDS = (
    "schema_version", "run_id", "round", "split", "algorithm", "uplink_codec", "fit_clients", "fit_failures",
    "evaluate_clients", "evaluate_failures", "train_samples", "val_samples",
    "client_train_loss", "client_train_accuracy",
    *("global_val_" + key for key in MODEL_METRICS),
    *("pooled_client_val_" + key for key in MODEL_METRICS),
    "mean_client_accuracy", "client_accuracy_variance", "worst_client_accuracy",
    "mean_client_f1_macro", "worst_client_f1_macro",
    "slowest_client_train_seconds", "median_client_train_seconds",
    "straggler_overhead_seconds", "estimated_wait_seconds_sum",
    "client_train_seconds_sum", "fit_phase_seconds", "aggregation_seconds", "uplink_decode_seconds",
    "global_validation_seconds", "round_latency_seconds", "elapsed_training_seconds",
    "fit_download_model_bytes", "fit_upload_model_bytes", "evaluate_download_model_bytes",
    "round_model_payload_bytes", "cumulative_model_payload_bytes",
    "client_train_energy_joules_observed_sum", "clients_with_train_energy",
)
CLIENT_FIELDS = (
    "schema_version", "run_id", "round", "phase", "split", "client_id", "device_type",
    "hostname", "device", "num_samples", "local_epochs", "proximal_mu", "uplink_codec", "uplink_encode_seconds",
    "train_loss", "train_accuracy", *MODEL_METRICS, "duration_seconds",
    "upload_seconds", "download_seconds", "waiting_seconds", *RESOURCE_FIELDS,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def load_config(path, overrides=None):
    with open(path, encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    defaults = {
        "federation": {"num_clients": 3, "num_rounds": 5, "partition_type": "iid",
                       "dirichlet_alpha": 0.3, "seed": 42, "min_train_samples": 10,
                       "min_val_samples": 1},
        "algorithm": {"name": "fedavg", "proximal_mu": 0.01},
        "communication": {"uplink_codec": "none"},
        "model": {"name": "resnet18", "pretrained": False},
        "training": {"batch_size": 32, "local_epochs": 1, "learning_rate": 1e-4,
                     "weight_decay": 0.01, "centralized_epochs": None},
        "dataset": {"name": "BloodMNIST", "image_size": 28, "root": None, "download": True,
                    "max_train_samples": None, "max_val_samples": None, "max_test_samples": None},
        "evaluation": {"selection_metric": "f1_macro", "target_accuracy": 0.8,
                       "max_failure_cases": 100},
        "runtime": {"backend": "flower", "deterministic": True, "torch_num_threads": 1,
                    "client_cpus": 1, "client_gpus": 0.0, "ray_num_cpus": 1},
        "monitoring": {"enabled": True, "interval_seconds": 0.2, "sensors": {}},
        "hardware": {"scenario": "pc_simulation", "client_device_types": {}},
        "paths": {"checkpoints_dir": "checkpoints", "results_dir": "results", "runs_dir": "runs"},
    }
    for section, values in defaults.items():
        config[section] = {**values, **config.get(section, {})}
    for section, values in (overrides or {}).items():
        config[section].update({key: value for key, value in values.items() if value is not None})
    fed, training = config["federation"], config["training"]
    fed["partition_type"] = fed["partition_type"].lower()
    config["algorithm"]["name"] = config["algorithm"]["name"].lower()
    if fed["partition_type"] not in ("iid", "dirichlet"):
        raise ValueError("partition_type must be iid or dirichlet")
    if config["algorithm"]["name"] not in ("fedavg", "fedprox"):
        raise ValueError("algorithm.name must be fedavg or fedprox")
    validate_codec(config["communication"]["uplink_codec"])
    if config["algorithm"]["proximal_mu"] < 0:
        raise ValueError("proximal_mu must be nonnegative")
    if not isinstance(fed["seed"], int) or not 0 <= fed["seed"] < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if fed["dirichlet_alpha"] <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    for value in (fed["num_clients"], fed["num_rounds"], training["batch_size"],
                  training["local_epochs"], config["runtime"]["torch_num_threads"]):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("Client/round/epoch/batch/thread counts must be positive integers")
    if training["centralized_epochs"] is None:
        training["centralized_epochs"] = fed["num_rounds"] * training["local_epochs"]
    if training["centralized_epochs"] < 1:
        raise ValueError("centralized_epochs must be positive")
    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        cap = config["dataset"][name]
        if cap is not None and (not isinstance(cap, int) or cap < 1):
            raise ValueError(f"{name} must be positive or null")
    info = dataset_info(config["dataset"]["name"])
    n_classes = len(info["label"])
    if config["model"].get("num_classes", n_classes) != n_classes:
        raise ValueError("model.num_classes does not match the selected MedMNIST dataset")
    config["model"]["num_classes"] = n_classes
    config["evaluation"]["class_names"] = [info["label"][str(i)] for i in range(n_classes)]
    if config["evaluation"]["selection_metric"] not in ("accuracy", "f1_macro", "loss"):
        raise ValueError("selection_metric must be accuracy, f1_macro or loss")
    target = config["evaluation"]["target_accuracy"]
    if target is not None and not 0 <= target <= 1:
        raise ValueError("target_accuracy must be null or in [0, 1]")
    if config["runtime"]["backend"] not in ("flower", "sequential"):
        raise ValueError("runtime.backend must be flower or sequential")
    runtime = config["runtime"]
    if runtime["client_cpus"] <= 0 or runtime["ray_num_cpus"] < runtime["client_cpus"]:
        raise ValueError("Ray must have enough CPUs for at least one client")
    if not 0 <= runtime["client_gpus"] <= 1:
        raise ValueError("client_gpus must be in [0, 1] for single-device clients")
    if config["evaluation"]["max_failure_cases"] < 0:
        raise ValueError("max_failure_cases must be nonnegative")
    if config["monitoring"]["interval_seconds"] <= 0:
        raise ValueError("monitoring.interval_seconds must be positive")
    # Detect NaN/infinity before creating artifacts.
    json.dumps(config, allow_nan=False)
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def append_csv(path, fields, row):
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class RunArtifacts:
    def __init__(self, config, group, mode="federated"):
        group = Path(group)
        if group.is_absolute() or ".." in group.parts:
            raise ValueError("Experiment group must be a relative path without '..'")
        self.config = config
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
        self.results = Path(config["paths"]["results_dir"]) / group / self.run_id
        self.checkpoints = Path(config["paths"]["checkpoints_dir"]) / group / self.run_id
        self.runs = Path(config["paths"]["runs_dir"]) / group / self.run_id
        for path in (self.results, self.checkpoints, self.runs):
            path.mkdir(parents=True, exist_ok=False)
        versions = {}
        for package in ("fedmedai", "torch", "torchvision", "flwr", "ray", "numpy", "medmnist", "psutil"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        source_root = Path(__file__).parent
        source_hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(source_root.rglob("*.py"))}
        self.manifest = {
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id, "status": "running",
            "started_at_utc": utc_now(), "mode": mode, "config": config,
            "uplink_codec": config["communication"]["uplink_codec"] if mode == "federated" else "none",
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "hostname": platform.node(), "cuda_available": torch.cuda.is_available(),
                            "torch_cuda_version": torch.version.cuda, "packages": versions},
            "source_sha256": source_hashes,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "artifact_paths": {"results": str(self.results.resolve()),
                               "checkpoints": str(self.checkpoints.resolve()),
                               "tensorboard": str(self.runs.resolve())},
            "measurement_notes": {
                "communication": "Serialized model tensors and codec scales; includes evaluation downloads; excludes RPC headers, metrics, retries. Simulation payload, not network capture.",
                "latency": "Wall time including fit, aggregation, global validation and client validation; simulation includes scheduling.",
                "straggler": "Local compute max-minus-median and estimated wait; not measured transport waiting.",
                "energy": "Observed client training windows only; missing sensors=null. Shared-host simulation energy must not be summed as fleet energy.",
                "selection": "Best checkpoint selected only on validation; official test evaluated once at the end.",
            },
        }
        self.json("run_manifest.json", self.manifest)
        (self.results / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def json(self, name, value):
        write_json(self.results / name, value)

    def log_round(self, row):
        record = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, **row}
        append_csv(self.results / "metrics.csv", ROUND_FIELDS, record)
        self.json(f"rounds/round_{row['round']:04d}.json", record)

    def log_client(self, record):
        record = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, **record}
        with (self.results / "client_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        flat = {**record, **record.get("resources", {}), **record.get("evaluation", {})}
        append_csv(self.results / "client_metrics.csv", CLIENT_FIELDS, flat)

    def finish(self, status, **details):
        self.manifest.update(status=status, finished_at_utc=utc_now(), **details)
        self.json("run_manifest.json", self.manifest)


def save_partition_artifacts(artifacts, metadata):
    artifacts.json("partition_metadata.json", metadata)
    compact = {**metadata, "clients": {
        str(cid): {key: value for key, value in info.items() if key != "sample_ids"}
        for cid, info in metadata["clients"].items()}}
    artifacts.json("partition_summary.json", compact)

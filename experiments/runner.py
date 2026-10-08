"""Shared data preparation and final held-out test reporting."""

import csv
import time

import torch
from torch.utils.data import Subset

from datasets.medmnist_code import get_centralized_test_loader
from client.evaluate import evaluate_classification, sample_id
from experiments.artifacts import MODEL_METRICS, save_partition_artifacts
from datasets.partition import create_client_dataloaders, pooled_loader


def prepare_data(config, artifacts):
    fed, data, training = config["federation"], config["dataset"], config["training"]
    train, val, metadata = create_client_dataloaders(
        dataset_name=data["name"], partition_type=fed["partition_type"],
        num_clients=fed["num_clients"], alpha=fed["dirichlet_alpha"], seed=fed["seed"],
        batch_size=training["batch_size"], image_size=data["image_size"],
        max_train_samples=data["max_train_samples"], max_val_samples=data["max_val_samples"],
        min_train_samples=fed["min_train_samples"], min_val_samples=fed["min_val_samples"],
        root=data["root"], download=data["download"], num_workers=data.get("num_workers", 0))
    save_partition_artifacts(artifacts, metadata)
    artifacts.manifest["partition_sha256"] = metadata["partition_sha256"]
    artifacts.manifest["train_samples"] = sum(
        info["num_train_samples"] for info in metadata["clients"].values())
    artifacts.json("run_manifest.json", artifacts.manifest)
    return train, val, pooled_loader(val, training["batch_size"])


def final_test(model, device, artifacts, summary, training_seconds):
    """Called exactly once after training/selection has finished."""
    config = artifacts.config
    data, fed = config["dataset"], config["federation"]
    checkpoint = torch.load(artifacts.checkpoints / "best_model.pt", map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = get_centralized_test_loader(
        dataset_name=data["name"], batch_size=config["training"]["batch_size"],
        image_size=data["image_size"], max_samples=data["max_test_samples"],
        seed=fed["seed"], root=data["root"], download=data["download"])
    test_start = time.perf_counter()
    report = evaluate_classification(
        model, loader, device, config["model"]["num_classes"],
        config["evaluation"]["class_names"], config["evaluation"]["max_failure_cases"])
    test_seconds = time.perf_counter() - test_start
    ids = [sample_id(loader.dataset, i) for i in range(len(loader.dataset))]
    artifacts.json("evaluation/test_sample_ids.json", {
        "dataset": data["name"], "split": "test", "sample_ids": ids,
        "selection_policy": "seeded_uniform_subset" if isinstance(loader.dataset, Subset) else "official_full_split",
        "seed": fed["seed"]})
    artifacts.json("evaluation/test.json", {
        "schema_version": 2, "run_id": artifacts.run_id, "split": "test",
        "checkpoint_round": checkpoint["round"], **report})
    with (artifacts.results / "test_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class/predicted_class", *report["class_names"]])
        writer.writerows([name, *row] for name, row in zip(report["class_names"], report["confusion_matrix"]))
    with (artifacts.results / "test_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["per_class"][0]))
        writer.writeheader()
        writer.writerows(report["per_class"])
    final = {
        "schema_version": 2, "run_id": artifacts.run_id, "mode": artifacts.manifest["mode"],
        "dataset": data["name"], "model": config["model"]["name"],
        "algorithm": config["algorithm"]["name"] if artifacts.manifest["mode"] == "federated" else "centralized",
        "partition_type": fed["partition_type"],
        "alpha": fed["dirichlet_alpha"] if fed["partition_type"] == "dirichlet" else None,
        "random_seed": fed["seed"], "num_clients": fed["num_clients"],
        "num_rounds": fed["num_rounds"] if artifacts.manifest["mode"] == "federated" else 0,
        "hardware_scenario": config["hardware"]["scenario"],
        "execution_backend": config["runtime"]["backend"] if artifacts.manifest["mode"] == "federated" else "centralized",
        "partition_sha256": artifacts.manifest["partition_sha256"],
        "train_samples": artifacts.manifest["train_samples"],
        "test_samples": report["num_samples"], "training_wall_seconds": training_seconds,
        "final_test_evaluation_seconds": test_seconds, "test_checkpoint_round": checkpoint["round"],
        **{"final_test_" + metric: report[metric] for metric in MODEL_METRICS},
        **summary, "test_evaluation": report,
        "results_dir": str(artifacts.results.resolve()),
    }
    artifacts.json("final_test_metrics.json", final)
    artifacts.finish("completed")
    return final

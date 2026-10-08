"""Centralized baseline using the same selected train/val samples as FL."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import time

import torch
from torch.utils.tensorboard import SummaryWriter

from client.evaluate import evaluate_classification
from experiments.artifacts import MODEL_METRICS, RunArtifacts, load_config, seed_everything
from models.cnn import get_device, get_model
from monitoring.resource import ResourceMonitor
from datasets.partition import pooled_loader
from experiments.runner import final_test, prepare_data
from client.train import train_local


def run_centralized(config_path="configs/experiment.yaml", seed_override=None,
                    partition_type_override=None, alpha_override=None):
    config = load_config(config_path, {"federation": {
        "seed": seed_override, "partition_type": partition_type_override,
        "dirichlet_alpha": alpha_override}})
    fed, training, model_cfg = config["federation"], config["training"], config["model"]
    group = "iid" if fed["partition_type"] == "iid" else f"alpha_{fed['dirichlet_alpha']}"
    artifacts = RunArtifacts(config, f"centralized/{group}", mode="centralized")
    writer = None
    try:
        seed_everything(fed["seed"], config["runtime"]["deterministic"])
        torch.set_num_threads(config["runtime"]["torch_num_threads"])
        train, _, val = prepare_data(config, artifacts)
        train_loader = pooled_loader(train, training["batch_size"], shuffle=True, seed=fed["seed"])
        model = get_model(model_name=model_cfg["name"], num_classes=model_cfg["num_classes"],
                          pretrained=model_cfg["pretrained"])
        artifacts.manifest["model_parameters"] = sum(p.numel() for p in model.parameters())
        artifacts.json("run_manifest.json", artifacts.manifest)
        device = get_device()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                     weight_decay=training["weight_decay"])
        writer = SummaryWriter(log_dir=str(artifacts.runs))
        metric = config["evaluation"]["selection_metric"]
        best_score, best_epoch, best_report, target_reached = None, None, None, None
        observed_energy = []
        started = time.perf_counter()
        for epoch in range(1, training["centralized_epochs"] + 1):
            epoch_started = time.perf_counter()
            with ResourceMonitor(device, **config["monitoring"]) as monitor:
                loss, accuracy = train_local(
                    model, train_loader, 1, training["learning_rate"], training["weight_decay"],
                    device, optimizer=optimizer)
            artifacts.log_client({
                "round": epoch, "phase": "fit", "split": "train", "client_id": "centralized",
                "device_type": "centralized", "num_samples": len(train_loader.dataset),
                "train_loss": loss, "train_accuracy": accuracy, "local_epochs": 1,
                "proximal_mu": 0.0, "resources": monitor.result})
            energy = monitor.result["energy_joules_observed"]
            if energy is not None:
                observed_energy.append(energy)
            validation_start = time.perf_counter()
            report = evaluate_classification(model, val, device, model_cfg["num_classes"],
                                             config["evaluation"]["class_names"])
            validation_seconds = time.perf_counter() - validation_start
            artifacts.json(f"evaluation/global_val_epoch_{epoch:04d}.json",
                           {"epoch": epoch, "split": "val", **report})
            checkpoint = {
                "schema_version": 2, "run_id": artifacts.run_id, "round": epoch, "epoch": epoch,
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "selection_split": "val", "selection_metric": metric,
                "validation_metrics": report, "model_config": model_cfg}
            torch.save(checkpoint, artifacts.checkpoints / f"epoch_{epoch}.pt")
            score = report[metric]
            if best_score is None or (score < best_score if metric == "loss" else score > best_score):
                best_score, best_epoch, best_report = score, epoch, report
                torch.save(checkpoint, artifacts.checkpoints / "best_model.pt")
            elapsed = time.perf_counter() - started
            target = config["evaluation"]["target_accuracy"]
            if target_reached is None and target is not None and report["accuracy"] >= target:
                target_reached = {"epoch": epoch, "seconds": elapsed}
            row = {
                "round": epoch, "split": "val", "algorithm": "centralized",
                "train_samples": len(train_loader.dataset), "val_samples": report["num_samples"],
                "client_train_loss": loss, "client_train_accuracy": accuracy,
                **{"global_val_" + key: report[key] for key in MODEL_METRICS},
                "client_train_seconds_sum": monitor.result["duration_seconds"],
                "global_validation_seconds": validation_seconds,
                "round_latency_seconds": time.perf_counter() - epoch_started,
                "elapsed_training_seconds": elapsed, "round_model_payload_bytes": 0,
                "cumulative_model_payload_bytes": 0,
                "client_train_energy_joules_observed_sum": energy,
                "clients_with_train_energy": int(energy is not None)}
            artifacts.log_round(row)
            for key in MODEL_METRICS:
                writer.add_scalar("Validation/" + key, report[key], epoch)
            writer.flush()
        summary = {
            "completed_epochs": training["centralized_epochs"], "best_epoch": best_epoch,
            "selection_split": "val", "selection_metric": metric,
            "best_validation_metrics": best_report, "total_model_payload_bytes": 0,
            "target_accuracy": config["evaluation"]["target_accuracy"],
            "epochs_to_target": target_reached["epoch"] if target_reached else None,
            "time_to_target_seconds": target_reached["seconds"] if target_reached else None,
            "target_reached": target_reached is not None,
            "client_train_energy_joules_observed_sum": sum(observed_energy) if observed_energy else None,
        }
        result = final_test(model, device, artifacts, summary, time.perf_counter() - started)
        print(f"Centralized results: {artifacts.results}")
        return result
    except BaseException as exc:
        artifacts.finish("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if writer is not None:
            writer.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    run_centralized(args.config)


if __name__ == "__main__":
    main()

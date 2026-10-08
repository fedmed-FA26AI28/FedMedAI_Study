"""Flower clients share training/evaluation and emit versioned JSON telemetry."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json
import time

import flwr as fl

from algorithms.communication import encode_uplink, validate_codec
from algorithms.coverage import CALIBRATION_METHODS, COVERAGE_METHODS, HEAD_METHODS
from client.evaluate import evaluate_classification
from experiments.artifacts import seed_everything
from models.cnn import get_device, get_parameters, set_parameters
from monitoring.resource import ResourceMonitor
from client.train import train_local


class FedMedClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, train_loader, val_loader, local_epochs=1,
                 lr=1e-4, weight_decay=1e-2, num_classes=8, class_names=None,
                 device=None, device_type="pc", monitoring=None,
                 seed=42, deterministic=True):
        self.client_id = int(client_id)
        self.model = model
        self.train_loader, self.val_loader = train_loader, val_loader
        self.local_epochs, self.lr, self.weight_decay = local_epochs, lr, weight_decay
        self.num_classes, self.class_names = num_classes, class_names
        self.device = device if device is not None else get_device()
        self.device_type = device_type
        self.monitoring = monitoring or {}
        self.seed, self.deterministic = seed, deterministic
        self.previous_local_weights = None
        self.previous_fit_round = 0

    def get_parameters(self, config):
        return get_parameters(self.model)

    def _monitor(self):
        return ResourceMonitor(self.device, **self.monitoring)

    def _base_record(self, config, phase, split, n):
        return {"client_id": self.client_id, "round": int(config.get("round", 0)),
                "phase": phase, "split": split, "num_samples": n,
                "device_type": self.device_type,
                "upload_seconds": None, "download_seconds": None, "waiting_seconds": None}

    def fit(self, parameters, config):
        codec = validate_codec(config.get("uplink_codec", "none"))
        algorithm = config.get("algorithm", "fedprox" if config.get("proximal_mu", 0) else "fedavg")
        set_parameters(self.model, parameters)
        server_round = int(config.get("round", 0))
        local_seed = (self.seed + 1000003 * server_round + self.client_id) % (2 ** 32)
        seed_everything(local_seed, self.deterministic)
        if self.train_loader.generator is not None:
            self.train_loader.generator.manual_seed(local_seed)
        epochs = int(config.get("local_epochs", self.local_epochs))
        mu = float(config.get("proximal_mu", 0.0)) if algorithm in (
            "fedprox", "calibrated_fedprox") else 0.0
        head_mu = float(config.get("head_mu", 1.0))
        coverage_kappa = float(config.get("coverage_kappa", 32.0))
        logit_tau = float(config.get("logit_tau", 1.0))
        prior_smoothing = float(config.get("prior_smoothing", 1.0))
        diagnostics = {} if algorithm in COVERAGE_METHODS or config.get("research_diagnostics") else None
        adaptation = None
        with self._monitor() as monitor:
            loss, accuracy = train_local(
                self.model, self.train_loader, epochs, float(config.get("lr", self.lr)),
                float(config.get("weight_decay", self.weight_decay)), self.device,
                proximal_mu=mu, algorithm=algorithm, head_mu=head_mu,
                coverage_kappa=coverage_kappa, logit_tau=logit_tau,
                prior_smoothing=prior_smoothing, diagnostics=diagnostics,
                max_steps=config.get("max_steps"))
        n = len(self.train_loader.dataset)
        encode_started = time.perf_counter()
        weights = encode_uplink(get_parameters(self.model), codec)
        encode_seconds = time.perf_counter() - encode_started
        record = {**self._base_record(config, "fit", "train", n),
                  "local_epochs": epochs, "proximal_mu": mu, "train_loss": loss,
                  "uplink_codec": codec, "uplink_encode_seconds": encode_seconds,
                  "train_accuracy": accuracy, "resources": monitor.result}
        if adaptation is not None:
            record.update(adaptation)
        if diagnostics is not None:
            record.update(diagnostics)
            record.update(algorithm=algorithm,
                          head_mu=head_mu if algorithm in HEAD_METHODS else 0.0,
                          coverage_kappa=coverage_kappa,
                          prior_smoothing=prior_smoothing,
                          logit_tau=logit_tau if algorithm in CALIBRATION_METHODS else 0.0)
        return weights, n, {
            "uplink_codec": codec,
            "client_id": self.client_id, "train_loss": loss, "train_acc": accuracy,
            "telemetry_json": json.dumps(record, allow_nan=False)}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        with self._monitor() as monitor:
            report = evaluate_classification(self.model, self.val_loader, self.device,
                                             self.num_classes, self.class_names)
        n = report["num_samples"]
        record = {**self._base_record(config, "evaluate", "val", n),
                  "evaluation": report, "resources": monitor.result}
        return report["loss"], n, {
            "client_id": self.client_id, "eval_loss": report["loss"],
            "eval_acc": report["accuracy"], "eval_f1_macro": report["f1_macro"],
            "telemetry_json": json.dumps(record, allow_nan=False)}



def main():
    """Connect a real client using the same deterministic train/val partition."""
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--max-wait-time", type=float, default=60.0)
    args = parser.parse_args()
    from experiments.artifacts import load_config
    from datasets.partition import create_client_dataloaders
    from models.cnn import get_model
    import torch
    config = load_config(args.config)
    fed, data, training = config["federation"], config["dataset"], config["training"]
    if not 0 <= args.client_id < fed["num_clients"]:
        parser.error("client-id must be within the configured client count")
    seed_everything(fed["seed"], config["runtime"]["deterministic"])
    torch.set_num_threads(config["runtime"]["torch_num_threads"])
    train, val, _ = create_client_dataloaders(dataset_name=data["name"],
        partition_type=fed["partition_type"], num_clients=fed["num_clients"],
        alpha=fed["dirichlet_alpha"], seed=fed["seed"], batch_size=training["batch_size"],
        image_size=data["image_size"], max_train_samples=data["max_train_samples"],
        max_val_samples=data["max_val_samples"], min_train_samples=fed["min_train_samples"],
        min_val_samples=fed["min_val_samples"], root=data["root"], download=data["download"], num_workers=data.get("num_workers", 0))
    instance = FedMedClient(args.client_id, get_model(num_classes=config["model"]["num_classes"]),
        train[args.client_id], val[args.client_id], local_epochs=training["local_epochs"],
        lr=training["learning_rate"], weight_decay=training["weight_decay"],
        num_classes=config["model"]["num_classes"], class_names=config["evaluation"]["class_names"],
        monitoring=config["monitoring"], seed=fed["seed"], deterministic=config["runtime"]["deterministic"])
    if args.max_wait_time <= 0:
        parser.error("max-wait-time must be positive")
    fl.client.start_client(server_address=args.server_address, client=instance.to_client(),
                           max_wait_time=args.max_wait_time)


if __name__ == "__main__":
    main()


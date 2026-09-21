"""Flower clients share training/evaluation and emit versioned JSON telemetry."""

import json

import flwr as fl

from fedmedai.evaluate import evaluate_classification
from fedmedai.experiment import seed_everything
from fedmedai.model import get_device, get_weights, set_weights
from fedmedai.monitoring.resource import ResourceMonitor
from fedmedai.train import train_local


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

    def get_parameters(self, config):
        return get_weights(self.model)

    def _monitor(self):
        return ResourceMonitor(self.device, **self.monitoring)

    def _base_record(self, config, phase, split, n):
        return {"client_id": self.client_id, "round": int(config.get("round", 0)),
                "phase": phase, "split": split, "num_samples": n,
                "device_type": self.device_type,
                "upload_seconds": None, "download_seconds": None, "waiting_seconds": None}

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        server_round = int(config.get("round", 0))
        local_seed = (self.seed + 1000003 * server_round + self.client_id) % (2 ** 32)
        seed_everything(local_seed, self.deterministic)
        if self.train_loader.generator is not None:
            self.train_loader.generator.manual_seed(local_seed)
        epochs = int(config.get("local_epochs", self.local_epochs))
        mu = float(config.get("proximal_mu", 0.0))
        with self._monitor() as monitor:
            loss, accuracy = train_local(
                self.model, self.train_loader, epochs, float(config.get("lr", self.lr)),
                float(config.get("weight_decay", self.weight_decay)), self.device,
                proximal_mu=mu)
        n = len(self.train_loader.dataset)
        record = {**self._base_record(config, "fit", "train", n),
                  "local_epochs": epochs, "proximal_mu": mu, "train_loss": loss,
                  "train_accuracy": accuracy, "resources": monitor.result}
        return get_weights(self.model), n, {
            "client_id": self.client_id, "train_loss": loss, "train_acc": accuracy,
            "telemetry_json": json.dumps(record, allow_nan=False)}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
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


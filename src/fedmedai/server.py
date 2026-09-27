"""Flower aggregation and validation; artifact storage is delegated to RunArtifacts."""

import json
import time
from dataclasses import replace

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from fedmedai.communication import decode_uplink
from fedmedai.evaluate import evaluate_classification
from fedmedai.experiment import MODEL_METRICS
from fedmedai.model import get_weights, set_weights
from fedmedai.monitoring.metrics import classification_metrics, client_fairness, straggler_metrics


def model_payload_bytes(parameters):
    """Serialized ndarray payload, excluding Flower/RPC envelope and scalar metrics."""
    return sum(len(tensor) for tensor in parameters.tensors)


class FedMedFedAvg(fl.server.strategy.FedAvg):
    """Sample-weighted FedAvg; FedProx uses the same aggregation and local penalty."""

    def __init__(self, global_model, val_loader, artifacts, writer, device):
        self.global_model, self.val_loader = global_model, val_loader
        self.artifacts, self.writer, self.device = artifacts, writer, device
        cfg = artifacts.config
        self.cfg = cfg
        self.uplink_codec = cfg.get("communication", {}).get("uplink_codec", "none")
        self.num_classes = cfg["model"]["num_classes"]
        self.class_names = cfg["evaluation"]["class_names"]
        self.selection_metric = cfg["evaluation"]["selection_metric"]
        self.best_score = None
        self.best_round = None
        self.best_validation = None
        self.round_history = {}
        self.fit_records = {}
        self.cumulative_bytes = 0
        self.training_started = None
        self.target_reached = None
        self.completed_rounds = 0
        clients = cfg["federation"]["num_clients"]
        super().__init__(
            fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=clients,
            min_evaluate_clients=clients, min_available_clients=clients,
            on_fit_config_fn=self.fit_config, on_evaluate_config_fn=lambda r: {"round": r},
            initial_parameters=ndarrays_to_parameters(get_weights(global_model)),
            accept_failures=False, fit_metrics_aggregation_fn=lambda _: {},
            evaluate_metrics_aggregation_fn=lambda _: {},
        )

    def fit_config(self, server_round):
        training, algorithm = self.cfg["training"], self.cfg["algorithm"]
        return {"round": server_round, "local_epochs": training["local_epochs"],
                "uplink_codec": self.uplink_codec,
                "lr": training["learning_rate"], "weight_decay": training["weight_decay"],
                "proximal_mu": algorithm["proximal_mu"] if algorithm["name"] == "fedprox" else 0.0}

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_started = time.perf_counter()
        if self.training_started is None:
            self.training_started = self.round_started
        instructions = super().configure_fit(server_round, parameters, client_manager)
        self.round_history[server_round] = {
            "round": server_round, "split": "val", "algorithm": self.cfg["algorithm"]["name"],
            "uplink_codec": self.uplink_codec,
            "fit_clients": len(instructions),
            "fit_download_model_bytes": sum(model_payload_bytes(ins.parameters) for _, ins in instructions),
            "fit_upload_model_bytes": 0, "evaluate_download_model_bytes": 0,
        }
        return instructions

    def _client_records(self, server_round, results, phase):
        records = []
        seen = set()
        for proxy, result in results:
            record = json.loads(result.metrics["telemetry_json"])
            cid = record["client_id"]
            if cid in seen or cid not in range(self.cfg["federation"]["num_clients"]):
                raise ValueError(f"Duplicate or invalid logical client ID: {cid}")
            seen.add(cid)
            if record["round"] != server_round or record["phase"] != phase:
                raise ValueError(f"Mismatched client telemetry from {proxy.cid}")
            if record["num_samples"] != result.num_examples or result.num_examples < 1:
                raise ValueError(f"Invalid sample count from {proxy.cid}")
            records.append(record)
            self.artifacts.log_client(record)
        return records

    def _check_results(self, server_round, phase, results, failures):
        if failures or len(results) != self.cfg["federation"]["num_clients"]:
            self.artifacts.json(f"failures/round_{server_round:04d}_{phase}.json", {
                "round": server_round, "phase": phase, "successful_clients": len(results),
                "failures": [str(item) for item in failures],
                "policy": "require_all_clients; round_not_accepted"})
            raise RuntimeError(f"Round {server_round} {phase} did not complete for all clients")

    def aggregate_fit(self, server_round, results, failures):
        # Async completion order must not change floating-point reduction order.
        results = sorted(results, key=lambda item: int(item[1].metrics["client_id"]))
        row = self.round_history[server_round]
        row["fit_failures"] = len(failures)
        records = self._client_records(server_round, results, "fit")
        self._check_results(server_round, "fit", results, failures)
        self.fit_records[server_round] = records
        # Count the received serialized payload, including INT8 scale metadata,
        # before replacing it with restored arrays for the standard aggregator.
        upload_bytes = sum(model_payload_bytes(result.parameters) for _, result in results)
        decode_started = time.perf_counter()
        template = get_weights(self.global_model)
        decoded_results = []
        for (proxy, result), record in zip(results, records):
            if (result.metrics.get("uplink_codec", "none") != self.uplink_codec
                    or record.get("uplink_codec", "none") != self.uplink_codec):
                raise ValueError("Client uplink codec differs from server configuration")
            arrays = decode_uplink(parameters_to_ndarrays(result.parameters), template, self.uplink_codec)
            decoded_results.append((proxy, replace(result, parameters=ndarrays_to_parameters(arrays))))
        row["uplink_decode_seconds"] = time.perf_counter() - decode_started
        start = time.perf_counter()
        parameters, metrics = super().aggregate_fit(server_round, decoded_results, failures)
        row["aggregation_seconds"] = time.perf_counter() - start
        if parameters is None:
            raise RuntimeError("Aggregation produced no parameters")
        n = sum(result.num_examples for _, result in results)
        row.update(
            train_samples=n,
            client_train_loss=sum(r["train_loss"] * r["num_samples"] for r in records) / n,
            client_train_accuracy=sum(r["train_accuracy"] * r["num_samples"] for r in records) / n,
            fit_upload_model_bytes=upload_bytes,
            fit_phase_seconds=time.perf_counter() - self.round_started,
        )
        times = [r["resources"]["duration_seconds"] for r in records]
        row.update(straggler_metrics(times))
        row["client_train_seconds_sum"] = sum(times)
        energies = [r["resources"]["energy_joules_observed"] for r in records
                    if r["resources"]["energy_joules_observed"] is not None]
        row["client_train_energy_joules_observed_sum"] = sum(energies) if energies else None
        row["clients_with_train_energy"] = len(energies)
        return parameters, metrics

    def evaluate(self, server_round, parameters):
        if server_round == 0:
            return None
        start = time.perf_counter()
        set_weights(self.global_model, parameters_to_ndarrays(parameters))
        report = evaluate_classification(self.global_model, self.val_loader, self.device,
                                         self.num_classes, self.class_names)
        self.artifacts.json(f"evaluation/global_val_round_{server_round:04d}.json",
                            {"round": server_round, "split": "val", **report})
        row = self.round_history[server_round]
        row["global_validation_seconds"] = time.perf_counter() - start
        row["val_samples"] = report["num_samples"]
        row.update({"global_val_" + key: report[key] for key in MODEL_METRICS})
        score = report[self.selection_metric]
        improved = self.best_score is None or (
            score < self.best_score if self.selection_metric == "loss" else score > self.best_score)
        checkpoint = {
            "schema_version": 2, "run_id": self.artifacts.run_id, "round": server_round,
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()},
            "selection_split": "val", "selection_metric": self.selection_metric,
            "validation_metrics": report, "model_config": self.cfg["model"],
        }
        torch.save(checkpoint, self.artifacts.checkpoints / f"global_round_{server_round}.pt")
        if improved:
            self.best_score, self.best_round, self.best_validation = score, server_round, report
            torch.save(checkpoint, self.artifacts.checkpoints / "best_model.pt")
        return report["loss"], {key: report[key] for key in MODEL_METRICS if key != "loss"}

    def configure_evaluate(self, server_round, parameters, client_manager):
        instructions = super().configure_evaluate(server_round, parameters, client_manager)
        if server_round in self.round_history:
            self.round_history[server_round]["evaluate_download_model_bytes"] = sum(
                model_payload_bytes(ins.parameters) for _, ins in instructions)
        return instructions

    def aggregate_evaluate(self, server_round, results, failures):
        results = sorted(results, key=lambda item: int(item[1].metrics["client_id"]))
        records = self._client_records(server_round, results, "evaluate")
        self._check_results(server_round, "evaluate", results, failures)
        reports = [record["evaluation"] for record in records]
        n = sum(report["num_samples"] for report in reports)
        pooled = classification_metrics(
            np.sum([r["confusion_matrix"] for r in reports], axis=0), self.class_names)
        pooled["loss"] = sum(r["loss"] * r["num_samples"] for r in reports) / n
        self.artifacts.json(f"evaluation/pooled_client_val_round_{server_round:04d}.json",
                            {"round": server_round, "split": "val", **pooled})
        row = self.round_history[server_round]
        row.update(evaluate_clients=len(records), evaluate_failures=len(failures))
        row.update({"pooled_client_val_" + key: pooled[key] for key in MODEL_METRICS})
        row.update(client_fairness(reports))
        row["round_model_payload_bytes"] = (
            row["fit_download_model_bytes"] + row["fit_upload_model_bytes"]
            + row["evaluate_download_model_bytes"])
        self.cumulative_bytes += row["round_model_payload_bytes"]
        row["cumulative_model_payload_bytes"] = self.cumulative_bytes
        row["round_latency_seconds"] = time.perf_counter() - self.round_started
        row["elapsed_training_seconds"] = time.perf_counter() - self.training_started
        target = self.cfg["evaluation"]["target_accuracy"]
        if self.target_reached is None and target is not None and row["global_val_accuracy"] >= target:
            self.target_reached = {"round": server_round, "elapsed_training_seconds": row["elapsed_training_seconds"],
                                   "model_payload_bytes": self.cumulative_bytes}
        self.completed_rounds += 1
        self.artifacts.log_round(row)
        for key, value in row.items():
            if isinstance(value, (int, float)) and key != "round":
                self.writer.add_scalar("Round/" + key, value, server_round)
        self.writer.flush()
        print(f"Round {server_round}: val accuracy={row['global_val_accuracy']:.4f}, "
              f"macro F1={row['global_val_f1_macro']:.4f}, "
              f"worst client={row['worst_client_accuracy']:.4f}")
        return pooled["loss"], {key: pooled[key] for key in MODEL_METRICS if key != "loss"}

    def summary(self):
        rows = list(self.round_history.values())
        observed_energy = [r["client_train_energy_joules_observed_sum"] for r in rows
                           if r.get("client_train_energy_joules_observed_sum") is not None]
        return {
            "completed_rounds": self.completed_rounds, "best_round": self.best_round,
            "selection_split": "val", "selection_metric": self.selection_metric,
            "best_validation_metrics": self.best_validation,
            "uplink_codec": self.uplink_codec,
            "total_fit_upload_model_bytes": sum(r.get("fit_upload_model_bytes", 0) for r in rows),
            "total_fit_download_model_bytes": sum(r.get("fit_download_model_bytes", 0) for r in rows),
            "total_evaluate_download_model_bytes": sum(r.get("evaluate_download_model_bytes", 0) for r in rows),
            "total_model_payload_bytes": self.cumulative_bytes,
            "target_accuracy": self.cfg["evaluation"]["target_accuracy"],
            "rounds_to_target": self.target_reached["round"] if self.target_reached else None,
            "time_to_target_seconds": self.target_reached["elapsed_training_seconds"] if self.target_reached else None,
            "model_payload_bytes_to_target": self.target_reached["model_payload_bytes"] if self.target_reached else None,
            "target_reached": self.target_reached is not None,
            "client_train_energy_joules_observed_sum": sum(observed_energy) if observed_energy else None,
            "total_fleet_energy_joules": None,
            "fairness_scope": "last_round_global_model_on_client_validation_partitions",
            "last_round_fairness": client_fairness([]) if not rows else {
                key: rows[-1].get(key) for key in client_fairness([])},
        }


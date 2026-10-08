"""Explicit sequential research policies for server damping and local BatchNorm.

Ordinary runs continue to use FedMedFedAvg. No adaptive optimizer is applied to
BN running moments or integer counters. This study retains the full-state wire
protocol, including local-BN states, and therefore claims no BN payload savings.
"""

import math
import time

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from algorithms.fedavg import FedMedFedAvg
from client.evaluate import evaluate_classification
from experiments.artifacts import MODEL_METRICS
from models.cnn import get_parameters, set_parameters
from monitoring.metrics import classification_metrics, client_fairness


def batchnorm_keys(model):
    keys = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            keys.update(f"{name}.{key}" for key in module.state_dict())
    return keys


def pooled_reports(reports, class_names):
    report = classification_metrics(np.sum([r["confusion_matrix"] for r in reports], axis=0), class_names)
    report["loss"] = sum(r["loss"] * r["num_samples"] for r in reports) / report["num_samples"]
    report["balanced_accuracy"] = report["recall_macro"]
    report["clients"] = reports
    report["client_fairness"] = client_fairness(reports)
    return report


class ServerUpdate:
    """Apply heavy-ball/damping only to trainable coordinates; eta=1,beta=0 is exact."""
    def __init__(self, model, eta=1.0, beta=0.0, local_bn=False):
        if not math.isfinite(eta) or not 0 < eta <= 1 or not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("Require 0 < server_eta <= 1 and 0 <= server_beta < 1")
        self.eta, self.beta, self.local_bn = eta, beta, local_bn
        self.keys = list(model.state_dict())
        self.trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.bn_keys = batchnorm_keys(model)
        self.velocity = {}

    def apply(self, before, average):
        output = []
        for name, old, avg in zip(self.keys, before, average):
            if self.local_bn and name in self.bn_keys:
                value = old.copy()
            elif name in self.trainable and (self.eta != 1 or self.beta != 0):
                delta = avg - old
                velocity = self.beta * self.velocity.get(name, np.zeros_like(delta)) + delta
                self.velocity[name] = velocity
                value = old + self.eta * velocity
            else:
                # Preserve Flower's reduction result, including the exact eta=1 control path.
                value = avg
            output.append(value)
        return output


class HeterogeneousFedAvg(FedMedFedAvg):
    """Same Flower aggregation path with explicit server and normalization policies."""
    def __init__(self, *args, research_policy, **kwargs):
        self.policy = research_policy
        self.val_loaders = self.policy["val_loaders"]
        self.local_bn = self.policy.get("bn_policy", "shared") == "local"
        if self.policy.get("bn_policy", "shared") not in ("shared", "local"):
            raise ValueError("Unknown BN policy")
        super().__init__(*args, **kwargs)
        self.server_update = ServerUpdate(self.global_model, self.policy.get("server_eta", 1.),
                                          self.policy.get("server_beta", 0.), self.local_bn)
        self.local_bn_states = {}

    def fit_config(self, server_round):
        config = super().fit_config(server_round)
        config["research_diagnostics"] = True
        if self.policy.get("max_steps") is not None:
            config["max_steps"] = self.policy["max_steps"]
        return config

    def aggregate_fit(self, server_round, results, failures):
        before = get_parameters(self.global_model)
        bn_indices = [i for i, key in enumerate(self.server_update.keys) if key in self.server_update.bn_keys]
        deltas, weights = [], []
        for _, result in sorted(results, key=lambda x: int(x[1].metrics["client_id"])):
            arrays = parameters_to_ndarrays(result.parameters)
            cid = int(result.metrics["client_id"])
            self.local_bn_states[cid] = {i: arrays[i].copy() for i in bn_indices}
            flat = np.concatenate([(arrays[i] - before[i]).ravel() for i,key in enumerate(self.server_update.keys)
                                   if key in self.server_update.trainable
                                   and not (self.local_bn and key in self.server_update.bn_keys)])
            deltas.append(flat.astype(np.float64))
            weights.append(result.num_examples)
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        arrays = self.server_update.apply(before, parameters_to_ndarrays(parameters))
        p = np.asarray(weights, dtype=float) / sum(weights)
        norms = np.array([np.linalg.norm(d) for d in deltas])
        denominator = float(p @ norms)
        ratio = float(np.linalg.norm(sum(w*d for w,d in zip(p,deltas))) / denominator) if denominator else None
        self.round_history[server_round]["update_cancellation_ratio"] = ratio
        self.round_history[server_round]["local_update_norms"] = norms.tolist()
        self.round_history[server_round]["server_eta"] = self.server_update.eta
        self.round_history[server_round]["server_beta"] = self.server_update.beta
        return ndarrays_to_parameters(arrays), metrics

    def _evaluate_domains(self, loaders, state, bn_states):
        shared = get_parameters(self.global_model)
        reports = []
        for cid in sorted(loaders):
            arrays = [bn_states[cid][i] if self.local_bn and i in bn_states[cid] else p
                      for i,p in enumerate(state)]
            set_parameters(self.global_model, arrays)
            reports.append(evaluate_classification(self.global_model, loaders[cid], self.device,
                                                   self.num_classes, self.class_names))
        set_parameters(self.global_model, shared)
        return pooled_reports(reports, self.class_names)

    def evaluate(self, server_round, parameters):
        if server_round == 0:
            return None
        started = time.perf_counter()
        state = parameters_to_ndarrays(parameters)
        set_parameters(self.global_model, state)
        report = self._evaluate_domains(self.val_loaders, state, self.local_bn_states)
        self.artifacts.json(f"evaluation/global_val_round_{server_round:04d}.json",
                            {"round":server_round,"split":"val",**report})
        row = self.round_history[server_round]
        row.update(global_validation_seconds=time.perf_counter()-started,val_samples=report["num_samples"])
        row.update({"global_val_"+key:report[key] for key in MODEL_METRICS})
        score = report[self.selection_metric]
        improved = self.best_score is None or score > self.best_score
        checkpoint = {"schema_version":3,"run_id":self.artifacts.run_id,"round":server_round,
                      "model_state_dict":{k:v.detach().cpu().clone() for k,v in self.global_model.state_dict().items()},
                      "selection_split":"val","selection_metric":self.selection_metric,
                      "validation_metrics":report,"model_config":self.cfg["model"],
                      "bn_policy":"local" if self.local_bn else "shared",
                      "client_bn_states":{cid:{i:torch.as_tensor(v.copy()) for i,v in values.items()}
                                          for cid,values in self.local_bn_states.items()} if self.local_bn else {},
                      "server_velocity":{k:torch.as_tensor(v.copy()) for k,v in self.server_update.velocity.items()}}
        torch.save(checkpoint,self.artifacts.checkpoints/f"global_round_{server_round}.pt")
        if improved:
            self.best_score,self.best_round,self.best_validation=score,server_round,report
            torch.save(checkpoint,self.artifacts.checkpoints/"best_model.pt")
        return report["loss"],{key:report[key] for key in MODEL_METRICS if key!="loss"}

    def evaluate_checkpoint(self, checkpoint, loaders):
        self.global_model.load_state_dict(checkpoint["model_state_dict"])
        bn = {int(cid):{int(i):v.numpy() for i,v in values.items()}
              for cid,values in checkpoint["client_bn_states"].items()}
        return self._evaluate_domains(loaders,get_parameters(self.global_model),bn)

    def summary(self):
        return {**super().summary(),"bn_policy":"local" if self.local_bn else "shared",
                "server_eta":self.server_update.eta,"server_beta":self.server_update.beta,
                "inference_endpoint":"known_client_personalized" if self.local_bn else "shared_global",
                "local_bn_communication_policy":"full_state_protocol; no communication saving claimed"}

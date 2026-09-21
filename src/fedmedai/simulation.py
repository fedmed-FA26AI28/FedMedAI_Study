"""Federated orchestration with Flower or an explicit sequential debug backend."""

import time
from types import SimpleNamespace

import flwr as fl
import torch
from flwr.common import Code, EvaluateRes, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from torch.utils.tensorboard import SummaryWriter

from fedmedai.client import FedMedClient
from fedmedai.experiment import RunArtifacts, load_config, seed_everything
from fedmedai.model import get_device, get_model, get_weights
from fedmedai.runner import final_test, prepare_data
from fedmedai.server import FedMedFedAvg


def run_sequential(strategy, clients, num_rounds):
    """Exercise the same Flower protocol synchronously, without Ray or a network.

    This backend is for correctness/debugging, not hardware latency comparisons.
    """
    manager = SimpleClientManager()
    for cid in clients:
        manager.register(SimpleNamespace(cid=str(cid)))
    parameters = ndarrays_to_parameters(get_weights(strategy.global_model))
    status = Status(code=Code.OK, message="")
    for server_round in range(1, num_rounds + 1):
        fit_results = []
        for proxy, ins in strategy.configure_fit(server_round, parameters, manager):
            weights, n, metrics = clients[int(proxy.cid)].fit(
                parameters_to_ndarrays(ins.parameters), ins.config)
            fit_results.append((proxy, FitRes(status, ndarrays_to_parameters(weights), n, metrics)))
        parameters, _ = strategy.aggregate_fit(server_round, fit_results, [])
        strategy.evaluate(server_round, parameters)
        eval_results = []
        for proxy, ins in strategy.configure_evaluate(server_round, parameters, manager):
            loss, n, metrics = clients[int(proxy.cid)].evaluate(
                parameters_to_ndarrays(ins.parameters), ins.config)
            eval_results.append((proxy, EvaluateRes(status, loss, n, metrics)))
        strategy.aggregate_evaluate(server_round, eval_results, [])


def run_federated_simulation(config_path="configs/config.yaml", partition_type_override=None,
                             alpha_override=None, output_subdir_override=None,
                             num_rounds_override=None, max_train_samples_override=None,
                             max_val_samples_override=None, algorithm_override=None,
                             seed_override=None, backend_override=None, num_clients_override=None):
    config = load_config(config_path, {
        "federation": {"partition_type": partition_type_override, "dirichlet_alpha": alpha_override,
                       "num_rounds": num_rounds_override, "seed": seed_override,
                       "num_clients": num_clients_override},
        "dataset": {"max_train_samples": max_train_samples_override, "max_val_samples": max_val_samples_override},
        "algorithm": {"name": algorithm_override}, "runtime": {"backend": backend_override}})
    fed, training, model_cfg = config["federation"], config["training"], config["model"]
    group = output_subdir_override or (
        "iid" if fed["partition_type"] == "iid" else f"non_iid/alpha_{fed['dirichlet_alpha']}")
    group = f"{group}/{config['algorithm']['name']}"
    artifacts = RunArtifacts(config, group)
    writer = None
    try:
        seed_everything(fed["seed"], config["runtime"]["deterministic"])
        torch.set_num_threads(config["runtime"]["torch_num_threads"])
        train, val, global_val = prepare_data(config, artifacts)
        device = get_device()
        model = get_model(model_name=model_cfg["name"], num_classes=model_cfg["num_classes"],
                          pretrained=model_cfg["pretrained"])
        artifacts.manifest["model_parameters"] = sum(p.numel() for p in model.parameters())
        artifacts.json("run_manifest.json", artifacts.manifest)
        writer = SummaryWriter(log_dir=str(artifacts.runs))
        strategy = FedMedFedAvg(model, global_val, artifacts, writer, device)

        def make_client(cid):
            # Ray workers have their own torch/thread RNG state.
            torch.set_num_threads(config["runtime"]["torch_num_threads"])
            device_types = config["hardware"]["client_device_types"]
            return FedMedClient(
                cid, get_model(model_name=model_cfg["name"], num_classes=model_cfg["num_classes"],
                               pretrained=False),
                train[cid], val[cid], training["local_epochs"], training["learning_rate"],
                training["weight_decay"], model_cfg["num_classes"],
                config["evaluation"]["class_names"],
                device_type=device_types.get(str(cid), device_types.get(cid, "pc")),
                monitoring=config["monitoring"], seed=fed["seed"],
                deterministic=config["runtime"]["deterministic"])

        started = time.perf_counter()
        if config["runtime"]["backend"] == "sequential":
            clients = {cid: make_client(cid) for cid in range(fed["num_clients"])}
            run_sequential(strategy, clients, fed["num_rounds"])
        else:
            def client_fn(context):
                cid = int(context.node_config["partition-id"])
                if not 0 <= cid < fed["num_clients"]:
                    raise ValueError(f"Invalid partition-id: {cid}")
                return make_client(cid).to_client()

            runtime = config["runtime"]
            if runtime["client_gpus"] and not torch.cuda.is_available():
                raise ValueError("client_gpus > 0 but CUDA is unavailable")
            fl.simulation.start_simulation(
                client_fn=client_fn, num_clients=fed["num_clients"],
                config=fl.server.ServerConfig(num_rounds=fed["num_rounds"]),
                strategy=strategy,
                client_resources={"num_cpus": runtime["client_cpus"], "num_gpus": runtime["client_gpus"]},
                ray_init_args={"include_dashboard": False, "num_cpus": runtime["ray_num_cpus"]})
        training_seconds = time.perf_counter() - started
        if strategy.completed_rounds != fed["num_rounds"]:
            raise RuntimeError("Simulation ended before all rounds were recorded")
        final = final_test(model, device, artifacts, strategy.summary(), training_seconds)
        print(f"Completed {artifacts.run_id}: test accuracy={final['final_test_accuracy']:.4f}, "
              f"macro F1={final['final_test_f1_macro']:.4f}\nResults: {artifacts.results}")
        return final
    except BaseException as exc:
        artifacts.finish("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if writer is not None:
            writer.close()


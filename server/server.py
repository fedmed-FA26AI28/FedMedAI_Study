"""Launch a network Flower server using the same validation/checkpoint protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--address", default="0.0.0.0:8080")
    args = parser.parse_args()
    import flwr as fl
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from algorithms import get_strategy
    from experiments.artifacts import RunArtifacts, load_config, seed_everything
    from experiments.runner import prepare_data, final_test
    from models.cnn import get_device, get_model
    config = load_config(args.config)

    config["deployment_mode"] = "network"
    config["runtime"]["backend"] = "network"
    config["hardware"]["scenario"] = "network_deployment"
    seed_everything(config["federation"]["seed"], config["runtime"]["deterministic"])
    torch.set_num_threads(config["runtime"]["torch_num_threads"])
    artifacts = RunArtifacts(config, f"federated/{config['algorithm']['name']}/network")
    writer = SummaryWriter(str(artifacts.runs))
    try:
        _, _, val = prepare_data(config, artifacts)
        model = get_model(**{ "model_name": config["model"]["name"],
                            "num_classes": config["model"]["num_classes"]})
        device = get_device()
        model.to(device)
        artifacts.manifest["model_parameters"] = sum(p.numel() for p in model.parameters())
        artifacts.json("run_manifest.json", artifacts.manifest)
        strategy = get_strategy(config["algorithm"]["name"], model, val, artifacts, writer, device)
        started = time.perf_counter()
        fl.server.start_server(server_address=args.address,
            config=fl.server.ServerConfig(num_rounds=config["federation"]["num_rounds"]), strategy=strategy)
        if strategy.completed_rounds != config["federation"]["num_rounds"]:
            raise RuntimeError("Server ended before all rounds were recorded")
        final_test(model, device, artifacts, strategy.summary(), time.perf_counter() - started)
    except Exception as error:
        artifacts.finish("failed", error=str(error))
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()

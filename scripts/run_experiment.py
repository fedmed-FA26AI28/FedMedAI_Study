"""Unified entry point for baseline and algorithm/client-count/seed ablations."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms import STRATEGIES
from experiments.train_centralized import run_centralized
from experiments.simulation import run_federated_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/experiment.yaml"))
    parser.add_argument("--mode", choices=["centralized", "federated"], default="federated")
    parser.add_argument("--partition", choices=["iid", "dirichlet"])
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--algorithm", choices=sorted(
        name for name, status in STRATEGIES.items() if status == "implemented"))
    parser.add_argument("--num-clients", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--backend", choices=["flower", "sequential"])
    args = parser.parse_args()
    if args.mode == "centralized":
        if any(value is not None for value in (args.algorithm, args.num_clients, args.rounds, args.backend)):
            parser.error("For centralized runs, set epochs and partition client count in --config")
        run_centralized(args.config, args.seed, args.partition, args.alpha)
    else:
        run_federated_simulation(
            args.config, partition_type_override=args.partition, alpha_override=args.alpha,
            seed_override=args.seed, algorithm_override=args.algorithm,
            num_clients_override=args.num_clients, num_rounds_override=args.rounds,
            backend_override=args.backend)


if __name__ == "__main__":
    main()

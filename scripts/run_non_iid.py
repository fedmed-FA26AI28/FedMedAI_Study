"""Run Dirichlet ablations for alpha 0.1 / 0.3 / 1.0, or one positional alpha."""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedmedai.experiment import load_config, write_json
from fedmedai.simulation import run_federated_simulation


def run_all_non_iid_experiments(config_path=None, alphas=None, algorithm=None, seed=None, backend=None):
    config_path = config_path or str(Path(__file__).resolve().parents[1] / "configs/config.yaml")
    config = load_config(config_path)
    comparison_dir = Path(config["paths"]["results_dir"]) / "comparisons" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    comparison_dir.mkdir(parents=True, exist_ok=False)
    records = []
    fields = ("run_id", "algorithm", "alpha", "random_seed", "num_clients", "dataset", "model",
              "partition_sha256", "final_test_loss", "final_test_accuracy", "final_test_f1_macro",
              "final_test_precision_macro", "final_test_recall_macro", "best_round",
              "rounds_to_target", "training_wall_seconds", "total_model_payload_bytes", "results_dir")
    for alpha in alphas or [0.1, 0.3, 1.0]:
        result = run_federated_simulation(
            config_path, partition_type_override="dirichlet", alpha_override=alpha,
            algorithm_override=algorithm, seed_override=seed, backend_override=backend)
        records.append({key: result[key] for key in fields})
        # Save after every completed run, preserving partial matrices if a later run fails.
        write_json(comparison_dir / "alpha_comparison.json", records)
        with (comparison_dir / "alpha_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    print(f"Comparison: {comparison_dir}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("alpha", nargs="?", type=float)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/config.yaml"))
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--backend", choices=["flower", "sequential"])
    args = parser.parse_args()
    run_all_non_iid_experiments(
        args.config, [args.alpha] if args.alpha is not None else None,
        args.algorithm, args.seed, args.backend)


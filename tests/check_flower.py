"""Separate offline Flower/Ray integration check; never a research benchmark."""
import argparse
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default="fedavg",
                        choices=("fedavg", "coverage_calibrated", "calibrated_fedprox"))
    args = parser.parse_args()
    import json
    import yaml
    from test_experiments import SyntheticMedicalDataset
    from experiments.simulation import run_federated_simulation
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = {"federation": {"num_clients": 1, "num_rounds": 1, "partition_type": "iid", "seed": 2026},
                  "algorithm": {"name": args.algorithm, "proximal_mu": 1.0,
                                "head_mu": 2.5, "coverage_kappa": 7.0,
                                "prior_smoothing": 0.25, "logit_tau": 0.75},
                  "model": {"name": "cnn"}, "training": {"batch_size": 8},
                  "runtime": {"backend": "flower", "torch_num_threads": 1,
                              "ray_num_cpus": 1, "client_cpus": 1, "client_gpus": 0.0},
                  "monitoring": {"enabled": False},
                  "paths": {"results_dir": str(root / "results"),
                            "checkpoints_dir": str(root / "checkpoints"), "runs_dir": str(root / "runs")}}
        path = root / "experiment.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        data = SyntheticMedicalDataset(16)
        with patch("datasets.partition.load_medmnist_split", return_value=data), \
                patch("datasets.medmnist_code.load_medmnist_split", return_value=data):
            result = run_federated_simulation(path)
        assert result["test_samples"] == 16
        manifest = json.loads((Path(result["results_dir"]) / "run_manifest.json").read_text())
        assert manifest["status"] == "completed"
        if args.algorithm != "fedavg":
            records = [json.loads(line) for line in
                       (Path(result["results_dir"]) / "client_metrics.jsonl").read_text().splitlines()]
            fit = next(record for record in records if record["phase"] == "fit")
            assert fit["algorithm"] == args.algorithm
            assert fit["prior_smoothing"] == 0.25 and fit["coverage_kappa"] == 7.0
            assert fit["logit_tau"] == 0.75
            assert fit["head_mu"] == (2.5 if args.algorithm == "coverage_calibrated" else 0.0)
            assert fit["proximal_mu"] == (1.0 if args.algorithm == "calibrated_fedprox" else 0.0)
        print(f"PASS: actual Flower/Ray, {args.algorithm}, one synthetic CNN client, one round, 16 test samples")


if __name__ == "__main__":
    main()

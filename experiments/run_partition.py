"""Generate deterministic train/validation partitions without training a model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path


def main():
    from experiments.artifacts import load_config
    from datasets.partition import create_client_dataloaders, visualize_partition
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--clients", type=int, nargs="+", default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    parser.add_argument("--output", default="data/partitions")
    args = parser.parse_args()
    config = load_config(args.config)
    fed, data = config["federation"], config["dataset"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for count in args.clients or [fed["num_clients"]]:
        for alpha in args.alphas:
            _, _, metadata = create_client_dataloaders(dataset_name=data["name"],
                partition_type="dirichlet", num_clients=count, alpha=alpha, seed=fed["seed"],
                batch_size=config["training"]["batch_size"], image_size=data["image_size"],
                root=data["root"], download=data["download"], num_workers=data.get("num_workers", 0),
                min_train_samples=fed["min_train_samples"], min_val_samples=fed["min_val_samples"],
                max_train_samples=data["max_train_samples"], max_val_samples=data["max_val_samples"])
            name = f"partition_seed{fed['seed']}_alpha{alpha}_clients{count}"
            (output / (name + ".json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            visualize_partition(metadata, output / "plots" / (name + ".png"))
            print(name, metadata["partition_sha256"])


if __name__ == "__main__":
    main()

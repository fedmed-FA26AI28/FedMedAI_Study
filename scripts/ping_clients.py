"""Probe configured real client endpoints and export CSV."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import yaml
    from monitoring.network import ClientPingLogger
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/jetson.yaml")
    parser.add_argument("--method", choices=("tcp", "icmp"), default="tcp")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--output", default="results/network/clients_ping.csv")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    clients = config.get("client_registry", [])
    if not clients:
        parser.error("client_registry is empty; configure actual endpoints first")
    rows = ClientPingLogger(args.output).log_round(args.round, clients, args.method, args.timeout)
    print(json.dumps(rows, indent=2))
    if not all(row["reachable"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

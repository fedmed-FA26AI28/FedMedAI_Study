"""Prepare or execute SCP transfers to configured Jetson hosts."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def transfer_command(source, client):
    source = Path(source).resolve()
    if not source.exists():
        raise ValueError(f"Missing source: {source}")
    host, user, remote = client["host"], client["ssh_user"], client["data_dir"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
        raise ValueError("Invalid SSH hostname")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", user):
        raise ValueError("Invalid SSH user")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote) or ".." in remote.split("/"):
        raise ValueError("data_dir must be an absolute remote path without shell metacharacters")
    port = int(client.get("ssh_port", 22))
    if not 1 <= port <= 65535:
        raise ValueError("Invalid SSH port")
    return ["scp", "-r", "-P", str(port), str(source), f"{user}@{host}:{remote}"]


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/jetson.yaml")
    parser.add_argument("--source", default="data/partitions")
    parser.add_argument("--execute", action="store_true", help="Execute the printed SCP commands")
    args = parser.parse_args()
    clients = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig")).get("client_registry", [])
    if not clients:
        parser.error("client_registry is empty; configure real hosts and destination paths")
    commands = [transfer_command(args.source, client) for client in clients]
    print(json.dumps(commands, indent=2))
    if args.execute:
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

"""Exercise actual server/client CLIs over loopback using synthetic data only."""
import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(role, config, address):
    from test_experiments import SyntheticMedicalDataset
    data = SyntheticMedicalDataset(16)
    with patch("datasets.partition.load_medmnist_split", return_value=data), \
            patch("datasets.medmnist_code.load_medmnist_split", return_value=data):
        if role == "server":
            from server.server import main
            sys.argv = ["server", "--config", config, "--address", address]
        else:
            from client.client import main
            sys.argv = ["client", "--config", config, "--server-address", address, "--client-id", "0"]
        main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("server", "client"))
    parser.add_argument("--config")
    parser.add_argument("--address")
    args = parser.parse_args()
    if args.role:
        worker(args.role, args.config, args.address)
        return
    import yaml
    with tempfile.TemporaryDirectory() as directory, socket.socket() as probe:
        root = Path(directory)
        probe.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{probe.getsockname()[1]}"
        probe.close()
        config = {"federation": {"num_clients": 1, "num_rounds": 1, "partition_type": "iid", "seed": 2026},
                  "model": {"name": "cnn"}, "training": {"batch_size": 8},
                  "runtime": {"backend": "sequential", "torch_num_threads": 1},
                  "monitoring": {"enabled": False},
                  "paths": {"results_dir": str(root / "results"),
                            "checkpoints_dir": str(root / "checkpoints"), "runs_dir": str(root / "runs")}}
        path = root / "experiment.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        processes = []
        try:
            with (root / "server.log").open("w") as server_log, (root / "client.log").open("w") as client_log:
                for role, log in (("server", server_log), ("client", client_log)):
                    processes.append(subprocess.Popen([sys.executable, "-B", __file__, "--role", role,
                        "--config", str(path), "--address", address], stdout=log, stderr=subprocess.STDOUT))
                    if role == "server":
                        deadline = time.monotonic() + 60
                        host, port = address.rsplit(":", 1)
                        while True:
                            if processes[0].poll() is not None:
                                raise RuntimeError("Server exited before listening")
                            try:
                                with socket.create_connection((host, int(port)), timeout=0.5):
                                    break
                            except OSError:
                                if time.monotonic() >= deadline:
                                    raise TimeoutError("Server did not start listening")
                                time.sleep(0.2)

                for process in processes:
                    if process.wait(timeout=60):
                        raise RuntimeError("Network process failed")
            manifests = list((root / "results").rglob("run_manifest.json"))
            assert len(manifests) == 1
            assert json.loads(manifests[0].read_text())["status"] == "completed"
            result = json.loads((manifests[0].parent / "final_test_metrics.json").read_text())
            assert result["test_samples"] == 16
            print("PASS: real Flower server/client CLIs, loopback, one CNN client/round, final test")
        except Exception:
            for log in root.glob("*.log"):
                print(log.name, log.read_text(errors="replace")[-12000:])
            raise
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()

"""Execute the frozen FedAvg/FedProx comparison at three Dirichlet alphas."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_communication_study import wait_with_deadline


def worker(lock_path, output):
    import torch
    import yaml
    from fedmedai.experiment import write_json
    from fedmedai.non_iid_study import ALPHAS, ARMS, compare_cells, load_cell, make_config, read_lock, verify_source
    from fedmedai.simulation import run_federated_simulation

    lock = read_lock(lock_path)
    verify_source(lock)
    shutil.copy2(lock_path, output / "baseline.lock.json")
    shutil.copy2(ROOT / "research/non_iid_protocol.md", output / "protocol.md")
    shutil.copy2(Path(__file__), output / "runner_snapshot.py")
    write_json(output / "snapshot_hashes.json", {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
               for name in ("baseline.lock.json", "protocol.md", "runner_snapshot.py")})
    entries = []
    for alpha in ALPHAS:
        for arm in ARMS:
            verify_source(lock)
            config = make_config(lock, alpha, arm)
            path = output / f"alpha_{alpha}_{arm}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(f"START alpha={alpha} seed={lock['seed']} arm={arm}", flush=True)
            result = run_federated_simulation(str(path), output_subdir_override=f"{output.name}/alpha_{alpha}/{arm}")
            entries.append(result["results_dir"])
            write_json(output / "index.json", {"status": "running", "runs": entries})
    verify_source(lock)
    cells = [load_cell(path, lock) for path in entries]
    analysis = compare_cells(cells, lock)
    write_json(output / "analysis.json", analysis)
    baseline = next(cell for cell in cells if cell["arm"] == "fedavg" and cell["result"]["alpha"] == 0.3)
    reference = baseline
    reference_root = Path(reference["result"]["results_dir"])
    print("START exact baseline replay alpha=0.3 seed=2026", flush=True)
    replay = run_federated_simulation(str(output / "alpha_0.3_fedavg.yaml"),
                                     output_subdir_override=f"{output.name}/baseline_replay")
    baseline = load_cell(replay["results_dir"], lock)
    exact = {}
    for key in ("final_test_accuracy", "final_test_f1_macro", "final_test_loss", "test_evaluation",
                "best_round", "total_model_payload_bytes", "partition_sha256"):
        exact[key] = baseline["result"][key] == reference["result"][key]
    states = []
    for cell in (reference, baseline):
        path = Path(cell["manifest"]["artifact_paths"]["checkpoints"]) / "best_model.pt"
        states.append(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
    exact["checkpoint_tensors"] = set(states[0]) == set(states[1]) and all(
        torch.equal(states[0][key], states[1][key]) for key in states[0])
    for number in range(1, lock["baseline_config"]["federation"]["num_rounds"] + 1):
        old = json.loads((reference_root / f"rounds/round_{number:04d}.json").read_text())
        new = json.loads((Path(baseline["result"]["results_dir"]) / f"rounds/round_{number:04d}.json").read_text())
        fields = [key for key in old if key.startswith("global_val_") or key.endswith("model_bytes")]
        exact[f"round_{number}_metrics"] = all(old[key] == new[key] for key in fields)
    write_json(output / "baseline_replay.json", {"reference": str(reference_root.resolve()),
               "new_baseline": baseline["result"]["results_dir"], "checks": exact,
               "exact_match": all(exact.values())})
    if not all(exact.values()):
        raise RuntimeError("Frozen FedAvg baseline failed exact replay")
    verify_source(lock)
    write_json(output / "index.json", {"status": "completed", "runs": entries, "replay": replay["results_dir"]})
    print(json.dumps(analysis, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "research/non_iid_baseline.lock.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("Timeout must be positive")
    if args.worker:
        worker(args.lock, args.output)
        return
    output = args.output or ROOT / "results/non_iid_study" / datetime.now(timezone.utc).strftime("suite_%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker",
               "--lock", str(args.lock.resolve()), "--output", str(output.resolve())]
    print(f"Study directory: {output.resolve()}", flush=True)
    record = {"command": command, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": args.timeout_seconds, "status": "running"}
    with (output / "execution.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record["pid"] = process.pid
        (output / "execution.json").write_text(json.dumps(record, indent=2))
        try:
            code = wait_with_deadline(process, args.timeout_seconds)
            record.update(status="completed" if code == 0 else "failed", exit_code=code)
        except subprocess.TimeoutExpired:
            print("Calendar deadline reached; terminating study worker.", flush=True)
            if process.poll() is None:
                process.kill()
            process.wait()
            code = 124
            record.update(status="timeout", exit_code=process.returncode)
    record.update(supervisor_exit_code=code, finished_at_utc=datetime.now(timezone.utc).isoformat())
    (output / "execution.json").write_text(json.dumps(record, indent=2))
    print(f"Study {record['status']}; see {output / 'execution.log'}", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

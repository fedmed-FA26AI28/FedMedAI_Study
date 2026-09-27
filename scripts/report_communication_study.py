"""Report the current Dirichlet ResNet-18 communication study.

The legacy evidence helper remains for offline schema regression coverage only.
The CLI always uses the current full evidence gate and never reports the retired matrix.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fedmedai.study import assert_matched, load_run


def validate_evidence(suite, index, runs):
    """Bind the report's verification claims to this suite before writing output."""
    read = lambda name: json.loads((suite / name).read_text(encoding="utf-8"))
    recorded_hash = read("protocol_sha256.json")["sha256"]
    if hashlib.sha256((suite / "protocol.md").read_bytes()).hexdigest() != recorded_hash:
        raise ValueError("Protocol snapshot hash mismatch")
    replay, execution = read("reproducibility.json"), read("execution.json")
    required = {"final_test_accuracy", "final_test_f1_macro", "final_test_loss",
                "best_round", "total_model_payload_bytes", "test_evaluation", "checkpoint_tensors"}
    if (execution["status"] != "completed" or execution["exit_code"] != 0
            or replay["exact_match"] is not True or set(replay["checks"]) != required
            or any(value is not True for value in replay["checks"].values())):
        raise ValueError("Execution/reproducibility gate failed")
    if replay["original"] not in index["runs"] or replay["replay"] != index["replay"]:
        raise ValueError("Replay is not linked to this suite")
    original = runs[index["runs"].index(replay["original"])]
    repeated = load_run(replay["replay"])
    assert_matched(original, repeated)
    for key in required - {"checkpoint_tensors"}:
        if original["result"][key] != repeated["result"][key]:
            raise ValueError(f"Replay does not reproduce {key}")
    import torch
    states = []
    for run in (original, repeated):
        path = Path(run["manifest"]["artifact_paths"]["checkpoints"]) / "best_model.pt"
        states.append(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
    if set(states[0]) != set(states[1]) or any(not torch.equal(states[0][key], states[1][key]) for key in states[0]):
        raise ValueError("Replay checkpoint tensors differ")
    return execution



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    from report_non_iid_study import generate
    generate(args.suite)


if __name__ == "__main__":
    main()

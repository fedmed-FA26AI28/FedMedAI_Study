"""Communication-study entry point for the current Dirichlet-only ResNet-18 matrix.

The retired IID/BloodCNN matrix cannot be launched through this entry point.
The deadline helper is also imported by run_non_iid_study.py.
"""

import subprocess
import time


def wait_with_deadline(process, timeout_seconds):
    """Recheck calendar time after bounded waits, including after host resume."""
    deadline = time.time() + timeout_seconds
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)
        try:
            code = process.wait(timeout=min(30, remaining))
        except subprocess.TimeoutExpired:
            continue
        if time.time() > deadline:
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)
        return code



def main():
    from run_non_iid_study import main as run_current_study
    run_current_study()


if __name__ == "__main__":
    main()

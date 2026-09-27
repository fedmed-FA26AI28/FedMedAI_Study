"""Hold a temporary Windows system-idle request only while a recorded worker runs."""

import argparse
import ctypes
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Windows only")
    record = json.loads((args.suite / "execution.json").read_text())
    process = psutil.Process(record["pid"])
    if record["status"] != "running" or process.cmdline()[1:] != record["command"][1:]:
        raise ValueError("Recorded study worker is not running with the expected command")
    deadline = datetime.fromisoformat(record["started_at_utc"]).timestamp() + record["timeout_seconds"]
    state = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
    state.argtypes = [ctypes.c_uint32]
    state.restype = ctypes.c_uint32
    if not state(0x80000001):  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; never request display/away mode.
        raise ctypes.WinError(ctypes.get_last_error())
    output = args.suite / "idle_request.json"
    audit = {"worker_pid": process.pid, "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "status": "active", "scope": "system idle only, until worker exit or suite deadline"}
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Temporary system-idle request active for the recorded study worker.", flush=True)
    try:
        while time.time() < deadline:
            try:
                process.wait(timeout=min(30, max(0, deadline - time.time())))
                break
            except psutil.TimeoutExpired:
                continue
    finally:
        released = bool(state(0x80000000))  # ES_CONTINUOUS clears this thread's request.
        audit.update(status="released" if released else "release_failed",
                     finished_at_utc=datetime.now(timezone.utc).isoformat())
        output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"Temporary system-idle request: {audit['status']}.", flush=True)


if __name__ == "__main__":
    main()

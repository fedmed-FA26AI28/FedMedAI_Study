"""Actual TCP/ICMP reachability. Failed probes have null latency."""
import csv
import platform
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def ping(host, port=8080, method="tcp", timeout=2.0):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]*", host):
        raise ValueError("Invalid hostname or IP address")
    if timeout <= 0 or not 1 <= int(port) <= 65535:
        raise ValueError("timeout and port must be positive and valid")
    if method not in ("tcp", "icmp"):
        raise ValueError("method must be tcp or icmp")
    started = time.perf_counter()
    try:
        if method == "tcp":
            with socket.create_connection((host, int(port)), timeout=timeout):
                pass
            latency = (time.perf_counter() - started) * 1000
        else:
            windows = platform.system() == "Windows"
            command = ["ping", "-n" if windows else "-c", "1", host]
            process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            if process.returncode:
                raise OSError("ICMP probe failed")
            # Localized output may be unparsable; do not substitute process time for RTT.
            match = re.search(r"(?:time|temps|zeit)([=<])\s*([\d.]+)\s*ms", process.stdout, re.I)
            latency = float(match.group(2)) if match and match.group(1) == "=" else None
        return {"host": host, "port": port if method == "tcp" else None,
                "method": method, "reachable": True, "latency_ms": latency, "error": None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"host": host, "port": port if method == "tcp" else None,
                "method": method, "reachable": False, "latency_ms": None, "error": str(error)}


class ClientPingLogger:
    """Append explicitly requested probes; timestamps are UTC and round is caller supplied."""
    def __init__(self, path="results/network/clients_ping.csv"):
        self.path = Path(path)

    def log_round(self, round_number, clients, method="tcp", timeout=2.0):
        rows = [{"timestamp": datetime.now(timezone.utc).isoformat(), "round": round_number,
                 "client_id": client["id"], **ping(client["host"], client.get("port", 8080), method, timeout)}
                for client in clients]
        if rows:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            header = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                if header:
                    writer.writeheader()
                writer.writerows(rows)
        return rows

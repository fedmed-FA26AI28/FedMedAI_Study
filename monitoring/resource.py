"""Process/host sampling and optional numeric sysfs sensors on edge devices."""

import math
import platform
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class ResourceMonitor:
    """Sample one operation. Missing sensors stay null, never synthetic zeros.

    sensors maps canonical names (gpu_utilization_percent, temperature_c,
    power_watts) to {path, scale}. Choose a whole-board power rail for board
    energy. CPU/RAM and sensor values refer to the shared host, RSS to this
    process, CUDA memory to this process's PyTorch allocator.
    """

    def __init__(self, device, interval_seconds=0.2, sensors=None, enabled=True):
        if interval_seconds <= 0:
            raise ValueError("Resource sampling interval must be positive")
        self.device = device
        self.interval = interval_seconds
        self.sensors = sensors or {}
        self.enabled = enabled
        self.samples = []
        self.stop_event = threading.Event()
        self.process = psutil.Process()
        self.result = {}
        self.thread = None

    def _sensor(self, name):
        spec = self.sensors.get(name)
        if not spec:
            return None
        try:
            value = float(Path(spec["path"]).read_text().strip()) * float(spec.get("scale", 1))
            return value if math.isfinite(value) else None
        except (OSError, ValueError, KeyError):
            return None

    def _sample(self):
        self.samples.append({
            "time": time.perf_counter(),
            "cpu_utilization_percent": psutil.cpu_percent(interval=None),
            "process_rss_bytes": self.process.memory_info().rss,
            "system_ram_used_bytes": psutil.virtual_memory().used,
            **{name: self._sensor(name) for name in
               ("gpu_utilization_percent", "temperature_c", "power_watts")},
        })

    def _poll(self):
        psutil.cpu_percent(interval=None)  # Prime per-thread counters.
        while not self.stop_event.wait(self.interval):
            self._sample()

    def __enter__(self):
        synchronize(self.device)
        if self.enabled and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        if self.enabled:
            self._sample()
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        synchronize(self.device)
        self.duration = time.perf_counter() - self.started
        self.stop_event.set()
        if self.thread:
            self.thread.join()
            self._sample()
        self.result = self._summarize()

    def _summarize(self):
        result = {"duration_seconds": self.duration, "resource_sample_count": len(self.samples),
                  "hostname": platform.node(), "device": str(self.device),
                  "device_name": (torch.cuda.get_device_name(self.device)
                                  if self.device.type == "cuda" else platform.processor()),
                  "resource_scope": "host_cpu_ram_sensors; process_rss_cuda_allocator"}
        for source, output, reducer in [
            ("cpu_utilization_percent", "cpu_utilization_percent_mean", np.mean),
            ("process_rss_bytes", "process_rss_bytes_peak", max),
            ("system_ram_used_bytes", "system_ram_used_bytes_peak", max),
            ("gpu_utilization_percent", "gpu_utilization_percent_mean", np.mean),
            ("temperature_c", "temperature_c_max", max),
            ("power_watts", "power_watts_mean", np.mean),
        ]:
            samples = self.samples[1:] if source == "cpu_utilization_percent" else self.samples
            values = [s[source] for s in samples if s[source] is not None]
            result[output] = float(reducer(values)) if values else None
        result["gpu_memory_allocated_bytes_peak"] = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.enabled and self.device.type == "cuda" else None)
        # Integrate only adjacent valid samples; do not bridge missing measurements.
        energy, covered = 0.0, 0.0
        for a, b in zip(self.samples, self.samples[1:]):
            if a["power_watts"] is not None and b["power_watts"] is not None:
                dt = b["time"] - a["time"]
                energy += (a["power_watts"] + b["power_watts"]) * 0.5 * dt
                covered += dt
        result["energy_joules_observed"] = energy if covered else None
        result["energy_observed_seconds"] = covered
        result["energy_scope"] = "configured_power_sensor_operation_window"
        return result

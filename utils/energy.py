"""GPU energy and memory accounting for cost reporting (frugality tables).

Energy comes from NVML's cumulative energy counter (mJ since driver load) when
the GPU supports it, else from power sampling.  NVML measures the whole GPU, so
the numbers are only attributable to one run when that run is alone on the GPU:
``gpu_processes`` records how many compute processes were on the device.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

import torch

try:
    import pynvml

    pynvml.nvmlInit()
except Exception:  # NVML missing or no driver: energy is reported as None
    pynvml = None


def _nvml_index(device: torch.device) -> int:
    index = device.index or 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return int(visible.split(",")[index])
    return index


@contextmanager
def measure(device: torch.device):
    """Yield a dict filled on exit: seconds, energy_j, peak_mem_mb, gpu_processes."""
    out: dict = {}
    cuda = device.type == "cuda" and torch.cuda.is_available()
    handle = None
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    if cuda and pynvml is not None:
        handle = pynvml.nvmlDeviceGetHandleByIndex(_nvml_index(device))
    counter = None
    samples: list[float] = []
    stop = threading.Event()
    if handle is not None:
        try:
            counter = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        except Exception:

            def sample():
                while not stop.is_set():
                    samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                    stop.wait(0.1)

            threading.Thread(target=sample, daemon=True).start()
        out["gpu_processes"] = len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
    started = time.perf_counter()
    try:
        yield out
    finally:
        if cuda:
            torch.cuda.synchronize(device)
        out["seconds"] = time.perf_counter() - started
        stop.set()
        if handle is not None and counter is not None:
            end = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            out["energy_j"] = (end - counter) / 1000.0
        elif samples:
            out["energy_j"] = sum(samples) / len(samples) * out["seconds"]
        else:
            out["energy_j"] = None
        out["peak_mem_mb"] = (
            torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None
        )


def start_carbon(run_dir, device: torch.device, enabled: bool = True):
    """Whole-run CodeCarbon tracker (CPU + GPU + RAM energy, kg CO2eq).

    Offline, with Spain's grid carbon intensity. ``tracking_mode="process"``
    attributes CPU/RAM to this process; GPU energy is still device-wide, so
    for cost tables run one job per GPU.  Returns None when disabled or
    unavailable.
    """
    if not enabled:
        return None
    try:
        from codecarbon import OfflineEmissionsTracker

        tracker = OfflineEmissionsTracker(
            country_iso_code="ESP",
            output_dir=str(run_dir),
            log_level="error",
            tracking_mode="process",
            gpu_ids=[_nvml_index(device)] if device.type == "cuda" else None,
        )
        tracker.start()
        return tracker
    except Exception:
        return None


def stop_carbon(tracker) -> dict | None:
    if tracker is None:
        return None
    try:
        emissions = tracker.stop()
        data = tracker.final_emissions_data
        return {
            "energy_kwh": data.energy_consumed,
            "cpu_kwh": data.cpu_energy,
            "gpu_kwh": data.gpu_energy,
            "ram_kwh": data.ram_energy,
            "duration_s": data.duration,
            "emissions_kg_co2eq": emissions,
        }
    except Exception:
        return None

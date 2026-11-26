from __future__ import annotations

import gc
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

MIB = 1024**2


def _read_gpu_used_mib() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.used) / MIB
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
        values = [float(line.strip()) for line in out.splitlines() if line.strip()]
        return max(values) if values else None
    except Exception:
        return None


def get_device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": os.sys.version.split()[0],
        "platform": os.sys.platform,
        "gpu_used_mib_start": _read_gpu_used_mib(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_mib"] = round(props.total_memory / MIB, 1)
            info["cuda_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        info["torch_error"] = str(exc)

    for module, key in [
        ("transformers", "transformers"),
        ("vllm", "vllm"),
        ("llama_cpp", "llama_cpp"),
        ("awq", "autoawq"),
        ("auto_gptq", "auto_gptq"),
    ]:
        try:
            mod = __import__(module)
            info[key] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[key] = None
    return info


class MemoryMonitor:
    """Sample process RSS and GPU memory while a workload runs."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, float | None]] = []

    def __enter__(self) -> "MemoryMonitor":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def sample(self) -> None:
        try:
            import psutil

            rss = float(psutil.Process().memory_info().rss) / MIB
        except Exception:
            rss = None
        self.samples.append({"gpu_used_mib": _read_gpu_used_mib(), "rss_mib": rss})

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.sample()
        gc.collect()

    def summary(self) -> dict[str, float | None]:
        gpus = [s["gpu_used_mib"] for s in self.samples if s["gpu_used_mib"] is not None]
        rss = [s["rss_mib"] for s in self.samples if s["rss_mib"] is not None]
        torch_peak: float | None = None
        try:
            import torch

            if torch.cuda.is_available():
                torch_peak = float(torch.cuda.max_memory_allocated()) / MIB
        except Exception:
            pass
        return {
            "peak_gpu_mib": round(max(gpus), 1) if gpus else None,
            "mean_gpu_mib": round(sum(gpus) / len(gpus), 1) if gpus else None,
            "peak_rss_mib": round(max(rss), 1) if rss else None,
            "torch_peak_allocated_mib": round(torch_peak, 1) if torch_peak is not None else None,
        }


def env_report(path: str | Path) -> dict[str, Any]:
    info = get_device_info()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(__import__("json").dumps(info, indent=2), encoding="utf-8")
    return info

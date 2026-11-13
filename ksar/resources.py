"""
Resource Manager: reads free RAM / disk (standard library only) and estimates what a model will need,
so the Chip Manager does not load something that would push the machine into swap.

Note: on an integrated GPU (e.g. Radeon 780M) the "VRAM" is just system RAM, so only RAM is tracked.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SystemResources:
    ram_total_mb: int = 0
    ram_free_mb: int = 0
    disk_free_gb: float = 0.0

    @property
    def ram_known(self) -> bool:
        return self.ram_total_mb > 0


def _windows_ram() -> Optional[tuple]:
    """GlobalMemoryStatusEx via ctypes: works on every Windows version (wmic is removed from Windows 11 24H2+)."""
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return st.ullTotalPhys // 2**20, st.ullAvailPhys // 2**20


def _linux_ram() -> Optional[tuple]:
    total = free = 0
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) // 1024
            elif line.startswith("MemAvailable:"):
                free = int(line.split()[1]) // 1024
    return (total, free) if total else None


def estimate_model_ram_mb(model_id: str, installed_bytes: int = 0, num_ctx: int = 8192) -> int:
    """RAM a model needs while loaded. Prefers the real file size from Ollama; else guesses from '32b' in the name."""
    kv_cache = 100 + num_ctx // 8                      # rough MB for the context window of a 7-32B model
    if installed_bytes:
        return int(installed_bytes / 2**20 * 1.1) + kv_cache
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model_id.lower().replace("-", ":").replace("_", ":").replace(":", " "))
    if m:
        return int(float(m.group(1)) * 620) + 500 + kv_cache      # ~Q4_K_M: ~0.6 GB per billion parameters
    return 6000


class ResourceManager:
    RESERVE_MB = 3072          # always leave this much for Windows and your other programs

    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path

    def get_resources(self) -> SystemResources:
        res = SystemResources()
        try:
            res.disk_free_gb = shutil.disk_usage(self.workspace).free // (2**30)
        except Exception:
            pass
        try:
            got = _windows_ram() if os.name == "nt" else _linux_ram() if os.path.exists("/proc/meminfo") else None
            if got:
                res.ram_total_mb, res.ram_free_mb = got
        except Exception:
            pass
        return res

    def can_load_chip(self, estimated_ram_mb: int) -> bool:
        return self.explain(estimated_ram_mb)[0]

    def explain(self, estimated_ram_mb: int) -> tuple:
        """(allowed, human-readable reason)."""
        res = self.get_resources()
        if not res.ram_known:
            return False, "cannot read free RAM on this system, so the advanced chip stays off for safety"
        need = estimated_ram_mb + self.RESERVE_MB
        if res.ram_free_mb >= need:
            return True, f"{res.ram_free_mb} MB free, needs about {need} MB incl. reserve"
        return False, f"only {res.ram_free_mb} MB free, needs about {need} MB incl. {self.RESERVE_MB} MB reserve"

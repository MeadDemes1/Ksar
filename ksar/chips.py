"""
Chip Manager: orchestrates multiple LLM models (chips), manages their lifecycle in Ollama and picks the
right one for each request based on task complexity, the performance priority and free RAM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .resources import estimate_model_ram_mb

RETRY_AFTER_FAILURE_S = 300        # do not hammer Ollama with a model that failed to load


class ChipRole(Enum):
    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    ADVANCED = "advanced"


class ChipState(Enum):
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ACTIVE = "active"


@dataclass
class Chip:
    name: str
    role: ChipRole
    model_id: str                    # Ollama model name
    state: ChipState = ChipState.UNLOADED
    last_used: float = 0.0
    estimated_ram_mb: int = 6000
    failed_at: float = 0.0
    last_error: str = ""


# retention -> Ollama keep_alive for (standard, advanced, light)
KEEP_ALIVE = {
    "persistent": (-1, -1, "30m"),
    "balanced": ("30m", "10m", "5m"),
    "aggressive": ("5m", 0, 0),
}
_SLOT = {ChipRole.STANDARD: 0, ChipRole.ADVANCED: 1, ChipRole.LIGHTWEIGHT: 2}


class ChipManager:
    def __init__(self, app, resources):
        self.app = app
        self.resources = resources
        self.chips: dict = {}
        self.active_chip: Optional[Chip] = None
        self._load_config()

    # ------------------------------------------------------------------ setup
    def _load_config(self) -> None:
        s = self.app.settings
        old = self.chips
        self.chips = {}
        std = getattr(s, "standard_chip", "") or "qwen3:8b"
        light = getattr(s, "light_chip", "")
        for name, role, model in (("standard", ChipRole.STANDARD, std),
                                  ("advanced", ChipRole.ADVANCED, getattr(s, "advanced_chip", "")),
                                  ("light", ChipRole.LIGHTWEIGHT, "" if light == std else light)):
            if not model:
                continue
            prev = old.get(name)
            if prev and prev.model_id == model:          # same model: keep what we know about it
                self.chips[name] = prev
                continue
            self.chips[name] = Chip(name, role, model, ChipState.UNLOADED,
                                    estimated_ram_mb=estimate_model_ram_mb(model, num_ctx=getattr(s, "num_ctx", 8192)))
        self.active_chip = None

    def _llm_call(self, name: str, *args, **kw):
        fn = getattr(self.app.llm, name, None)
        return fn(*args, **kw) if callable(fn) else None

    def refresh(self) -> None:
        """Sync chip states with what Ollama really has loaded (it unloads idle models on its own)."""
        try:
            running = self._llm_call("running_models")
            installed = self._llm_call("installed_models") or {}
        except Exception:
            return
        if running is None:
            return
        for chip in self.chips.values():
            chip.state = ChipState.LOADED if chip.model_id in running else ChipState.UNLOADED
            size = running.get(chip.model_id) or installed.get(chip.model_id)
            if size:
                chip.estimated_ram_mb = estimate_model_ram_mb(chip.model_id, size,
                                                              getattr(self.app.settings, "num_ctx", 8192))

    # ------------------------------------------------------------------ routing
    def keep_alive_for(self, chip: Chip):
        retention = getattr(self.app.settings, "chips_retention", "balanced")
        return KEEP_ALIVE.get(retention, KEEP_ALIVE["balanced"])[_SLOT[chip.role]]

    def _advanced_allowed(self, complexity: str) -> bool:
        s = self.app.settings
        if "advanced" not in self.chips or not getattr(s, "auto_escalation", True):
            return False
        priority = getattr(s, "performance_priority", "balanced")
        if priority == "speed":
            return False
        if priority == "quality":
            return True
        return complexity == "complex"

    def get_chip_for_task(self, task_complexity: str = "normal") -> Chip:
        std = self.chips.get("standard")
        if not std:
            raise RuntimeError("No standard chip configured (/config set chips.standard <model>)")
        if self._advanced_allowed(task_complexity):
            adv = self.chips["advanced"]
            if adv.state == ChipState.UNLOADED and time.time() - adv.failed_at > RETRY_AFTER_FAILURE_S:
                self.refresh()
                if adv.state == ChipState.UNLOADED:
                    ok, why = self.resources.explain(adv.estimated_ram_mb)
                    if ok:
                        self._load_chip(adv)
                    else:
                        adv.last_error = why
            if adv.state != ChipState.UNLOADED:
                self.active_chip = adv
                return adv
        self.active_chip = std
        return std

    # ------------------------------------------------------------------ light slot (idle moments)
    def light_for_idle(self):
        """(chip or None, reason). The light chip is only ever used for background work, never for your chat."""
        chip = self.chips.get("light")
        if chip is None:
            return None, "no light chip configured (/config set chips.light qwen3:1.7b)"
        if time.time() - chip.failed_at < RETRY_AFTER_FAILURE_S and chip.failed_at:
            return None, f"light chip failed recently: {chip.last_error}"
        if chip.state == ChipState.UNLOADED:
            self.refresh()
            if chip.state == ChipState.UNLOADED:
                ok, why = self.resources.explain(chip.estimated_ram_mb)
                if not ok:
                    return None, why
        return chip, "ok"

    def ensure_loaded(self, chip: Chip) -> bool:
        """Loads a chip if Ollama does not have it yet (quietly: this runs in the background)."""
        if chip.state != ChipState.UNLOADED:
            return True
        return self._load_chip(chip, quiet=True)

    # ------------------------------------------------------------------ lifecycle
    def _load_chip(self, chip: Chip, quiet: bool = False) -> bool:
        try:
            if callable(getattr(self.app.llm, "load", None)):
                self.app.llm.load(chip.model_id, keep_alive=self.keep_alive_for(chip))
            else:                                         # minimal LLM objects (tests, custom back-ends)
                self.app.llm.chat([{"role": "user", "content": "ok"}], model=chip.model_id,
                                  keep_alive=self.keep_alive_for(chip), json_mode=False)
            chip.state = ChipState.LOADED
            chip.last_used = time.time()
            chip.last_error = ""
            return True
        except Exception as e:
            chip.state = ChipState.UNLOADED
            chip.failed_at = time.time()
            chip.last_error = str(e)
            if not quiet:
                print(f"  (chip '{chip.name}' could not be loaded: {e})")
            return False

    def load_chip(self, chip_name: str) -> str:
        chip = self.chips.get(chip_name)
        if not chip:
            return f"no chip '{chip_name}'"
        ok, why = self.resources.explain(chip.estimated_ram_mb)
        if not ok:
            return f"not loaded: {why}"
        return f"{chip_name}: {chip.state.value}" if self._load_chip(chip) else f"{chip_name}: failed - {chip.last_error}"

    def unload_chip(self, chip_name: str) -> str:
        chip = self.chips.get(chip_name)
        if not chip:
            return f"no chip '{chip_name}'"
        try:
            if callable(getattr(self.app.llm, "unload", None)):
                self.app.llm.unload(chip.model_id)
            else:
                self.app.llm.chat([{"role": "user", "content": "ok"}], model=chip.model_id, keep_alive=0)
            chip.state = ChipState.UNLOADED
            return f"{chip_name}: unloaded"
        except Exception as e:
            return f"{chip_name}: could not unload ({e})"

    def release_chip(self, chip: Chip) -> None:
        """Called after each turn/task; Ollama applies keep_alive itself, we only record usage."""
        chip.last_used = time.time()
        chip.state = ChipState.LOADED if chip.state == ChipState.UNLOADED else chip.state

    # ------------------------------------------------------------------ reporting
    def status_rows(self) -> list:
        self.refresh()
        try:
            installed = self._llm_call("installed_models")
        except Exception:
            installed = None
        rows = []
        for name, chip in self.chips.items():
            rows.append({"name": name, "model": chip.model_id, "state": chip.state.value,
                         "est_ram_mb": chip.estimated_ram_mb, "keep_alive": self.keep_alive_for(chip),
                         "installed": None if installed is None else chip.model_id in installed,
                         "note": chip.last_error})
        return rows

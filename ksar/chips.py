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


# User-facing names (internal identifiers stay: standard / advanced / light)
DISPLAY = {"standard": "Basic", "advanced": "Advanced", "light": "Idle/Light"}
ALIASES = {"basic": "standard", "standard": "standard", "advanced": "advanced", "auto": "auto",
           "light": "light", "idle": "light"}


def display_name(chip_name: str) -> str:
    return DISPLAY.get(chip_name, chip_name)


@dataclass
class ChipChoice:
    """Which foreground chip answers one message, and why. Small and structured so routing can grow later."""
    chip: Optional["Chip"]                 # None when a manual choice could not be honoured
    mode: str = "auto"                     # "auto" | "manual"
    reason: str = ""                       # why this chip
    requested: str = ""                    # internal name the user asked for ("" = automatic)
    note: str = ""                         # e.g. why Advanced was wanted but Basic is used (auto mode only)
    error: str = ""                        # manual choice failed: the message is NOT answered by another chip
    complexity: str = "normal"

    @property
    def name(self) -> str:
        return self.chip.name if self.chip else (self.requested or "?")


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
        self.override: Optional[str] = None          # "standard" | "advanced": applies to the NEXT message only
        self.last_choice: Optional[ChipChoice] = None
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

    def _auto(self, complexity: str, signals: str = "") -> ChipChoice:
        std = self.chips.get("standard")
        if not std:
            raise RuntimeError("No standard chip configured (/config set chips.standard <model>)")
        s = self.app.settings
        why = signals or ("complex request" if complexity == "complex" else "ordinary request")
        priority = getattr(s, "performance_priority", "balanced")
        basic = lambda reason, note="": ChipChoice(std, "auto", reason, "", note, "", complexity)
        if complexity != "complex" and priority != "quality":
            return basic(f"{why} -> Basic")
        if "advanced" not in self.chips:
            return basic(f"{why} -> Basic", "complex request, but no Advanced chip is configured"
                         if complexity == "complex" else "")
        if not getattr(s, "auto_escalation", True):
            return basic(f"{why} -> Basic", "automatic escalation is off (chips.auto_escalation)")
        if priority == "speed":
            return basic(f"{why} -> Basic", "performance priority is 'speed'")
        adv = self.chips["advanced"]
        if adv.state == ChipState.UNLOADED and time.time() - adv.failed_at > RETRY_AFTER_FAILURE_S:
            self.refresh()
            if adv.state == ChipState.UNLOADED:
                ok, reason = self.resources.explain(adv.estimated_ram_mb)
                if ok:
                    self._load_chip(adv)
                else:
                    adv.last_error = reason
        if adv.state != ChipState.UNLOADED:
            return ChipChoice(adv, "auto", f"{why} -> Advanced" if complexity == "complex"
                              else f"priority 'quality' -> Advanced", "", "", "", complexity)
        return basic(f"{why} -> Basic", f"Advanced not available: {adv.last_error or 'could not be loaded'}")

    def get_chip_for_task(self, task_complexity: str = "normal") -> Chip:
        """Automatic choice (used by background tasks and kept for compatibility)."""
        choice = self._auto(task_complexity)
        self.active_chip = choice.chip
        return choice.chip

    # ------------------------------------------------------------------ manual choice (/use)
    def set_override(self, word: str):
        """Validates a /use request NOW, so a broken override is never created. Returns (ok, message)."""
        name = ALIASES.get((word or "").strip().lower())
        if name is None:
            return False, "unknown chip. Use: /use basic | advanced | auto | status"
        if name == "auto":
            self.override = None
            return True, "automatic routing (no override)"
        if name == "light":
            return False, "the Idle/Light chip is reserved for background work and cannot answer chat"
        err = self._manual_problem(name)
        if err:
            return False, err
        self.override = name
        return True, f"the next message will be answered by {display_name(name)} ({self.chips[name].model_id})"

    def _manual_problem(self, name: str, loading: bool = False) -> str:
        """'' when the chip can be used right now; otherwise the reason. Never bypasses the RAM check."""
        chip = self.chips.get(name)
        if chip is None:
            return ("no Advanced chip is configured: /config set chips.advanced <model>" if name == "advanced"
                    else "no Basic chip is configured: /config set chips.standard <model>")
        if name == "standard":
            return ""
        try:
            installed = self._llm_call("installed_models")
        except Exception:
            installed = None
        if installed is not None and chip.model_id not in installed:
            return f"{display_name(name)} model '{chip.model_id}' is not installed. Run: ollama pull {chip.model_id}"
        if chip.state == ChipState.UNLOADED:
            self.refresh()
        if chip.state == ChipState.UNLOADED:
            ok, why = self.resources.explain(chip.estimated_ram_mb)
            if not ok:
                return f"{display_name(name)} cannot be loaded safely: {why}"
        return ""

    def choose(self, complexity: str = "normal", signals: str = "", consume: bool = True) -> ChipChoice:
        """The chip for ONE user message. A pending override wins over complexity and performance_priority, is used
        up by this call, and is re-checked (RAM, installed) at this moment. If it cannot be honoured the choice
        carries an error: the message is not silently answered by a different chip."""
        name, self.override = (self.override, None) if consume else (self.override, self.override)
        if name is None:
            choice = self._auto(complexity, signals)
        else:
            err = self._manual_problem(name)
            chip = self.chips.get(name)
            if not err and chip.state == ChipState.UNLOADED:
                if not self._load_chip(chip, quiet=True):
                    err = f"{display_name(name)} failed to load: {chip.last_error}"
            if err:
                choice = ChipChoice(None, "manual", f"you asked for {display_name(name)}", name, "", err, complexity)
            else:
                choice = ChipChoice(chip, "manual", f"you asked for {display_name(name)}", name, "", "", complexity)
        self.last_choice = choice
        if choice.chip:
            self.active_chip = choice.chip
        return choice

    def override_status(self) -> str:
        if self.override:
            return f"manual: the next message goes to {display_name(self.override)} ({self.chips[self.override].model_id}), then back to auto"
        return "auto: complexity and performance_priority choose the chip"

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

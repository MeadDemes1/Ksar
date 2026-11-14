"""Settings and on-disk layout. Everything Ksar owns lives under one data folder."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Settings:
    standard_chip: str = "qwen3:8b"
    advanced_chip: str = ""
    light_chip: str = ""              # small model for idle moments, e.g. qwen3:1.7b ("" = off)
    performance_priority: str = "balanced"  # speed, balanced, quality
    auto_escalation: bool = True
    chips_retention: str = "balanced"  # aggressive, balanced, persistent
    model: str = "qwen3:8b"           # legacy; standard_chip is the real setting
    ollama_url: str = "http://localhost:11434"
    num_ctx: int = 8192
    think: bool = False               # Qwen3 reasoning: on = smarter but slower; /think on|off
    show_thinking: bool = True        # show the thoughts live (dim) when think is on
    temperature: float = 0.3
    request_timeout: int = 900
    max_agent_steps: int = 6          # tool calls allowed per user message
    recent_messages: int = 16         # chat history sent to the model (trimmed in steps, see Conversation.window)
    history_chars: int = 12000        # ...or fewer messages when they are long
    idle_enabled: bool = True         # background work while you are not talking to Ksar
    idle_after_s: int = 20            # seconds of quiet before idle work starts
    warm_on_start: bool = True        # load the standard model in the background at start-up
    recall_chars: int = 2000          # memory text injected, only when a message needs memory
    consolidate_every: int = 8        # exchanges between memory updates (also on /exit and "remember")
    split_fact_threshold: int = 25    # a note with more facts gets branched
    max_fact_chars: int = 300


class Paths:
    def __init__(self, base: Path):
        self.base = Path(base).resolve()
        self.memory = self.base / "memory"
        self.tasks = self.base / "tasks"
        self.workspace = self.base / "workspace"   # the ONLY folder Ksar may write by default
        self.logs = self.base / "logs"
        self.config = self.base / "config"

    def ensure(self) -> "Paths":
        for p in (self.base, self.memory, self.tasks, self.workspace, self.logs, self.config):
            p.mkdir(parents=True, exist_ok=True)
        return self


def default_home() -> Path:
    env = os.environ.get("KSAR_HOME")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "ksar_data"


def load_settings(paths: Paths) -> Settings:
    f = paths.config / "settings.json"
    s = Settings()
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            known = {x.name for x in fields(Settings)}
            for k, v in data.items():
                if k in known:
                    setattr(s, k, v)
            if "standard_chip" not in data and data.get("model"):
                s.standard_chip = data["model"]        # settings.json written by v0.1/v0.2: keep their model
            if any(x.name not in data for x in fields(Settings)):
                save_settings(paths, s)                # add the new keys so the file shows every option
        except Exception:
            pass
    else:
        save_settings(paths, s)
    return s


def save_settings(paths: Paths, s: Settings) -> None:
    """Atomic write, so a crash can never leave a half-written settings.json."""
    f = paths.config / "settings.json"
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(s), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, f)

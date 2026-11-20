"""Wires the modules together. Each part only knows the interfaces of the others."""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .agent import Agent, Conversation
from .chips import ChipManager
from .idle import IdleWorker
from .memory.jobs import JobQueue
from .timeinfo import TimeService
from .config import Paths, default_home, load_settings, save_settings
from .llm import OllamaLLM
from .memory import Index, MemoryStore
from .memory.keeper import Keeper
from .memory.store import today
from .resources import ResourceManager
from .tasks import TaskManager
from .tools import Permissions, Session, ToolContext, ToolManager, register_default_tools


@dataclass
class Names:
    name: str = "Ksar"
    user: str = "boss"


class Ksar:
    def __init__(self, home: Optional[Path] = None, llm=None, approver: Optional[Callable] = None):
        self.paths = Paths(home or default_home()).ensure()
        self.settings = load_settings(self.paths)
        self.names = self._load_names()
        self.llm = llm or OllamaLLM(self.settings)
        self.memory = MemoryStore(self.paths.memory, max_fact_chars=self.settings.max_fact_chars)
        self.index = Index(self.memory)
        self.perms = Permissions(self.paths)
        self.session = Session(src=f"chat:{today()}")
        ctx = ToolContext(self.settings, self.paths, self.perms, self.memory, self.index, self.session)
        self.tools = ToolManager(ctx, self.paths.logs / "audit.jsonl", approver)
        register_default_tools(self.tools)
        self.keeper = Keeper(self.memory, self.index, self.llm)
        self.tasks = TaskManager(self.paths.tasks)
        self.resources = ResourceManager(self.paths.workspace)
        self.chips = ChipManager(self, self.resources)       # must exist before the Agent runs
        self.agent = Agent(self)
        # app.lock only serialises foreground entry points (console commands). The background worker never takes it and
        # nothing slow (model calls, memory work) runs under it; memory has its own short store-level lock.
        self.lock = threading.RLock()
        self.time = TimeService(self.settings.timezone, self.paths.config / "session_state.json",
                                long_gap_hours=self.settings.time_long_gap_hours)
        self.jobs = JobQueue(self.paths.memory / ".jobs" / "queue.json")
        self.idle = IdleWorker(self)             # started by the console; see idle.py
        self.ensure_core()
        self.catch_up()

    def save_settings(self) -> None:
        save_settings(self.paths, self.settings)

    # -- journal watermark: which conversation turns have been through memory extraction
    def _watermark(self) -> dict:
        wm = self.jobs.get_meta("watermark")
        return dict(wm) if isinstance(wm, dict) else None

    def mark_processed(self, srcs) -> None:
        wm = self._watermark() or {}
        for src in srcs:
            m = re.match(r"^chat:(\d{4}-\d{2}-\d{2})#(\d+)", str(src))
            if m:
                wm[m.group(1)] = max(int(m.group(2)), int(wm.get(m.group(1), 0)))
        self.jobs.set_meta("watermark", wm)

    def mark_processed_through_now(self) -> None:
        """Everything in today's journal so far counts as handled (used after a synchronous memory save)."""
        last = self.memory.last_turn_src()
        if last:
            self.mark_processed([last])

    def unprocessed_turns(self, days: int = 3) -> list:
        wm = self._watermark() or {}
        out = []
        for i in range(days - 1, -1, -1):
            day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            out += [t for t in self.memory.read_journal(day) if t["n"] > int(wm.get(day, 0))]
        return out

    def catch_up(self) -> None:
        """First start of this version: older journal turns count as handled (they went through the old process).
        Later starts: turns that never reached memory (crash, queue lost) are queued again."""
        try:
            if self._watermark() is None:
                wm = {}
                for i in range(3):
                    day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                    n = len(self.memory.read_journal(day))
                    if n:
                        wm[day] = n
                self.jobs.set_meta("watermark", wm)
            elif self.settings.memory_async and self.unprocessed_turns():
                self.jobs.add("extract", {"after_watermark": True}, priority=7, dedupe_key="extract:catch-up")
        except Exception:
            pass

    # -- identity
    def _names_file(self) -> Path:
        return self.paths.config / "profile.json"

    def _load_names(self) -> Names:
        f = self._names_file()
        if f.exists():
            try:
                return Names(**json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        return Names()

    def has_profile(self) -> bool:
        return self._names_file().exists()

    def save_names(self, name: str, user: str) -> None:
        self.names = Names(name.strip()[:30] or "Ksar", user.strip()[:30] or "boss")
        self._names_file().write_text(json.dumps(self.names.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        core = self.memory.get("core")
        if core:                      # keep the core note in sync with the chosen names
            for f in list(core.facts):
                if f.text.startswith(("My name:", "Owner:")):
                    self.memory.retract_fact("core", f.fid, "renamed")
            self.memory.add_fact("core", f"My name: {self.names.name}", "stated", "setup")
            self.memory.add_fact("core", f"Owner: {self.names.user}", "stated", "setup")

    def ensure_core(self) -> None:
        """Creates the root of the memory tree on first run (deterministic, no model needed)."""
        if self.memory.exists("core"):
            return
        self.memory.create("core", "Core", "core", "Central note: who the owner is, stable preferences, rules and main projects, with links to the branches.")
        for text in (f"My name: {self.names.name}",
                     f"Owner: {self.names.user}",
                     "I am a local assistant: my memory is linked Markdown notes; every fact keeps its source and history",
                     "Style: warm, concise, honest; I say when I don't know"):
            self.memory.add_fact("core", text, "stated", "setup", allow_supersede=False)

    def swap_session(self, session: Session) -> Session:
        old = self.tools.ctx.session
        self.tools.ctx.session = session
        self.session = session if session.task_id == "chat" else self.session
        return old

    def conversation(self) -> Conversation:
        return Conversation(self)

"""
Persistent tasks. Every task is a JSON file that is rewritten (atomically) after each step, so a crash,
power loss or restart never loses progress, and finished steps are never executed twice.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

STATES = ("pending", "planning", "in_progress", "waiting_approval", "paused", "failed", "completed", "cancelled")
TRANSITIONS = {
    "pending": {"planning", "in_progress", "cancelled"},
    "planning": {"in_progress", "paused", "failed", "cancelled"},
    "in_progress": {"waiting_approval", "paused", "failed", "completed", "cancelled"},
    "waiting_approval": {"in_progress", "paused", "failed", "cancelled"},
    "paused": {"in_progress", "failed", "cancelled"},
    "failed": {"pending", "in_progress", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}
MAX_TRANSCRIPT = 40


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def step_key(tool: str, args: dict) -> str:
    return hashlib.sha1((tool + json.dumps(args, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()[:12]


@dataclass
class Task:
    id: str
    title: str
    objective: str
    state: str = "pending"
    created: str = ""
    updated: str = ""
    memory_refs: list = field(default_factory=list)
    steps: list = field(default_factory=list)         # dicts: n, tool, args, key, status, summary, effects, time
    transcript: list = field(default_factory=list)    # model conversation, so a task can resume mid-way
    checkpoints: list = field(default_factory=list)
    result: str = ""
    verification: list = field(default_factory=list)  # tool-confirmed effects (not model claims)
    error: str = ""
    tainted: bool = False                             # untrusted content was read during this task


class TaskManager:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, tid: str) -> Path:
        return self.dir / f"{tid}.json"

    def save(self, t: Task) -> None:
        t.updated = _now()
        tmp = self._path(t.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(t), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self._path(t.id))

    def create(self, title: str, objective: str, memory_refs: Optional[list] = None) -> Task:
        tid = f"t-{datetime.now():%Y%m%d-%H%M}-{secrets.token_hex(2)}"
        t = Task(tid, title.strip()[:80] or "task", objective.strip(), created=_now(), memory_refs=memory_refs or [])
        self.save(t)
        return t

    def get(self, tid: str) -> Optional[Task]:
        tid = tid.strip()
        p = self._path(tid)
        if not p.is_file():
            matches = [f for f in self.dir.glob("t-*.json") if f.stem.endswith(tid) or f.stem.startswith(tid)]
            if len(matches) != 1:
                return None
            p = matches[0]
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return Task(**{k: v for k, v in data.items() if k in Task.__dataclass_fields__})
        except Exception:
            return None

    def list(self, states: Optional[tuple] = None) -> list:
        out = [self.get(p.stem) for p in sorted(self.dir.glob("t-*.json"))]
        out = [t for t in out if t]
        return [t for t in out if not states or t.state in states]

    def transition(self, t: Task, new_state: str, note: str = "") -> None:
        if new_state == t.state:
            return
        if new_state not in TRANSITIONS.get(t.state, set()):
            raise ValueError(f"illegal task transition {t.state} -> {new_state}")
        t.state = new_state
        if note:
            t.checkpoints.append({"time": _now(), "state": new_state, "note": note})
        self.save(t)

    def begin_step(self, t: Task, tool: str, args: dict) -> dict:
        step = {"n": len(t.steps) + 1, "tool": tool, "args": args, "key": step_key(tool, args),
                "status": "running", "summary": "", "effects": [], "time": _now()}
        t.steps.append(step)
        self.save(t)
        return step

    def end_step(self, t: Task, step: dict, ok: bool, summary: str, effects: list) -> None:
        step["status"] = "done" if ok else "failed"
        step["summary"] = summary[:300]
        step["effects"] = list(effects)
        self.save(t)

    @staticmethod
    def already_done(t: Task, tool: str, args: dict) -> Optional[dict]:
        key = step_key(tool, args)
        for s in t.steps:
            if s["key"] == key and s["status"] == "done":
                return s
        return None

    def add_transcript(self, t: Task, role: str, content: str) -> None:
        t.transcript.append({"role": role, "content": content[:6000]})
        t.transcript = t.transcript[-MAX_TRANSCRIPT:]

    def recover(self) -> list:
        """Call at startup: tasks that were mid-run when the app stopped become 'paused'."""
        recovered = []
        for t in self.list(("in_progress", "planning", "waiting_approval")):
            for s in t.steps:
                if s["status"] == "running":
                    s["status"] = "interrupted"
            t.state = "paused"
            t.error = "interrupted (application stopped)"
            t.checkpoints.append({"time": _now(), "state": "paused", "note": "recovered after restart"})
            self.save(t)
            recovered.append(t)
        return recovered

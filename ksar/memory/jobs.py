"""
Persistent background job queue for memory work.

Jobs survive restarts (queue.json is rewritten atomically after every change). Running jobs found at start-up go back to
pending. The queue holds only small descriptions ("extract memory from these journal turns"); the journal and the notes
stay the source of truth, so a lost queue can be rebuilt: catch_up() re-queues journal turns that were never processed.
The queue has its own tiny lock and never touches the memory store, so enqueueing is instant for the foreground.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

PENDING, RUNNING, DONE, FAILED, DEFERRED, CANCELLED = "pending", "running", "done", "failed", "deferred", "cancelled"
MAX_KEEP_FINISHED = 120


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


@dataclass
class Job:
    id: str
    type: str
    created: str
    status: str = PENDING
    priority: int = 5                    # 1 = most urgent (explicit "remember"), 9 = housekeeping
    payload: dict = field(default_factory=dict)
    source_refs: list = field(default_factory=list)      # journal sources such as 'chat:2026-11-19#4'
    attempts: int = 0
    max_attempts: int = 3
    next_try: str = ""                   # ISO time before which the job must not run (retry back-off)
    finished: str = ""
    result: str = ""
    dedupe_key: str = ""


class JobQueue:
    def __init__(self, path: Path, clock: Callable = _now):
        self.path = Path(path)
        self.clock = clock
        self._lock = threading.Lock()
        self.jobs: list = []
        self.meta: dict = {}                 # small facts: last_extract_ok, watermark per journal day
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        fields = set(Job.__dataclass_fields__)
        self.jobs = []
        for d in data.get("jobs", []):
            if isinstance(d, dict) and {"id", "type", "created"} <= set(d):
                j = Job(**{k: v for k, v in d.items() if k in fields})
                if j.status == RUNNING:          # the app stopped in the middle of it
                    j.status = PENDING
                self.jobs.append(j)
        self.meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}

    def _save(self) -> None:
        finished = [j for j in self.jobs if j.status in (DONE, FAILED, CANCELLED)]
        if len(finished) > MAX_KEEP_FINISHED:
            drop = {j.id for j in finished[:len(finished) - MAX_KEEP_FINISHED]}
            self.jobs = [j for j in self.jobs if j.id not in drop]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({"jobs": [asdict(j) for j in self.jobs], "meta": self.meta},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass

    # ------------------------------------------------------------------ producing
    @staticmethod
    def key_for(type_: str, payload: dict) -> str:
        return type_ + ":" + hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def add(self, type_: str, payload: Optional[dict] = None, priority: int = 5, source_refs: Optional[list] = None,
            dedupe_key: str = "", max_attempts: int = 3, delay_s: float = 0) -> Optional[Job]:
        """Queues a job. Returns None when an identical job is already waiting (no duplicates)."""
        payload = payload or {}
        key = dedupe_key or self.key_for(type_, payload)
        with self._lock:
            if any(j.dedupe_key == key and j.status in (PENDING, RUNNING, DEFERRED) for j in self.jobs):
                return None
            nxt = (self.clock() + timedelta(seconds=delay_s)).isoformat(timespec="seconds") if delay_s else ""
            job = Job(uuid.uuid4().hex[:10], type_, self.clock().isoformat(timespec="seconds"), PENDING, priority,
                      payload, list(source_refs or []), 0, max_attempts, nxt, "", "", key)
            self.jobs.append(job)
            self._save()
            return job

    # ------------------------------------------------------------------ consuming
    def _ready(self, j: Job, now: datetime) -> bool:
        if j.status != PENDING:
            return False
        if j.next_try:
            try:
                return datetime.fromisoformat(j.next_try) <= now
            except ValueError:
                return True
        return True

    def claim(self, types=None, limit: int = 1) -> list:
        """Marks up to `limit` ready jobs of the given types as running (most urgent, then oldest first)."""
        now = self.clock()
        with self._lock:
            ready = sorted((j for j in self.jobs if self._ready(j, now) and (types is None or j.type in types)),
                           key=lambda j: (j.priority, j.created, j.id))[:limit]
            for j in ready:
                j.status = RUNNING
            if ready:
                self._save()
            return list(ready)

    def has_ready(self, types=None) -> bool:
        now = self.clock()
        with self._lock:
            return any(self._ready(j, now) and (types is None or j.type in types) for j in self.jobs)

    def release(self, jobs, count_attempt: bool = False) -> None:
        """Gives claimed jobs back (e.g. the owner returned): they run again later."""
        with self._lock:
            for j in jobs:
                j.status = PENDING
                if count_attempt:
                    j.attempts += 1
            self._save()

    def done(self, jobs, result: str = "") -> None:
        with self._lock:
            for j in jobs:
                j.status, j.result, j.finished = DONE, result[:200], self.clock().isoformat(timespec="seconds")
            self._save()

    def fail(self, jobs, error: str, retry_in_s: float = 300) -> None:
        """Counts an attempt; retried later until max_attempts, then marked failed."""
        with self._lock:
            for j in jobs:
                j.attempts += 1
                j.result = error[:200]
                if j.attempts >= j.max_attempts:
                    j.status, j.finished = FAILED, self.clock().isoformat(timespec="seconds")
                else:
                    j.status = PENDING
                    j.next_try = (self.clock() + timedelta(seconds=retry_in_s * j.attempts)).isoformat(timespec="seconds")
            self._save()

    def defer(self, job: Job, hours: float, result: str = "") -> None:
        with self._lock:
            job.status, job.result = PENDING, result[:200]
            job.next_try = (self.clock() + timedelta(hours=hours)).isoformat(timespec="seconds")
            self._save()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            for j in self.jobs:
                if j.id == job_id and j.status in (PENDING, DEFERRED, RUNNING):
                    j.status = CANCELLED
                    self._save()
                    return True
        return False

    # ------------------------------------------------------------------ inspection
    def counts(self) -> dict:
        with self._lock:
            out: dict = {}
            for j in self.jobs:
                out[j.status] = out.get(j.status, 0) + 1
            return out

    def pending_count(self, types=None) -> int:
        with self._lock:
            return sum(1 for j in self.jobs if j.status in (PENDING, RUNNING) and (types is None or j.type in types))

    def listing(self, limit: int = 15) -> list:
        with self._lock:
            live = [j for j in self.jobs if j.status in (PENDING, RUNNING, DEFERRED)]
            recent = [j for j in self.jobs if j.status in (DONE, FAILED, CANCELLED)][-5:]
            return [Job(**asdict(j)) for j in (live + recent)[:limit]]

    def set_meta(self, key: str, value) -> None:
        with self._lock:
            self.meta[key] = value
            self._save()

    def get_meta(self, key: str, default=None):
        with self._lock:
            return self.meta.get(key, default)

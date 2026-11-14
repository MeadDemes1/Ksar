"""
Idle worker: uses the quiet moments to get ready for your next message and to keep memory tidy.

It only runs while you are NOT talking to Ksar (default: 20 s after the last answer). The moment you press Enter it
stops (a running model call is cancelled at its next token), so it never makes your chat slower.

Jobs, in order (one per tick):
  warm         load the standard model into RAM, so your first message does not wait for the model to load
  consolidate  turn the latest chat exchanges into memory, on the LIGHT chip, in conservative mode:
               only additions, and only facts whose wording is really found in what you said
  split        branch a note that has grown too big, on the LIGHT chip
  doctor       once a day, check the memory notes for damage (no model; only reports)

Thread safety: the model call happens WITHOUT the app lock; memory is read and written only while holding
app.lock, which the console holds while it executes your commands.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime

from .chips import ChipState
from .llm import no_think
from .memory.maintenance import split_apply, split_prompt
from .memory.store import today


class Aborted(Exception):
    """The user came back; the background job stops at once."""


class IdleWorker:
    TICK = 1.0
    BACKOFF_S = 600

    def __init__(self, app, clock=time.monotonic):
        self.app, self.clock = app, clock
        self.convo = None
        self.abort = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.last_active = clock()                                   # start-up counts as activity (for warm)
        self.last_idle_start = clock() - app.settings.idle_after_s   # ...but we are idle right away: warm early
        self._warmed_once = False
        self._last_ps = 0.0
        self._doctor_day = ""
        self._skip_until: dict = {}
        self.notes: list = []                                        # shown before your next prompt
        self.log: deque = deque(maxlen=60)
        self.busy = ""
        self.jobs = [("warm", self.job_warm), ("consolidate", self.job_consolidate),
                     ("split", self.job_split), ("doctor", self.job_doctor)]

    # ------------------------------------------------------------------ wiring
    def attach(self, convo) -> None:
        self.convo = convo

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ksar-idle", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.abort.set()
        if self._thread:
            self._thread.join(timeout)

    def user_active(self) -> None:
        """Call when the user submits something: background work stops now."""
        self.last_active = self.clock()
        self.abort.set()

    def user_idle(self) -> None:
        """Call when Ksar has finished answering: the quiet period starts now."""
        self.last_idle_start = self.clock()
        self.abort.clear()

    # ------------------------------------------------------------------ scheduling
    def is_idle(self) -> bool:
        s = self.app.settings
        return (bool(s.idle_enabled) and not self.abort.is_set()
                and self.clock() - self.last_idle_start >= s.idle_after_s)

    def _loop(self) -> None:
        while not self._stop.wait(self.TICK):
            try:
                self.run_once()
            except Exception as e:                                   # a background job must never kill the console
                self._record("worker", f"error: {type(e).__name__}: {e}")

    def run_once(self, force: bool = False):
        """One scheduler tick. Returns the (job, message) that ran, or None when there was nothing to do."""
        if not force and not self.is_idle():
            return None
        if force:
            self.abort.clear()
        for name, job in self.jobs:
            if self._skip_until.get(name, 0) > self.clock():
                continue
            self.busy = name
            try:
                msg = job()
            except Aborted:
                self._record(name, "stopped: you came back")
                return name, "stopped"
            except Exception as e:
                self._skip_until[name] = self.clock() + self.BACKOFF_S
                self._record(name, f"failed: {type(e).__name__}: {e}")
                continue
            finally:
                self.busy = ""
            if msg:
                self._record(name, msg, notify=True)
                return name, msg
        return None

    def _record(self, job: str, msg: str, notify: bool = False) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log.append((stamp, job, msg))
        if notify:
            self.notes.append(f"{job}: {msg}")
        try:
            with open(self.app.paths.logs / "idle.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": stamp, "job": job, "msg": msg}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def pop_notes(self) -> list:
        out, self.notes = self.notes, []
        return out

    # ------------------------------------------------------------------ the light model
    def _light_chat(self, prompt: str, chip, max_tokens: int = 700) -> str:
        """JSON reply from the light chip, cancellable: stops at the next token once the user is back."""
        a, llm = self.app, self.app.llm
        msgs = [{"role": "user", "content": prompt}]
        ka = a.chips.keep_alive_for(chip)
        if self.abort.is_set():
            raise Aborted()
        if not callable(getattr(llm, "stream_chat", None)):
            return llm.chat(msgs, model=chip.model_id, json_mode=True, keep_alive=ka, temperature=0.2, **no_think(llm))
        parts = []
        gen = llm.stream_chat(msgs, model=chip.model_id, json_mode=True, keep_alive=ka, temperature=0.2,
                              think=False, max_tokens=max_tokens)
        try:
            for kind, text in gen:
                if self.abort.is_set():
                    raise Aborted()
                if kind == "content":
                    parts.append(text)
        finally:
            gen.close()                                              # closes the connection: Ollama stops generating
        return "".join(parts)

    # ------------------------------------------------------------------ jobs
    def job_warm(self):
        a, s = self.app, self.app.settings
        std = a.chips.chips.get("standard")
        if not s.warm_on_start or std is None:
            return None
        if time.time() - std.failed_at < 300 and std.failed_at:
            return None
        if self.clock() - self._last_ps > 60:                        # what does Ollama really have loaded?
            a.chips.refresh()
            self._last_ps = self.clock()
        if std.state != ChipState.UNLOADED:
            self._warmed_once = True
            return None
        if self._warmed_once and self.clock() - self.last_active > 1800:
            return None                                              # nobody around for 30 min: leave it unloaded
        self._warmed_once = True
        if a.chips.ensure_loaded(std):
            return f"{std.model_id} is loaded and ready"
        return None

    def job_consolidate(self):
        a, convo = self.app, self.convo
        if convo is None:
            return None
        with a.lock:
            if not convo.pending:
                return None
            chip, why = a.chips.light_for_idle()
            if chip is None:                                         # no light chip: the normal save points handle it
                return None
            snapshot, gen = list(convo.pending), convo.gen
            src, tainted = a.session.src, a.session.tainted
            prompt = a.keeper.prepare(snapshot)
        if not a.chips.ensure_loaded(chip):
            raise RuntimeError(f"light chip could not be loaded: {chip.last_error}")
        raw = self._light_chat(prompt, chip)
        with a.lock:
            if convo.gen != gen:                                     # a normal save happened meanwhile
                return None
            rep = a.keeper.finish(raw, src, tainted, strict_for=snapshot)
            if not rep.ok:
                self._skip_until["consolidate"] = self.clock() + self.BACKOFF_S
                return None
            del convo.pending[:len(snapshot)]
            convo.gen += 1
        saved = [l for l in rep.lines if not l.startswith("nothing")]
        return f"memory updated ({len(saved)} change{'s' if len(saved) != 1 else ''})" if saved else \
               "checked the last chat, nothing to save"

    def job_split(self):
        a = self.app
        with a.lock:
            chip, _ = a.chips.light_for_idle()
            if chip is None:
                return None
            thr = a.settings.split_fact_threshold
            cand = next((n for n in a.memory.all_notes() if len(n.facts) > thr
                         and n.type not in ("episodic", "core")), None)
            if cand is None:
                return None
            nid, expect = cand.id, [f.fid for f in cand.facts]
            prompt = split_prompt(a.memory, nid, thr)
        if not prompt:
            self._skip_until["split"] = self.clock() + 3600
            return None
        if not a.chips.ensure_loaded(chip):
            raise RuntimeError(f"light chip could not be loaded: {chip.last_error}")
        raw = self._light_chat(prompt, chip, max_tokens=500)
        with a.lock:
            made = split_apply(a.memory, nid, raw, thr, expect=expect)
        if not made:
            self._skip_until["split"] = self.clock() + 3600
            return None
        return f"note '{nid}' branched into {', '.join(made)}"

    def job_doctor(self):
        a = self.app
        day = today()
        if self._doctor_day == day:
            return None
        with a.lock:
            issues = a.memory.doctor(fix=False)
        self._doctor_day = day
        n = len([i for i in issues if not getattr(i, "fixed", False)])
        return f"memory check found {n} issue(s): run /doctor fix" if n else None

    # ------------------------------------------------------------------ report
    def status(self) -> dict:
        s = self.app.settings
        return {"enabled": bool(s.idle_enabled), "idle_after_s": s.idle_after_s, "busy": self.busy,
                "idle_now": self.is_idle(), "light": s.light_chip or "(off)",
                "pending": len(self.convo.pending) if self.convo else 0, "log": list(self.log)[-8:]}

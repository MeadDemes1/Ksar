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

from .chips import ChipRole, ChipState
from .llm import no_think
from .memory.graph import organize_plan, refresh_branch_summaries, validate_graph
from .memory.keeper import Conflict
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
        self._last_ps = None                                         # None = never asked Ollama yet
        self._doctor_day = ""
        self._skip_until: dict = {}
        self.notes: list = []                                        # shown before your next prompt
        self.log: deque = deque(maxlen=60)
        self.busy = ""
        self._index_gen = -1
        self._last_scan = 0.0
        self.last_extract = ""
        self.jobs = [("warm", self.job_warm), ("extract", self.job_extract), ("consolidate", self.job_consolidate),
                     ("recheck", self.job_recheck), ("index", self.job_index), ("split", self.job_split),
                     ("doctor", self.job_doctor), ("organize", self.job_organize), ("clock", self.job_clock)]

    @property
    def running(self) -> bool:
        """True while the worker thread is alive and background work is enabled: memory is then saved in the
        background instead of during the chat."""
        return bool(self._thread and self._thread.is_alive() and self.app.settings.idle_enabled)

    @property
    def queue(self):
        return self.app.jobs

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
            raw = llm.chat(msgs, model=chip.model_id, json_mode=True, keep_alive=ka, temperature=0.2, **no_think(llm))
            if self.abort.is_set():                                  # you came back during a non-streaming call
                raise Aborted()
            return raw
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
        if self._last_ps is None or self.clock() - self._last_ps > 60:   # what does Ollama really have loaded?
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
        with convo.state:
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
        with convo.state:
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
        chip, _ = a.chips.light_for_idle()
        if chip is None:
            return None
        thr = a.settings.split_fact_threshold
        cand = next((n for n in a.memory.cached_notes() if len(n.facts) > thr
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
        issues = a.memory.doctor(fix=False) + validate_graph(a.memory)
        self._doctor_day = day
        n = len([i for i in issues if not getattr(i, "fixed", False)])
        return f"memory check found {n} issue(s): run /doctor fix" if n else None

    # ------------------------------------------------------------------ memory extraction (the independent worker)
    EXTRACT_BATCH = 8            # exchanges processed together as one coherent episode
    CHUNK_CHARS = 9000           # a long conversation is processed in chunks of about this size

    def _memory_chip(self):
        """The chip for background memory work: the Idle/Light chip when configured, otherwise Basic.
        (The Idle chip never answers chat, and chat never waits for this.)"""
        a, mode = self.app, self.app.settings.memory_chip
        if mode in ("auto", "light"):
            chip, why = a.chips.light_for_idle()
            if chip is not None:
                return chip, "light"
            if mode == "light":
                return None, why
        std = a.chips.chips.get("standard")
        if std is None or (std.failed_at and time.time() - std.failed_at < 300):
            return None, "no usable chip"
        return std, "standard"

    def _turns_for(self, jobs) -> tuple:
        """Journal turns named by the claimed jobs (the journal is the original record) and whether any was explicit."""
        a = self.app
        srcs, explicit, turns = [], False, {}
        for j in jobs:
            p = j.payload
            explicit = explicit or bool(p.get("explicit"))
            srcs += list(p.get("srcs", [])) + list(j.source_refs)
            if p.get("since"):
                from .timeinfo import parse_iso
                since = parse_iso(p["since"])
                if since:
                    for t in a.memory.journal_since(since.astimezone().replace(tzinfo=None)):
                        turns[t["src"]] = t
            if p.get("after_watermark"):
                for t in a.unprocessed_turns():
                    turns[t["src"]] = t
        for t in a.memory.journal_turns(srcs):
            turns[t["src"]] = t
        return sorted(turns.values(), key=lambda t: (t["day"], t["n"])), explicit

    def job_extract(self):
        a, q = self.app, self.queue
        if not a.settings.memory_async or not q.has_ready(("extract",)):
            return None
        chip, kind = self._memory_chip()
        if chip is None:
            self._skip_until["extract"] = self.clock() + 120          # no model available: try again in a while
            return None
        jobs = q.claim(("extract",), limit=self.EXTRACT_BATCH)
        if not jobs:
            return None
        try:
            turns, explicit = self._turns_for(jobs)
            if not turns:
                q.done(jobs, "nothing to read")
                return None
            chunk, size = [], 0
            for t in turns:                                           # long conversations: a chunk now, the rest queued
                size += len(t["user"]) + len(t["assistant"][:400])
                if chunk and size > self.CHUNK_CHARS:
                    break
                chunk.append(t)
            rest = turns[len(chunk):]
            if not a.chips.ensure_loaded(chip):
                raise RuntimeError(f"{chip.name} chip could not be loaded: {chip.last_error}")
            allow_correct = chip.role != ChipRole.LIGHTWEIGHT
            plan = rep = None
            for attempt in range(3):
                prompt, ctx = a.keeper.prepare_v2(chunk, explicit)
                raw = self._light_chat(prompt, chip, max_tokens=1000)
                plan = a.keeper.plan(raw, ctx, allow_correct=allow_correct,
                                     verify=lambda p: self._light_chat(p, chip, max_tokens=40))
                if not plan.ok:
                    raise ValueError("the model's reply was not readable")
                try:
                    rep = a.keeper.apply(plan, chunk[-1]["src"])
                    break
                except Conflict:                                      # a note changed meanwhile: plan again from fresh data
                    if attempt == 2:
                        raise
        except Aborted:
            q.release(jobs)                                           # you came back: nothing lost, runs again later
            raise
        except Exception as e:
            q.fail(jobs, f"{type(e).__name__}: {e}")
            raise
        for pf in plan.deferred:                                      # unclear facts wait for more evidence
            q.add("recheck", {"note": pf.note, "text": pf.text, "kind": pf.kind, "conf": pf.conf,
                              "evidence": pf.evidence, "created": a.time.stamp()}, priority=8, delay_s=6 * 3600,
                  max_attempts=4)
        if rest:
            q.add("extract", {"srcs": [t["src"] for t in rest], "explicit": explicit}, priority=3)
        a.mark_processed([t["src"] for t in chunk])
        changes = [l for l in rep.lines if not l.startswith(("nothing", "skipped"))]
        q.done(jobs, f"{len(changes)} change(s) on the {kind} chip")
        stamp = a.time.stamp()
        q.set_meta("last_extract_ok", stamp)
        self.last_extract = stamp
        extra = f", {len(plan.deferred)} waiting for more evidence" if plan.deferred else ""
        return (f"memory updated ({len(changes)} change{'s' if len(changes) != 1 else ''}{extra})" if changes
                else f"checked {len(chunk)} exchange(s), nothing worth saving{extra}")

    def job_recheck(self):
        """Facts that could not be confirmed earlier are re-checked against the owner's newer messages, then dropped."""
        a, q = self.app, self.queue
        jobs = q.claim(("recheck",), limit=1)
        if not jobs:
            return None
        job = jobs[0]
        chip, _ = self._memory_chip()
        if chip is None:
            q.release(jobs)
            self._skip_until["recheck"] = self.clock() + 600
            return None
        try:
            from datetime import datetime as _dt, timedelta as _td
            p = job.payload
            since = _dt.now() - _td(days=7)
            turns = a.memory.journal_since(since)[-40:]
            if not turns:
                q.done(jobs, "no conversation to check against")
                return None
            if not a.chips.ensure_loaded(chip):
                raise RuntimeError("chip could not be loaded")
            raw = json.dumps({"ops": [{"op": "add", "note": p["note"], "facts": [
                {"text": p["text"], "kind": p.get("kind", "fact"), "conf": p.get("conf", "stated"),
                 "evidence": p.get("evidence", "")}]}]})
            ctx = {"revs": {}, "user_text": " \n".join(t["user"] for t in turns),
                   "turn_srcs": [(t["src"], t["user"]) for t in turns]}
            plan = a.keeper.plan(raw, ctx, allow_correct=False, verify=lambda pr: self._light_chat(pr, chip, 40))
            if plan.deferred:
                age_ok = True
                try:
                    from .timeinfo import parse_iso
                    created = parse_iso(p.get("created"))
                    age_ok = created is None or (a.time.now() - created).days < 14
                except Exception:
                    pass
                if age_ok and job.attempts < job.max_attempts - 1:
                    job.attempts += 1
                    q.defer(job, 24, "still unclear")
                    return None
                q.done(jobs, "expired: never confirmed")
                return None
            if plan.empty:
                q.done(jobs, "rejected on re-check")
                return None
            rep = a.keeper.apply(plan, turns[-1]["src"])
            q.done(jobs, "confirmed")
            return f"a pending fact was confirmed and saved ({len(rep.lines)} change(s))"
        except Aborted:
            q.release(jobs)
            raise
        except Conflict:
            q.release(jobs)
            return None
        except Exception as e:
            q.fail(jobs, f"{type(e).__name__}: {e}")
            raise

    def job_index(self):
        """Incremental upkeep: notice edits made outside Ksar, then refresh branch summaries (only if they changed)."""
        a = self.app
        now = self.clock()
        if now - self._last_scan > 30:
            self._last_scan = now
            a.memory.scan()
        if a.memory.gen == self._index_gen:
            return None
        self._index_gen = a.memory.gen
        changed = refresh_branch_summaries(a.memory)
        self._index_gen = a.memory.gen
        return f"branch summaries refreshed: {', '.join(changed)}" if changed else None

    def job_organize(self):
        a = self.app
        if not a.settings.memory_auto_organize:
            return None
        moves = organize_plan(a.memory, limit=3)
        done = [f"{nid} -> {folder}" for nid, folder in moves if a.memory.move_note(nid, folder)]
        return "moved notes into branch folders: " + "; ".join(done) if done else None

    def job_clock(self):
        """Deterministic time upkeep: the date rolled over while Ksar kept running."""
        ro = self.app.time.rollover()
        if not ro:
            return None
        self._doctor_day = ""                                         # a new day: run the daily memory check again
        return f"new day: {ro[1].isoformat()}"

    # ------------------------------------------------------------------ report
    def status(self) -> dict:
        s = self.app.settings
        return {"enabled": bool(s.idle_enabled), "idle_after_s": s.idle_after_s, "busy": self.busy,
                "idle_now": self.is_idle(), "light": s.light_chip or "(off)",
                "pending": len(self.convo.pending) if self.convo else 0, "log": list(self.log)[-8:],
                "queue": self.queue.counts(), "last_extract": self.last_extract or self.queue.get_meta("last_extract_ok", "")}

"""
Time service: reliable, deterministic time awareness (no model involved).

It knows the current local date/time and timezone, when the session started, when the owner last wrote and when Ksar last
answered, and how long the owner has been away. All timestamps are timezone-aware. The OS timezone is used unless
`timezone` is set in settings (an IANA name such as "Europe/Berlin"; needs the OS tz database, otherwise the OS zone is used).

Time reaches the model only when it matters, so the cached chat prompt never changes:
  * the message is about time ("what day is it", "tomorrow", "how long ago", ...), or
  * the owner comes back after a long pause - a short factual note, once per pause. Ksar can mention it naturally;
    it is told not to guess what the owner did meanwhile.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

try:                                       # Python 3.9+; on Windows the IANA database may be missing
    from zoneinfo import ZoneInfo
except ImportError:                        # pragma: no cover
    ZoneInfo = None

TIME_Q_RE = re.compile(
    r"\b(what(?:'s| is)? the (?:time|date|day)|what time|what day|what date|current (?:time|date)|today|tonight|tomorrow|"
    r"yesterday|this (?:morning|evening|week|month|year)|next (?:week|month|year|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)|how (?:long|many (?:days|hours|minutes|weeks)) (?:ago|until|since|left|till)|"
    r"\d+ (?:minutes?|hours?|days?|weeks?) ago|in \d+ (?:minutes?|hours?|days?|weeks?)|deadline|remind me|"
    r"schedule|due date)\b|сегодня|завтра|вчера|сколько времени|который час|какое число|какой день|дедлайн|напомни",
    re.I)


def system_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(text) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo else dt.astimezone()          # old/naive values are read as local time


def describe(td: timedelta) -> str:
    s = int(td.total_seconds())
    if s < 90:
        return "a minute"
    if s < 3600 * 2:
        return f"{s // 60} minutes"
    if s < 86400 * 2:
        return f"{s // 3600} hours"
    return f"{s // 86400} days"


class TimeService:
    def __init__(self, tz_name: str = "", state_path: Optional[Path] = None, clock: Callable = system_now,
                 long_gap_hours: float = 4.0):
        self.clock = clock
        self.tz_name = (tz_name or "").strip()
        self.tz_problem = ""
        self._tz = self._resolve(self.tz_name)
        self.long_gap = timedelta(hours=max(0.1, float(long_gap_hours)))
        self.state_path = Path(state_path) if state_path else None
        self.session_start = self.now()
        st = self._load()
        self.last_interaction: Optional[datetime] = parse_iso(st.get("last_interaction"))
        self.last_response: Optional[datetime] = parse_iso(st.get("last_response"))
        self.previous: Optional[datetime] = None          # the owner's message BEFORE the current one
        self._acked: Optional[datetime] = None            # the pause we already mentioned
        self._day = self.now().date()

    # ------------------------------------------------------------------ clock
    def _resolve(self, name: str):
        if not name:
            return None                                   # follow the operating system (incl. daylight saving)
        if ZoneInfo is None:
            self.tz_problem = "zoneinfo is not available; using the operating system timezone"
            return None
        try:
            return ZoneInfo(name)
        except Exception:
            self.tz_problem = f"timezone '{name}' not found (on Windows: pip install tzdata); using the operating system timezone"
            return None

    def configure(self, tz_name: str, long_gap_hours: float) -> None:
        self.tz_name, self.tz_problem = (tz_name or "").strip(), ""
        self._tz = self._resolve(self.tz_name)
        self.long_gap = timedelta(hours=max(0.1, float(long_gap_hours)))

    def now(self) -> datetime:
        dt = self.clock()
        return dt.astimezone(self._tz) if self._tz else dt.astimezone()

    def tz_label(self) -> str:
        off = self.now().strftime("%z") or "+0000"
        label = f"UTC{off[:3]}:{off[3:]}"
        return f"{label} ({self.tz_name})" if self._tz and self.tz_name else label

    def stamp(self) -> str:
        """Timezone-aware ISO timestamp for logs and queues."""
        return self.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------ persistence of "last seen"
    def _load(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path else {}
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        if not self.state_path:
            return
        data = {"last_interaction": self.last_interaction.isoformat() if self.last_interaction else "",
                "last_response": self.last_response.isoformat() if self.last_response else ""}
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_name(self.state_path.name + ".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            pass

    # ------------------------------------------------------------------ interaction tracking
    def begin_message(self) -> None:
        """Call when the owner's message arrives."""
        self.previous = self.last_interaction
        self.last_interaction = self.now()
        self._save()

    def note_response(self) -> None:
        """Call when Ksar finished answering."""
        self.last_response = self.now()
        self._save()

    def gap(self) -> Optional[timedelta]:
        """Silence before the current message (None for the very first message ever)."""
        return (self.last_interaction - self.previous) if self.previous and self.last_interaction else None

    def session_age(self) -> timedelta:
        return self.now() - self.session_start

    def idle_for(self) -> Optional[timedelta]:
        return (self.now() - self.last_interaction) if self.last_interaction else None

    def rollover(self):
        """(old date, new date) once when the local date changes while Ksar keeps running; else None."""
        today = self.now().date()
        if today != self._day:
            old, self._day = self._day, today
            return old, today
        return None

    # ------------------------------------------------------------------ what the model gets (only when relevant)
    def return_note(self) -> str:
        g = self.gap()
        if g is None or g < self.long_gap or self._acked == self.previous:
            return ""
        self._acked = self.previous
        prev = self.previous.astimezone(self.now().tzinfo)
        new_day = prev.date() != self.now().date()
        text = (f"The owner is back after {describe(g)} (their previous message was "
                f"{'yesterday' if (self.now().date() - prev.date()).days == 1 else prev.strftime('%a %d %b')} at "
                f"{prev.strftime('%H:%M')}). It is now {self.now().strftime('%A %H:%M')}"
                + (", a new day" if new_day else "") + ". You may acknowledge this naturally once if it fits; do not "
                "guess what they did in the meantime.")
        return text

    def context_for(self, text: str) -> str:
        """'' normally. A bracketed note for this ONE message when it asks about time or the owner just came back."""
        bits = []
        if TIME_Q_RE.search(text or ""):
            n = self.now()
            bits.append(f"Now: {n.strftime('%A %d %B %Y, %H:%M')} ({self.tz_label()}).")
        back = self.return_note()
        if back:
            bits.append(back)
        return "[Time: " + " ".join(bits) + "]" if bits else ""

    def status_lines(self) -> list:
        n = self.now()
        out = [f"now:           {n.strftime('%A %d %B %Y %H:%M:%S')} {self.tz_label()}",
               f"session:       started {self.session_start.strftime('%H:%M')} ({describe(self.session_age())} ago)"]
        if self.last_interaction:
            out.append(f"last message:  {self.last_interaction.astimezone(n.tzinfo).strftime('%a %d %b %H:%M')}")
        if self.last_response:
            out.append(f"last answer:   {self.last_response.astimezone(n.tzinfo).strftime('%a %d %b %H:%M')}")
        if self.tz_problem:
            out.append(f"note:          {self.tz_problem}")
        return out

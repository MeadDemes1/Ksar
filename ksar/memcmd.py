"""
Recognises a pure memory command ("remember the important things we discussed during the last hour", "save this
conversation", "запомни всё за последний час") so Ksar can answer at once and hand the work to the background worker.
A message that carries its own content ("remember that my dog is Rex") is NOT a pure command: it is answered normally
and its facts are extracted from the journal like everything else.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

TRIGGER = re.compile(r"^\s*(?:please\s+|пожалуйста\s+)?(?:remember|memori[sz]e|save|store|keep|запомни(?:те)?|сохрани(?:те)?)\b(.*)$",
                     re.I | re.S)
FILLER = set("""please the a an all everything important things stuff points details we you discussed talked about during in
over this our conversation chat session so far today last hour hours minute minutes day days and of that them it from
earlier just now what for me to memory your into пожалуйста все всё важное важные вещи что мы обсуждали обсудили в за
эту нашу беседу разговор чат сессию последний последние последнюю час часа часов минут минуты день дня дней сегодня
это нас о об и в память""".split())
WINDOW_RE = re.compile(r"(?:last|past|previous)\s+(\d+\s*)?(hour|hours|minute|minutes|min|day|days)\b|"
                       r"последн\w+\s+(\d+\s*)?(час\w*|минут\w*|д(?:ень|ня|ней))", re.I)


def parse_memory_command(text: str, now: datetime, session_start: datetime) -> Optional[dict]:
    """None, or {"since": aware datetime, "label": str} when the whole message is just a request to save the
    conversation (optionally over a time window)."""
    m = TRIGGER.match(text or "")
    if not m:
        return None
    rest = m.group(1).strip(" .!?,;:")
    words = re.findall(r"[^\W_]+", rest.lower(), flags=re.U)
    if any(w not in FILLER and not w.isdigit() for w in words):
        return None                                   # it contains real content: not a pure command
    w = WINDOW_RE.search(rest)
    if w:
        n_txt = (w.group(1) or w.group(3) or "").strip()
        unit = (w.group(2) or w.group(4) or "").lower()
        n = int(n_txt) if n_txt.isdigit() else 1
        if unit.startswith(("hour", "час")):
            delta, label = timedelta(hours=n), f"the last {n} hour(s)"
        elif unit.startswith(("min", "мин")):
            delta, label = timedelta(minutes=n), f"the last {n} minute(s)"
        else:
            delta, label = timedelta(days=n), f"the last {n} day(s)"
        return {"since": now - delta, "label": label}
    if re.search(r"\b(today|сегодня)\b", rest, re.I):
        return {"since": now.replace(hour=0, minute=0, second=0, microsecond=0), "label": "today"}
    return {"since": session_start, "label": "this conversation"}


def canned_reply(text: str) -> str:
    return ("Хорошо, я обработаю это в фоновом режиме." if re.search(r"[а-яё]", text or "", re.I)
            else "Okay, I'll process that in the background.")

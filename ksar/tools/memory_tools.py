"""Memory tools: the model reads and writes memory through the same validated pipeline as the keeper."""
from __future__ import annotations

from ..memory.keeper import INJECTION
from ..memory.store import NOTE_TYPES, render_note, safe_id
from .base import Param, Risk, Tool, ToolResult


def _conf(ctx) -> str:
    return "inferred" if ctx.session.tainted else "stated"


def _src(ctx) -> str:
    return ctx.session.src + ("+untrusted" if ctx.session.tainted else "")


def mem_search(ctx, a):
    text = ctx.index.context(a["query"], budget=3500, limit=int(a.get("limit", 4)))
    return ToolResult(True, "memory search", text)


def mem_read(ctx, a):
    note = ctx.memory.get(a["note"])
    if not note:
        return ToolResult(False, f"no note '{a['note']}'")
    if not a.get("history"):
        note.history = []
    return ToolResult(True, note.id, render_note(note)[:5000])


def mem_remember(ctx, a):
    if INJECTION.search(a["fact"]):
        return ToolResult(False, "refused: that looks like an instruction, not a fact")
    
    nid = safe_id(a["note"])
    if not nid:
        # Improved error message to guide the model
        return ToolResult(False, f"bad note name: '{a.get('note')}'. Use a short lowercase hyphenated id like 'user_prefs'")
    
    m = ctx.memory
    if not m.exists(nid):
        t = a.get("type") if a.get("type") in NOTE_TYPES and a.get("type") != "core" else "semantic"
        m.create(nid, a.get("title", ""), t, a.get("summary", ""), a.get("parent", "core"))
    
    r = m.add_fact(nid, a["fact"], _conf(ctx), _src(ctx), allow_supersede=(m.get(nid).type != "episodic"))
    
    if r.status in ("added", "superseded"):
        msg = f"stored in [{nid}]: {r.fact.text}" + (f" (replaced '{r.old.text}', kept in history)" if r.old else "")
        return ToolResult(True, msg, effects=[f"memory fact {r.status}: [{nid}] #{r.fact.fid}"])
    if r.status == "duplicate":
        return ToolResult(True, "already remembered")
    if r.status == "conflict":
        return ToolResult(False, "conflicts with a verified fact; sent to the owner for review")
    
    # Improved rejection message
    return ToolResult(False, r.note or "rejected (ensure 'fact' is not empty and 'note' is valid)")


def mem_correct(ctx, a):
    if INJECTION.search(a["new_text"]):
        return ToolResult(False, "refused: that looks like an instruction, not a fact")
    r = ctx.memory.correct_fact(a["note"], a["fact"], a["new_text"], _conf(ctx), _src(ctx))
    if r.status == "superseded":
        return ToolResult(True, f"corrected: '{r.old.text}' -> '{r.fact.text}' (old version kept in history)",
                          effects=[f"memory fact corrected in [{safe_id(a['note'])}]"])
    return ToolResult(False, r.note or "blocked by a verified fact (sent to review)")


def mem_link(ctx, a):
    ok = ctx.memory.link(a["a"], a["b"], a.get("rel", "related"))
    return ToolResult(ok, "linked" if ok else "both notes must exist", effects=["memory link created"] if ok else [])


def mem_attach(ctx, a):
    p = ctx.perms.resolve(a["path"])
    why = ctx.perms.check(p, "read")
    if why:
        return ToolResult(False, f"read not allowed: {why}")
    entry = ctx.memory.attach_file(a["note"], str(p))
    if not entry:
        return ToolResult(False, "note or file not found")
    return ToolResult(True, f"file attached with fingerprint {entry[1][:12]}",
                      effects=[f"file reference stored in [{safe_id(a['note'])}]: {p}"])


def mem_forget(ctx, a):
    r = ctx.memory.retract_fact(a["note"], a["fact"], "retracted by request")
    return ToolResult(r.old is not None, "moved to the note's History (recoverable)" if r.old else "fact not found",
                      effects=["memory fact retracted"] if r.old else [])


def tools() -> list:
    P = Param
    return [
        Tool("memory.search", "Search long-term memory (returns notes, facts with #ids, links, files).",
             {"query": P(), "limit": P("integer", required=False, default=4)}, Risk.READ, mem_search),
        Tool("memory.read", "Open one memory note in full.", {"note": P(), "history": P("boolean", required=False, default=False)},
             Risk.READ, mem_read),
        Tool("memory.remember", "Store one fact. Use 'Key: value' for attributes. Creates the note if new.",
             {"note": P(desc="note id"), "fact": P(), "title": P(required=False), "type": P(required=False),
              "summary": P(required=False), "parent": P(required=False, default="core")}, Risk.WRITE, mem_remember),
        Tool("memory.correct", "Replace a wrong fact (old one is kept in history).",
             {"note": P(), "fact": P(desc="fact #id"), "new_text": P()}, Risk.WRITE, mem_correct),
        Tool("memory.link", "Connect two notes (e.g. an assignment and its template).",
             {"a": P(), "b": P(), "rel": P(required=False, default="related")}, Risk.WRITE, mem_link),
        Tool("memory.attach_file", "Record a file (with fingerprint) as the source of a note.",
             {"note": P(), "path": P()}, Risk.WRITE, mem_attach),
        Tool("memory.forget", "Retract a fact (moves to history). Needs owner approval.",
             {"note": P(), "fact": P(desc="fact #id")}, Risk.DESTRUCTIVE, mem_forget),
    ]

"""
Deterministic memory retrieval (no embeddings, no model calls, same query -> same result).

1. BM25 over weighted fields (title, id, summary, file names, facts)
2. graph expansion: neighbours (links, parent, children) of the best hits get a share of the score
3. fact-level selection inside each note, so the model sees the relevant lines, not whole files
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .store import MemoryStore, Note, norm

STOP = set("""the a an and or of to in on at for with is are was were be been it this that these those my your
our his her their i you he she we they me him them do does did have has had not no yes so as by from but if
about what which who whom when where why how can could would should will just please tell remember know
что это как мне мой моя мои мое ты вы он она они мы и в на с по для не да но или там тут""".split())


def stem(w: str) -> str:
    w = w.replace("our", "or") if w.isascii() else w          # colour -> color, favourite -> favorite
    return w[:6] if w.isascii() else w[:5]


def tokenize(text: str) -> list:
    out = []
    for w in re.findall(r"[^\W_]+", str(text).lower(), flags=re.U):
        if len(w) < 2 or w in STOP:
            continue
        out.append(stem(w))
    return out


@dataclass
class Hit:
    note_id: str
    score: float
    via: str = "match"                  # match | link:<id> | recent
    facts: list = field(default_factory=list)   # selected Fact objects


class Index:
    K1, B = 1.4, 0.75
    W = {"title": 4.0, "id": 3.0, "summary": 3.0, "file": 2.5, "link": 1.0, "fact": 1.0}

    def __init__(self, store: MemoryStore):
        self.store = store
        self._sig = None
        self._docs: dict = {}
        self._type: dict = {}
        self._idf: dict = {}
        self._avg = 1.0

    def refresh(self) -> None:
        sig = self.store.signature()
        if sig == self._sig:
            return
        self._sig = sig
        self._docs = {}
        self._type = {}
        df: Counter = Counter()
        for n in self.store.all_notes():
            tf: Counter = Counter()
            for t in tokenize(n.title):
                tf[t] += self.W["title"]
            for t in tokenize(n.id.replace("-", " ")):
                tf[t] += self.W["id"]
            for t in tokenize(n.summary):
                tf[t] += self.W["summary"]
            for path, _, _ in n.files:
                for t in tokenize(re.sub(r"[\\/_.\-]", " ", path.replace("\\", "/").split("/")[-1])):
                    tf[t] += self.W["file"]
            for _, rel, blurb in n.links:
                for t in tokenize(blurb):
                    tf[t] += self.W["link"] * 0.5
            for f in n.facts:
                for t in tokenize(f.text):
                    tf[t] += self.W["fact"]
            self._docs[n.id] = (tf, sum(tf.values()) or 1.0)
            self._type[n.id] = n.type
            df.update(tf.keys())
        N = max(len(self._docs), 1)
        self._avg = sum(l for _, l in self._docs.values()) / N if self._docs else 1.0
        self._idf = {t: math.log(1 + (N - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _bm25(self, qtokens: list, nid: str) -> float:
        tf, length = self._docs[nid]
        s = 0.0
        for t in set(qtokens):
            f = tf.get(t, 0.0)
            if f:
                s += self._idf.get(t, 0.0) * (f * (self.K1 + 1)) / (f + self.K1 * (1 - self.B + self.B * length / self._avg))
        return s

    def search(self, query: str, limit: int = 4, expand: bool = True, fallback_recent: int = 2) -> list:
        self.refresh()
        q = tokenize(query)
        scores = {}
        for nid in self._docs:
            if nid == "core":
                continue
            sc = self._bm25(q, nid) if q else 0.0
            if self._type.get(nid) == "episodic":
                sc *= 0.7                      # daily logs rank below real facts
            if sc > 0:
                scores[nid] = sc
        hits = {nid: Hit(nid, sc, "match") for nid, sc in scores.items()}
        if expand and hits:
            seeds = sorted(hits.values(), key=lambda h: (-h.score, h.note_id))[:3]
            for seed in seeds:
                note = self.store.get(seed.note_id, copy_=False)
                neighbours = {l[0] for l in note.links} | ({note.parent} if note.parent else set()) \
                    | {c.id for c in self.store.all_notes() if c.parent == note.id}
                for nb in neighbours:
                    if nb in ("core",) or nb not in self._docs:
                        continue
                    share = seed.score * 0.3
                    if nb not in hits or hits[nb].score < share:
                        hits[nb] = Hit(nb, share, f"link:{seed.note_id}")
        ranked = sorted(hits.values(), key=lambda h: (-h.score, h.note_id))[:limit]
        if not ranked and fallback_recent:
            recent = sorted((n for n in self.store.all_notes() if n.id != "core" and n.type != "episodic"),
                            key=lambda n: n.updated, reverse=True)[:fallback_recent]
            ranked = [Hit(n.id, 0.0, "recent") for n in recent]
        for h in ranked:
            note = self.store.get(h.note_id, copy_=False)
            h.facts = self._pick_facts(note, q)
        return ranked

    def _pick_facts(self, note: Note, q: list, cap: int = 7) -> list:
        qs = set(q)

        def fscore(f):
            return sum(self._idf.get(t, 1.0) for t in set(tokenize(f.text)) & qs)

        scored = sorted(note.facts, key=lambda f: (-fscore(f), f.date, f.fid))
        matching = [f for f in scored if fscore(f) > 0][:cap]
        if len(matching) < cap:
            newest = sorted((f for f in note.facts if f not in matching), key=lambda f: f.date, reverse=True)
            matching += newest[:min(3, cap - len(matching))]
        return matching

    def context(self, query: str, budget: int = 4500, limit: int = 4) -> str:
        """Text block for the prompt: core + the most relevant notes/facts, within a budget."""
        parts, used = [], 0
        core = self.store.get("core", copy_=False)
        if core:
            lines = [f"[core] {core.summary}"] + [f"  - {f.text}" for f in core.facts[:12]]
            block = "\n".join(lines)[:1500]
            parts.append(block)
            used += len(block)
        for h in self.search(query, limit=limit):
            note = self.store.get(h.note_id, copy_=False)
            head = f"[{note.id}] ({note.type}) {note.title} - {note.summary}"
            lines = [head]
            for f in h.facts:
                tag = " (unverified)" if f.conf == "inferred" else ""
                lines.append(f"  - {f.text}{tag}  #{f.fid}")
            if note.files:
                lines.append("  files: " + "; ".join(p for p, _, _ in note.files[:3]))
            rel = [f"{t}({r})" for t, r, _ in note.links[:5]]
            if rel:
                lines.append("  links: " + ", ".join(rel))
            block = "\n".join(lines)
            if used + len(block) > budget and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts) or "(no notes yet)"

    def explain(self, query: str) -> str:
        hits = self.search(query, limit=8)
        rows = [f"{h.note_id:<28} score={h.score:6.2f} via={h.via} facts={len(h.facts)}" for h in hits]
        return "\n".join(rows) or "(no matches)"

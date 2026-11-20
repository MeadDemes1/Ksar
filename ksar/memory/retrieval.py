"""
Deterministic memory retrieval (no embeddings, no model calls: same memory + same query = same result).

Three levels, all answered from RAM (nothing here touches the disk once the index is current):
  LEVEL 1  CORE    the small central note, cached by the store (core_digest)
  LEVEL 2  BRANCH  which branch/index notes does the request point at?  (BM25 over their titles and summaries,
                   which list their children)
  LEVEL 3  NOTES   BM25 over the detailed notes, boosted inside the matching branch; then a bounded one-step walk
                   along meaningful links (explains / caused_by / originated_from ...); then fact-level selection
Restrictions (facts tagged #rule / #restriction ...) are checked separately and ALWAYS surface when a request
touches what they forbid or permit, whatever the rest of the ranking says.

The index is updated incrementally from the store's change log: a message never triggers a scan of the vault.
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .store import MemoryStore, Note, fact_tags, norm, strip_tags

STOP = set("""the a an and or of to in on at for with is are was were be been it this that these those my your
our his her their i you he she we they me him them do does did have has had not no yes so as by from but if
about what which who whom when where why how can could would should will just please tell remember know
что это как мне мой моя мои мое ты вы он она они мы и в на с по для не да но или там тут""".split())

RULE_TAGS = {"rule", "restriction", "prohibition", "permission", "boundary"}
RELS_FOLLOW = ("explains", "caused_by", "originated_from", "belongs_to", "supersedes", "contradicts", "derived_from")
CONF_W = {"verified": 1.15, "stated": 1.0, "inferred": 0.75}
TIME_RE = re.compile(r"\b(today|yesterday|recent(?:ly)?|lately|latest|earlier|last (?:week|month|time|night)|"
                     r"this (?:week|month|morning))\b|сегодня|вчера|недавно|последн", re.I)
STRONG_IDF = 0.8


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


# words that make a rule sound like a rule but say nothing about WHAT it covers
RULE_GENERIC = {stem(w) for w in ("never always without asking ask first must should shall not dont don't do allowed allow "
                                  "permission permissions only unless explicit explicitly any anything owner user "
                                  "ever need needs before without confirm confirmation").split()}


@dataclass
class Hit:
    note_id: str
    score: float
    via: str = "match"                  # match | link:<id> | recent | branch:<id>
    facts: list = field(default_factory=list)   # selected Fact objects


@dataclass
class Block:
    note_id: str
    level: str                           # rule | branch | detail | link
    score: float
    reason: str
    facts: list = field(default_factory=list)


@dataclass
class Recall:
    query: str = ""
    core: str = ""
    branches: list = field(default_factory=list)
    blocks: list = field(default_factory=list)       # Block, best first
    rules: list = field(default_factory=list)        # Block(level="rule")
    matched: list = field(default_factory=list)      # query terms found in the best note
    worthwhile: bool = False                          # strong enough to inject without being asked
    why: str = ""
    index: Optional["Index"] = field(default=None, repr=False)

    def text(self, budget: int = 1800, include_core: bool = False, core_chars: int = 600) -> str:
        parts, used = [], 0
        ix = self.index
        shown = {f.fid for b in self.rules for f in b.facts if f.fid}      # a rule is not repeated in its note's block
        if self.rules:
            lines = ["RULES from the owner that apply here (follow them; if unsure, ask first):"]
            for b in self.rules:
                for f in b.facts:
                    lines.append(f"- {strip_tags(f.text)}  [{b.note_id}]")
            parts.append("\n".join(lines))
            used += len(parts[-1])
        if include_core and self.core:
            parts.append(self.core[:core_chars])
            used += len(parts[-1])
        for b in self.blocks:
            note = ix.store.peek(b.note_id) if ix else None
            if not note:
                continue
            head = f"[{note.id}] ({note.type}) {note.title} - {note.summary}"
            lines = [head]
            for f in b.facts:
                if f.fid and f.fid in shown:
                    continue
                tag = " (unverified)" if f.conf == "inferred" else ""
                lines.append(f"  - {strip_tags(f.text)}{tag}  #{f.fid}")
            if len(lines) == 1 and b.level in ("branch", "detail") and not b.facts:
                continue
            if len(lines) == 1 and b.level == "branch":
                kids = sorted(ix._kids.get(note.id, ()))[:6]
                if kids:
                    lines.append("  contains: " + ", ".join(kids))
            block = "\n".join(lines)
            if used + len(block) > budget and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def explain(self) -> str:
        rows = [f"query terms matched: {', '.join(self.matched) or '-'}   worthwhile: {self.worthwhile}   ({self.why})"]
        if self.branches:
            rows.append("branches:    " + ", ".join(self.branches))
        for b in self.rules:
            rows.append(f"rule         {b.note_id:<28} {b.reason}")
        for b in self.blocks:
            rows.append(f"{b.level:<12} {b.note_id:<28} score={b.score:6.2f}  {b.reason}  facts={len(b.facts)}")
        return "\n".join(rows)


class Index:
    K1, B = 1.4, 0.75
    W = {"title": 4.0, "id": 3.0, "summary": 3.0, "file": 2.5, "link": 1.0, "fact": 1.0}

    def __init__(self, store: MemoryStore):
        self.store = store
        self._lock = threading.RLock()
        self._gen = -1
        self.full_rebuilds = 0
        self.incremental_updates = 0
        self._reset()

    def _reset(self) -> None:
        self._docs: dict = {}               # id -> (term frequencies, length)
        self._type: dict = {}
        self._parent: dict = {}
        self._links: dict = {}              # id -> [(target, relation)]
        self._kids: dict = {}               # id -> set of child ids
        self._back: dict = {}               # id -> set of ids linking to it
        self._upd: dict = {}
        self._meta: dict = {}               # id -> tokens of title/id/summary
        self._rules: dict = {}              # id -> [(Fact, token set)]
        self._df: Counter = Counter()
        self._idf: dict = {}
        self._avg = 1.0

    # ------------------------------------------------------------------ incremental maintenance
    def _drop(self, nid: str) -> None:
        doc = self._docs.pop(nid, None)
        if doc is not None:
            self._df.subtract(doc[0].keys())
        self._type.pop(nid, None)
        par = self._parent.pop(nid, None)
        if par is not None and par in self._kids:
            self._kids[par].discard(nid)
        for t, _ in self._links.pop(nid, []):
            if t in self._back:
                self._back[t].discard(nid)
        self._upd.pop(nid, None)
        self._meta.pop(nid, None)
        self._rules.pop(nid, None)

    def _add(self, n: Note) -> None:
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
        rules = []
        for f in n.facts:
            toks = tokenize(strip_tags(f.text))
            for t in toks:
                tf[t] += self.W["fact"]
            if n.id != "core" and fact_tags(f.text) & RULE_TAGS:      # the core only POINTS at rules; the note holds them
                rules.append((f, {t for t in toks if t not in RULE_GENERIC}))
        self._docs[n.id] = (tf, sum(tf.values()) or 1.0)
        self._df.update(tf.keys())
        self._type[n.id] = n.type
        self._parent[n.id] = n.parent
        if n.parent:
            self._kids.setdefault(n.parent, set()).add(n.id)
        self._links[n.id] = [(t, r) for t, r, _ in n.links]
        for t, _ in self._links[n.id]:
            self._back.setdefault(t, set()).add(n.id)
        self._upd[n.id] = n.updated
        self._meta[n.id] = set(tokenize(n.title + " " + n.id.replace("-", " ") + " " + n.summary))
        if rules:
            self._rules[n.id] = rules

    def refresh(self) -> None:
        """Brings the index up to date with the store. No disk access when nothing changed (the normal case)."""
        st = self.store
        if self._gen == st.gen:
            return
        with self._lock:
            if self._gen == st.gen:
                return
            gen = st.gen
            changed = st.changed_since(self._gen) if self._gen >= 0 else None
            if changed is None:
                self._reset()
                ids = st.known_ids()
                self.full_rebuilds += 1
            else:
                ids = sorted(changed)
                self.incremental_updates += 1
            for nid in ids:
                self._drop(nid)
                if st.has(nid):
                    note = st.peek(nid)
                    if note:
                        self._add(note)
            N = max(len(self._docs), 1)
            self._df = Counter({t: c for t, c in self._df.items() if c > 0})
            self._avg = sum(l for _, l in self._docs.values()) / N if self._docs else 1.0
            self._idf = {t: math.log(1 + (N - c + 0.5) / (c + 0.5)) for t, c in self._df.items()}
            self._gen = gen

    # ------------------------------------------------------------------ scoring
    def _bm25(self, qtokens: list, nid: str) -> float:
        tf, length = self._docs[nid]
        s = 0.0
        for t in set(qtokens):
            f = tf.get(t, 0.0)
            if f:
                s += self._idf.get(t, 0.0) * (f * (self.K1 + 1)) / (f + self.K1 * (1 - self.B + self.B * length / self._avg))
        return s

    def _recent(self, nid: str, days: int = 7) -> bool:
        try:
            return datetime.fromisoformat(self._upd.get(nid) or "") >= datetime.now() - timedelta(days=days)
        except ValueError:
            return False

    def _pick_facts(self, note: Note, q: list, cap: int = 7) -> list:
        qs = set(q)

        def fscore(f):
            base = sum(self._idf.get(t, 1.0) for t in set(tokenize(strip_tags(f.text))) & qs)
            return base * CONF_W.get(f.conf, 1.0)

        scored = sorted(note.facts, key=lambda f: (-fscore(f), f.date, f.fid))
        matching = [f for f in scored if fscore(f) > 0][:cap]
        if len(matching) < cap:
            newest = sorted((f for f in note.facts if f not in matching), key=lambda f: f.date, reverse=True)
            matching += newest[:min(3, cap - len(matching))]
        return matching

    # ------------------------------------------------------------------ the API
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
                neighbours = {t for t, _ in self._links.get(seed.note_id, [])} | self._kids.get(seed.note_id, set())
                if self._parent.get(seed.note_id):
                    neighbours.add(self._parent[seed.note_id])
                for nb in neighbours:
                    if nb == "core" or nb not in self._docs:
                        continue
                    share = seed.score * 0.3
                    if nb not in hits or hits[nb].score < share:
                        hits[nb] = Hit(nb, share, f"link:{seed.note_id}")
        ranked = sorted(hits.values(), key=lambda h: (-h.score, h.note_id))[:limit]
        if not ranked and fallback_recent:
            recent = sorted((i for i in self._docs if i != "core" and self._type[i] != "episodic"),
                            key=lambda i: self._upd.get(i, ""), reverse=True)[:fallback_recent]
            ranked = [Hit(i, 0.0, "recent") for i in recent]
        for h in ranked:
            note = self.store.peek(h.note_id)
            if note:
                h.facts = self._pick_facts(note, q)
        return ranked

    def recall(self, query: str, limit: int = 4, expand: int = 3) -> Recall:
        """Three-level retrieval for one request. Pure RAM work, deterministic, bounded."""
        self.refresh()
        rc = Recall(query=query, index=self, core=self.store.core_digest(1500))
        q = tokenize(query)
        qset = set(q)
        if not self._docs or not q:
            rc.why = "empty memory or nothing to match"
            return rc
        timey = bool(TIME_RE.search(query))
        raw = {nid: self._bm25(q, nid) for nid in self._docs if nid != "core"}

        # LEVEL 2: which branches does the request point at?
        br = sorted(((sc, nid) for nid, sc in raw.items() if sc > 0 and self._type[nid] == "index"),
                    key=lambda x: (-x[0], x[1]))[:2]
        rc.branches = [nid for _, nid in br]
        in_branch = set(rc.branches)

        def under_branch(nid: str) -> bool:
            cur, hops = self._parent.get(nid), 0
            while cur and hops < 3:
                if cur in in_branch:
                    return True
                cur, hops = self._parent.get(cur), hops + 1
            return False

        # LEVEL 3: detailed notes
        scores = {}
        for nid, sc in raw.items():
            if sc <= 0:
                continue
            w = 0.7 if self._type[nid] == "episodic" else (0.6 if self._type[nid] == "index" else 1.0)
            if under_branch(nid):
                w *= 1.2
            if timey and self._recent(nid):
                w *= 1.3
            scores[nid] = sc * w
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        detail = [(nid, sc) for nid, sc in ranked if self._type[nid] != "index"][:limit]
        blocks = []
        for nid, sc in detail:
            note = self.store.peek(nid)
            if note:
                terms = [t for t in qset if self._docs[nid][0].get(t)]
                blocks.append(Block(nid, "detail", sc, "matches: " + ", ".join(sorted(terms)),
                                    self._pick_facts(note, q)))

        # restrictions that touch this request
        rule_notes = []
        for nid in sorted(self._rules):
            hit_facts, hit_terms = [], set()
            for f, toks in self._rules[nid]:
                ov = qset & toks
                if ov:
                    hit_facts.append(f)
                    hit_terms |= ov
            if hit_facts:
                rc.rules.append(Block(nid, "rule", 10.0 + len(hit_terms), "rule matches: " + ", ".join(sorted(hit_terms)),
                                      hit_facts))
                rule_notes.append(nid)

        # bounded one-step walk along meaningful links (seeds: best two notes and the notes holding matching rules)
        have = {b.note_id for b in blocks} | set(rule_notes)
        extra = 0
        seeds = [b.note_id for b in blocks[:2]] + rule_notes[:2]
        for sid in seeds:
            for t, rel in self._links.get(sid, []):
                if extra >= expand or t in have or t == "core" or t not in self._docs or rel not in RELS_FOLLOW:
                    continue
                note = self.store.peek(t)
                if note:
                    base = scores.get(sid, 3.0)
                    blocks.append(Block(t, "link", base * 0.35, f"{rel} of {sid}", self._pick_facts(note, q, cap=4)))
                    have.add(t)
                    extra += 1
        # only a branch matched: show what it contains (newest children first)
        if not detail and rc.branches:
            bid = rc.branches[0]
            kids = sorted(self._kids.get(bid, ()), key=lambda i: self._upd.get(i, ""), reverse=True)[:3]
            for k in kids:
                note = self.store.peek(k)
                if note and k not in have:
                    blocks.append(Block(k, "branch", raw.get(bid, 0) * 0.5, f"inside branch {bid}", self._pick_facts(note, q, cap=4)))
                    have.add(k)
        rc.blocks = sorted(blocks, key=lambda b: (-b.score, b.note_id))[:limit + expand]

        # is this strong enough to use without being asked?
        if detail:
            top = detail[0][0]
            tf = self._docs[top][0]
            matched = sorted(t for t in qset if tf.get(t))
            rc.matched = matched
            strong = [t for t in matched if self._idf.get(t, 0) >= STRONG_IDF]
            title_hit = any(t in self._meta[top] for t in strong)
            max_idf = max((self._idf.get(t, 0) for t in strong), default=0)
            rc.worthwhile = len(strong) >= 2 or (len(strong) == 1 and (title_hit or max_idf >= 1.6))
            rc.why = f"{len(strong)} strong term(s)" + (", in the title/summary" if title_hit else "")
        elif rc.branches:
            rc.worthwhile = False
            rc.why = "only a branch overview matched"
        else:
            rc.why = "nothing in memory matches"
        if rc.rules:
            rc.worthwhile = True
            rc.why = "a stored rule applies" + (f"; {rc.why}" if rc.why else "")
        return rc

    def context(self, query: str, budget: int = 4500, limit: int = 4, include_core: bool = True) -> str:
        """Text block for the prompt: core + the most relevant notes/facts, within a budget."""
        parts, used = [], 0
        core = self.store.peek("core") if include_core else None
        if core:
            lines = [f"[core] {core.summary}"] + [f"  - {strip_tags(f.text)}" for f in core.facts[:12]]
            block = "\n".join(lines)[:1500]
            parts.append(block)
            used += len(block)
        for h in self.search(query, limit=limit):
            note = self.store.peek(h.note_id)
            if not note:
                continue
            head = f"[{note.id}] ({note.type}) {note.title} - {note.summary}"
            lines = [head]
            for f in h.facts:
                tag = " (unverified)" if f.conf == "inferred" else ""
                lines.append(f"  - {strip_tags(f.text)}{tag}  #{f.fid}")
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
        return "\n\n".join(parts) or ("(no notes yet)" if include_core else "")

    def explain(self, query: str) -> str:
        hits = self.search(query, limit=8)
        rows = [f"{h.note_id:<28} score={h.score:6.2f} via={h.via} facts={len(h.facts)}" for h in hits]
        return "\n".join(rows) or "(no matches)"

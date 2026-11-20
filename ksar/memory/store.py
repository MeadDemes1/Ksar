"""
Markdown memory store.

One note = one .md file (human readable/editable, Obsidian-compatible):

    ---
    id: lab-3
    title: Lab 3
    type: project            core | index | semantic | episodic | project | person | task | document
    summary: one line
    parent: school           branch this note hangs under
    ---
    ## Facts
    - Deadline: 2026-11-20 <!--f:a1b2c3 c:stated d:2026-11-13 s:chat:2026-11-13#4-->
    ## Links
    - [[lab-3-template]] (derived_from) - blurb
    ## Files
    - C:\\path\\lab3.pdf | sha256:ab12... | 2026-11-13
    ## History
    - ~~Deadline: 2026-11-18~~ (superseded 2026-11-14) <!--...-->

Guarantees: atomic writes, a snapshot before every change, soft deletes, facts are never
silently overwritten (they move to History), provenance + confidence on every fact.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

NOTE_TYPES = ("core", "index", "semantic", "episodic", "project", "person", "task", "document")
CONFIDENCE = ("inferred", "stated", "verified")      # ascending trust
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
            *(f"lpt{i}" for i in range(1, 10))}
KNOWN_SECTIONS = ("facts", "links", "files", "history")


# ----------------------------------------------------------------- helpers
def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def one_line(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def safe_id(name) -> str:
    n = str(name or "").lower().strip()
    n = re.sub(r"\.md$", "", n)
    n = re.sub(r"[^\w\- ]", "", n, flags=re.U).strip().replace(" ", "-")
    n = re.sub(r"-+", "-", n).strip("-_")[:60]
    if n in RESERVED:
        n += "-note"
    return n


TAG_RE = re.compile(r"(?<![\w#])#([^\W\d][\w\-]*)", re.U)


def fact_tags(text: str) -> set:
    """Obsidian-style #tags written in a fact, e.g. '#rule' (lower-case, without '#')."""
    return {t.lower() for t in TAG_RE.findall(str(text))}


def strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", str(text))).strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", strip_tags(text).lower(), flags=re.U)).strip()


def safe_folder(folder) -> str:
    """Relative folder inside notes/ ('' = root): at most 3 levels, no dots/drive letters/illegal characters."""
    parts = []
    for seg in re.split(r"[\\/]+", str(folder or "")):
        seg = re.sub(r'[<>:"|?*\x00-\x1f]', "", seg).strip().strip(".")
        if seg and seg not in (".", "..") and not seg.startswith("."):
            parts.append(seg[:40])
    return "/".join(parts[:3])


def jaccard(a: str, b: str) -> float:
    x, y = set(norm(a).split()), set(norm(b).split())
    return len(x & y) / len(x | y) if x and y else 0.0


def split_kv(text: str):
    """'Favourite colour: dark blue' -> ('Favourite colour', 'dark blue'); else (None, None)."""
    m = re.match(r"^([^:\d][^:]{1,59}):\s+(.+)$", text.strip())
    if m and len(m.group(1).split()) <= 7:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def clean_fact(text, limit=300) -> str:
    t = one_line(str(text).replace("<!--", "").replace("-->", "").replace("```", ""))
    t = re.sub(r"^[-*]\s*", "", t)
    return t[:limit]


def clean_src(src) -> str:
    return re.sub(r"[\s<>]|-->", "_", str(src or ""))[:80]


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path, limit_bytes: int = 500 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit_bytes:
                raise ValueError("file too large to fingerprint")
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- model
@dataclass
class Fact:
    text: str
    fid: str = ""
    conf: str = "stated"
    src: str = ""
    date: str = ""
    reason: str = ""      # only for History entries

    def line(self, history: bool = False) -> str:
        meta = f"f:{self.fid} c:{self.conf} d:{self.date}" + (f" s:{self.src}" if self.src else "")
        if history:
            return f"- ~~{self.text}~~" + (f" ({self.reason})" if self.reason else "") + f" <!--{meta}-->"
        return f"- {self.text} <!--{meta}-->"


@dataclass
class FactResult:
    status: str                     # added | duplicate | superseded | conflict | rejected
    fact: Optional[Fact] = None
    old: Optional[Fact] = None
    note: str = ""


@dataclass
class Note:
    id: str
    title: str = ""
    type: str = "semantic"
    summary: str = ""
    parent: str = ""
    status: str = "active"
    created: str = ""
    updated: str = ""
    facts: list = field(default_factory=list)
    history: list = field(default_factory=list)
    links: list = field(default_factory=list)    # (target, rel, blurb)
    files: list = field(default_factory=list)    # (path, sha256, date)
    extra: dict = field(default_factory=dict)    # unknown sections, preserved verbatim
    id_mismatch: bool = False
    rev: str = ""                                # file revision (mtime+size) this object was read from; not saved
    intro: list = field(default_factory=list)    # free text the owner wrote under the title, preserved verbatim
    props: dict = field(default_factory=dict)    # front-matter keys Ksar does not use (aliases, tags, ...): raw lines, preserved


@dataclass
class Issue:
    kind: str
    note: str
    detail: str
    fixed: bool = False


# ----------------------------------------------------------------- parsing
def parse_fact(line: str, history: bool = False) -> Optional[Fact]:
    s = line.strip()
    if not s.startswith(("-", "*")):
        return None
    s = s[1:].strip()
    meta = {}
    m = re.search(r"<!--(.*?)-->\s*$", s)
    if m:
        for tok in m.group(1).split():
            k, _, v = tok.partition(":")
            if k in ("f", "c", "d", "s"):
                meta[k] = v
        s = s[:m.start()].strip()
    reason = ""
    if history:
        mm = re.match(r"^~~(.*)~~(?:\s*\((.*)\))?$", s)
        if mm:
            s, reason = mm.group(1), mm.group(2) or ""
    if not s:
        return None
    return Fact(s, meta.get("f", ""), meta.get("c", "stated"), meta.get("s", ""), meta.get("d", ""), reason)


LINK_RE = re.compile(r"^[-*]\s*\[\[([^\]]+)\]\](?:\s*\(([^)]*)\))?(?:\s*[-:\u2014]\s*(.*))?$")


_FM_KEY = re.compile(r"^([A-Za-z_][\w\- ]*):(.*)$")


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] == '"':
        return v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if len(v) >= 2 and v[0] == v[-1] == "'":
        return v[1:-1].replace("''", "'")
    return v


def yaml_scalar(v) -> str:
    """A front-matter value that is valid YAML (and so valid for Obsidian Properties): values containing ': ', ' #',
    leading indicators, or that look like numbers/booleans are double-quoted."""
    v = one_line(v)
    if not v:
        return ""
    risky = (re.search(r":(\s|$)| #", v) or re.match(r"^[\[\]{}&*!|>'\"%@`#?,-]", v) or re.fullmatch(r"[\d.]+", v)
             or v.lower() in ("null", "true", "false", "yes", "no", "~", "on", "off"))
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"' if risky else v


def split_frontmatter_ex(text: str):
    """(meta, raw property blocks, body). meta = known single-line values (unquoted). raw = every key's original lines,
    including the indented '- item' lines Obsidian writes for lists (tags, aliases), so nothing the owner added is lost."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S) if text.startswith("---") else None
    if not m:
        return {}, {}, text
    meta, raw, cur = {}, {}, None
    for line in m.group(1).splitlines():
        km = _FM_KEY.match(line) if line and not line[0].isspace() else None
        if km:
            cur = km.group(1).strip().lower()
            meta[cur] = _unquote(km.group(2))
            raw[cur] = [line]
        elif cur is not None:
            raw[cur].append(line)
    return meta, raw, m.group(2)


def split_frontmatter(text: str):
    meta, _, body = split_frontmatter_ex(text)
    return meta, body


def parse_note(text: str, fallback_id: str) -> Note:
    meta, raw_props, body = split_frontmatter_ex(text)
    legacy = not meta
    n = Note(id=safe_id(meta.get("id")) or fallback_id, title=meta.get("title", ""),
             type=meta.get("type", "semantic"), summary=meta.get("summary", ""),
             parent=safe_id(meta.get("parent")), status=meta.get("status", "active") or "active",
             created=meta.get("created", ""), updated=meta.get("updated", ""))
    if n.type not in NOTE_TYPES:
        n.type = "semantic"
    if not legacy:
        n.props = {k: v for k, v in raw_props.items() if k not in ("id", "title", "type", "summary", "parent", "status",
                                                                   "created", "updated")}
    sections: dict = {}
    key = "_intro"
    for line in body.splitlines():
        if line.startswith("# ") and not n.title:
            n.title = line[2:].strip()
            continue
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            key = line[3:].strip()
            sections.setdefault(key, [])
            continue
        sections.setdefault(key, []).append(line)
    if not legacy:
        intro = sections.pop("_intro", [])
        while intro and not intro[0].strip():
            intro.pop(0)
        while intro and not intro[-1].strip():
            intro.pop()
        n.intro = intro
    if legacy:                                   # old flat notes: bullets become 'inferred' facts
        for line in sections.get("_intro", []):
            m = re.match(r"^summary:\s*(.+)$", line, re.I)
            if m:
                n.summary = m.group(1).strip()
        for lines in sections.values():
            for line in lines:
                f = parse_fact(line)
                if f and not LINK_RE.match(line.strip()):
                    f.conf, f.src = "inferred", "legacy"
                    n.facts.append(f)
        sections = {}
    lower = {k.lower(): k for k in sections}
    for name in KNOWN_SECTIONS:
        lines = sections.pop(lower.get(name, ""), []) if name in lower else []
        for line in lines:
            if name == "facts":
                f = parse_fact(line)
                if f:
                    n.facts.append(f)
                elif line.strip():                      # free text the owner typed under Facts: keep it, do not drop it
                    sections.setdefault("Notes", []).append(line)
            elif name == "history":
                f = parse_fact(line, history=True)
                if f:
                    n.history.append(f)
            elif name == "links":
                m = LINK_RE.match(line.strip())
                if m:
                    n.links.append((safe_id(m.group(1)), (m.group(2) or "related").strip(), (m.group(3) or "").strip()))
            elif name == "files":
                parts = [p.strip() for p in line.strip().lstrip("-* ").split(" | ")]
                if parts and parts[0]:
                    sha = parts[1].replace("sha256:", "") if len(parts) > 1 else ""
                    n.files.append((parts[0], sha, parts[2] if len(parts) > 2 else ""))
    n.extra = {k: [l for l in v] for k, v in sections.items() if any(l.strip() for l in v)}
    if not n.title:
        n.title = n.id.replace("-", " ").title()
    return n


def render_note(n: Note) -> str:
    out = ["---", f"id: {n.id}", f"title: {yaml_scalar(n.title)}", f"type: {n.type}",
           f"summary: {yaml_scalar(n.summary)}", f"parent: {n.parent}", f"status: {n.status}",
           f"created: {n.created}", f"updated: {n.updated}"]
    for block in n.props.values():
        out += list(block)
    out += ["---", f"# {one_line(n.title)}", ""]
    if n.intro:
        out += list(n.intro) + [""]
    out += ["## Facts"]
    out += [f.line() for f in n.facts]
    if n.links:
        out += ["", "## Links"]
        out += [f"- [[{t}]] ({r})" + (f" - {b}" if b else "") for t, r, b in n.links]
    if n.files:
        out += ["", "## Files"]
        out += [f"- {p} | sha256:{h} | {d}" for p, h, d in n.files]
    if n.history:
        out += ["", "## History"]
        out += [f.line(history=True) for f in n.history]
    for name, lines in n.extra.items():
        out += ["", f"## {name}"] + list(lines)
    return "\n".join(out).rstrip() + "\n"


# ----------------------------------------------------------------- store
class MemoryStore:
    def __init__(self, root: Path, keep_versions: int = 40, max_fact_chars: int = 300):
        self.root = Path(root)
        self.notes_dir = self.root / "notes"
        self.hist = self.root / ".history"
        self.trash = self.root / ".trash"
        self.journal_dir = self.root / "journal"
        self.keep_versions = keep_versions
        self.max_fact_chars = max_fact_chars
        for d in (self.notes_dir, self.hist, self.trash, self.journal_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._cache: dict = {}
        # Thread safety: ONE small re-entrant lock around single store operations (file write + snapshot, a few ms).
        # Nothing slow (model calls, searches over many notes) ever runs while it is held.
        self._lock = threading.RLock()
        self._paths: dict = {}              # note id -> file path (notes may live in sub-folders)
        self.gen = 0                        # bumped on every change; lets the search index update incrementally
        self._changes: list = []            # (gen, note id), bounded; see changed_since()
        self._floor = 0
        self._core_cache = None
        self.core_builds = 0                # how often the core digest was really rebuilt (tests / /memory status)
        self.scan()

    # ---- change tracking, registry and revisions
    def _touch(self, nid: str) -> None:
        self.gen += 1
        self._changes.append((self.gen, nid))
        if len(self._changes) > 4096:
            drop = self._changes[:2048]
            self._changes = self._changes[2048:]
            self._floor = drop[-1][0]
        if nid == "core":
            self._core_cache = None

    def changed_since(self, gen: int):
        """Ids changed after generation `gen`, or None when the log no longer reaches back that far (rebuild)."""
        if gen < self._floor:
            return None
        return {nid for g, nid in self._changes if g > gen}

    def scan(self) -> bool:
        """Rebuild the id -> file registry from disk and notice edits made outside Ksar (Obsidian, an editor).
        Never called on the chat path; the idle worker and commands call it. True when something changed."""
        with self._lock:
            found: dict = {}
            for dirpath, dirnames, filenames in os.walk(self.notes_dir):
                dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                for fn in sorted(filenames):
                    if fn.endswith(".md"):
                        nid = fn[:-3]
                        if safe_id(nid) == nid and nid not in found:
                            found[nid] = Path(dirpath) / fn
            changed = set(self._paths) ^ set(found)
            for nid, p in found.items():
                if self._paths.get(nid) not in (None, p):
                    changed.add(nid)
                hit = self._cache.get(nid)
                if hit is not None:
                    try:
                        if hit[0] != p.stat().st_mtime_ns:
                            changed.add(nid)
                    except OSError:
                        changed.add(nid)
            self._paths = found
            for nid in changed:
                self._cache.pop(nid, None)
                self._touch(nid)
            return bool(changed)

    def known_ids(self) -> list:
        """Ids from the in-memory registry: no disk access (use ids() for a fresh scan). list() of a dict is atomic in
        CPython, so another thread adding a note cannot break this iteration."""
        return sorted(list(self._paths))

    def has(self, nid: str) -> bool:
        return nid in self._paths

    def path(self, nid: str) -> Path:
        return self._paths.get(nid) or self.notes_dir / f"{nid}.md"

    def exists(self, nid: str) -> bool:
        nid = safe_id(nid)
        return bool(nid) and self.path(nid).is_file()

    def ids(self) -> list:
        self.scan()
        return sorted(self._paths)

    def signature(self) -> tuple:
        return tuple((nid, p.stat().st_mtime_ns) for nid, p in sorted(self._paths.items()) if p.exists())

    def folder_of(self, nid: str) -> str:
        p = self._paths.get(nid)
        if not p:
            return ""
        try:
            rel = p.parent.relative_to(self.notes_dir)
        except ValueError:
            return ""
        return "" if str(rel) == "." else rel.as_posix()

    def get(self, nid: str, copy_: bool = True) -> Optional[Note]:
        nid = safe_id(nid)
        p = self.path(nid)
        if not nid or not p.is_file():
            return None
        with self._lock:
            st = p.stat()
            mt = st.st_mtime_ns
            hit = self._cache.get(nid)
            if not hit or hit[0] != mt:
                note = parse_note(p.read_text(encoding="utf-8", errors="replace"), nid)
                if note.id != nid:
                    note.id, note.id_mismatch = nid, True
                note.rev = f"{mt}-{st.st_size}"
                hit = (mt, note)
                self._paths.setdefault(nid, p)
                self._cache[nid] = hit
            return copy.deepcopy(hit[1]) if copy_ else hit[1]

    def peek(self, nid: str) -> Optional[Note]:
        """Cached note WITHOUT touching the disk when it is already in memory (chat path). Do not modify the result."""
        hit = self._cache.get(nid)
        return hit[1] if hit else self.get(nid, copy_=False)

    def rev(self, nid: str) -> str:
        """Current revision of a note's file ('' if missing). Used to detect that a note changed while a
        background job was thinking."""
        try:
            st = self.path(safe_id(nid)).stat()
        except OSError:
            return ""
        return f"{st.st_mtime_ns}-{st.st_size}"

    def revs(self, ids) -> dict:
        return {i: self.rev(i) for i in ids}

    def stale(self, expected: dict) -> list:
        """Ids whose file changed since `expected` (a revs() snapshot)."""
        return [i for i, r in expected.items() if self.rev(i) != r]

    def all_notes(self, copy_: bool = False) -> list:
        return [n for n in (self.get(i, copy_) for i in self.ids()) if n]

    def cached_notes(self) -> list:
        """All notes from memory (no disk scan); the registry is kept fresh by scan()/writes."""
        return [n for n in (self.peek(i) for i in self.known_ids()) if n]

    def children(self, nid: str) -> list:
        return [n.id for n in self.all_notes() if n.parent == nid]

    # ---- logging / snapshots / saving
    def _log(self, msg: str) -> None:
        with open(self.hist / "ops.log", "a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] {msg}\n")

    def _snapshot(self, nid: str) -> None:
        p = self.path(nid)
        if not p.is_file():
            return
        d = self.hist / nid
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d / f"{datetime.now():%Y%m%d-%H%M%S-%f}.md")
        snaps = sorted(d.glob("*.md"))
        for old in snaps[:-self.keep_versions]:
            old.unlink(missing_ok=True)

    def save(self, note: Note, reason: str = "update") -> None:
        note.updated = now_iso()
        if not note.created:
            note.created = note.updated
        with self._lock:
            self._snapshot(note.id)
            p = self.path(note.id)
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(p, render_note(note))
            st = p.stat()
            note.rev = f"{st.st_mtime_ns}-{st.st_size}"
            self._paths[note.id] = p
            self._cache[note.id] = (st.st_mtime_ns, copy.deepcopy(note))
            self._touch(note.id)
            self._log(f"{reason}: {note.id}")

    def versions(self, nid: str) -> list:
        d = self.hist / safe_id(nid)
        return sorted(p.name for p in d.glob("*.md")) if d.is_dir() else []

    def restore_version(self, nid: str, version: str) -> bool:
        nid = safe_id(nid)
        src = self.hist / nid / Path(version).name
        if not src.is_file():
            return False
        self._snapshot(nid)
        with self._lock:
            atomic_write(self.path(nid), src.read_text(encoding="utf-8"))
            self._cache.pop(nid, None)
            self._paths[nid] = self.path(nid)
            self._touch(nid)
            self._log(f"restored {nid} from {src.name}")
        return True

    # ---- notes
    def create(self, nid: str, title: str = "", type_: str = "semantic", summary: str = "",
               parent: str = "core", folder: str = "") -> Note:
        nid = safe_id(nid)
        if not nid:
            raise ValueError("bad note id")
        existing = self.get(nid)
        if existing:
            return existing
        if type_ not in NOTE_TYPES:
            type_ = "semantic"
        parent = safe_id(parent)
        if nid == "core":
            parent = ""
        elif not self.exists(parent):
            parent = "core" if self.exists("core") else ""
        title = one_line(title) or nid.replace("-", " ").title()
        note = Note(id=nid, title=title, type=type_, summary=one_line(summary)[:160] or title,
                    parent=parent, created=now_iso())
        folder = safe_folder(folder)
        if folder:                                   # a physical sub-folder (e.g. User/Rules); the id never changes
            self._paths[nid] = self.notes_dir / folder / f"{nid}.md"
        self.save(note, "create")
        if parent:
            self.link(parent, nid, "branch", note.summary)
        return self.get(nid)

    def delete_note(self, nid: str) -> bool:
        """Soft delete: file goes to .trash, references are cleaned up."""
        nid = safe_id(nid)
        note = self.get(nid)
        if not note or nid == "core":
            return False
        with self._lock:
            shutil.move(str(self.path(nid)), str(self.trash / f"{nid}.{datetime.now():%Y%m%d-%H%M%S}.md"))
            self._cache.pop(nid, None)
            self._paths.pop(nid, None)
            self._touch(nid)
        for other in self.all_notes(copy_=True):
            changed = False
            keep = [l for l in other.links if l[0] != nid]
            if len(keep) != len(other.links):
                other.links, changed = keep, True
            if other.parent == nid:
                other.parent, changed = (note.parent or "core"), True
            if changed:
                self.save(other, f"unlink-deleted {nid}")
        self._log(f"deleted note {nid} (moved to .trash)")
        return True

    def purge(self, nid: str) -> bool:
        """Permanent erase of a DELETED note's trash + history (explicit user request only)."""
        nid = safe_id(nid)
        if not nid or self.exists(nid):
            return False
        for f in self.trash.glob(f"{nid}.*.md"):
            f.unlink(missing_ok=True)
        shutil.rmtree(self.hist / nid, ignore_errors=True)
        self._log(f"purged {nid}")
        return True

    # ---- facts
    def _new_fid(self, note: Note, text: str) -> str:
        base = hashlib.sha1(f"{note.id}|{norm(text)}|{time.time_ns()}".encode()).hexdigest()
        used = {f.fid for f in note.facts + note.history}
        for i in range(len(base) - 6):
            if base[i:i + 6] not in used:
                return base[i:i + 6]
        return base[:6]

    @staticmethod
    def find_fact(note: Note, ref: str) -> Optional[Fact]:
        ref = (ref or "").strip().lstrip("#")
        if not ref:
            return None
        for f in note.facts:
            if f.fid and f.fid == ref:
                return f
        pre = [f for f in note.facts if f.fid and len(ref) >= 3 and f.fid.startswith(ref)]
        if len(pre) == 1:
            return pre[0]
        nr = norm(ref)
        exact = [f for f in note.facts if norm(f.text) == nr]
        if len(exact) == 1:
            return exact[0]
        sub = [f for f in note.facts if nr and nr in norm(f.text)]
        return sub[0] if len(sub) == 1 else None

    def flag_review(self, message: str) -> None:
        self.create("review", "Needs review", "index", "Items waiting for the owner's decision", "core")
        self.add_fact("review", message, conf="inferred", src="system", allow_supersede=False)

    def add_fact(self, nid: str, text: str, conf: str = "stated", src: str = "",
                 allow_supersede: bool = True) -> FactResult:
        note = self.get(nid)
        if not note:
            return FactResult("rejected", note="no such note")
        text = clean_fact(text, self.max_fact_chars)
        if not text:
            return FactResult("rejected", note="empty fact")
        if conf not in CONFIDENCE:
            conf = "inferred"
        key, val = split_kv(text)
        old = None
        if key and allow_supersede:
            for f in note.facts:
                k2, v2 = split_kv(f.text)
                if k2 and norm(k2) == norm(key):
                    if norm(v2) == norm(val):
                        return FactResult("duplicate", f)
                    old = f
                    break
        if old is None:
            for f in note.facts:
                if norm(f.text) == norm(text) or jaccard(f.text, text) >= 0.85:
                    return FactResult("duplicate", f)
        if old is not None:
            if old.conf == "verified" and conf != "verified":
                self.flag_review(f"Conflict in [[{note.id}]]: verified '{old.text}' vs new ({conf}) '{text}' - not applied")
                return FactResult("conflict", None, old)
            note.facts.remove(old)
            old.reason = f"superseded {today()}"
            note.history.append(old)
        fact = Fact(text, self._new_fid(note, text), conf, clean_src(src), today())
        note.facts.append(fact)
        self.save(note, "fact-supersede" if old else "fact-add")
        return FactResult("superseded" if old else "added", fact, old)

    def correct_fact(self, nid: str, ref: str, new_text: str, conf: str = "stated", src: str = "") -> FactResult:
        note = self.get(nid)
        if not note:
            return FactResult("rejected", note="no such note")
        old = self.find_fact(note, ref)
        if not old:
            return FactResult("rejected", note="fact not found")
        new_text = clean_fact(new_text, self.max_fact_chars)
        if not new_text:
            return FactResult("rejected", note="empty fact")
        if conf not in CONFIDENCE:
            conf = "inferred"
        if old.conf == "verified" and conf != "verified":
            self.flag_review(f"Correction in [[{note.id}]] blocked: verified '{old.text}' vs proposed '{new_text}'")
            return FactResult("conflict", None, old)
        note.facts.remove(old)
        old.reason = f"corrected {today()}"
        note.history.append(old)
        fact = Fact(new_text, self._new_fid(note, new_text), conf, clean_src(src), today())
        note.facts.append(fact)
        self.save(note, "fact-correct")
        return FactResult("superseded", fact, old)

    def retract_fact(self, nid: str, ref: str, reason: str = "retracted") -> FactResult:
        note = self.get(nid)
        old = self.find_fact(note, ref) if note else None
        if not old:
            return FactResult("rejected", note="fact not found")
        note.facts.remove(old)
        old.reason = f"{reason} {today()}"
        note.history.append(old)
        self.save(note, "fact-retract")
        return FactResult("added", None, old, "moved to History")

    # ---- links & files
    def link(self, a: str, b: str, rel: str = "related", blurb: str = "") -> bool:
        a, b = safe_id(a), safe_id(b)
        A, B = self.get(a), self.get(b)
        if not A or not B or a == b:
            return False
        rel = re.sub(r"[^a-z_\-]", "", str(rel or "related").lower())[:20] or "related"

        def add(src: Note, tgt: Note) -> bool:
            if any(l[0] == tgt.id and l[1] == rel for l in src.links):
                return False
            src.links.append((tgt.id, rel, one_line(blurb) or tgt.summary[:80]))
            return True

        if add(A, B):
            self.save(A, f"link {a}->{b}")
        if rel != "branch" and add(B, A):
            self.save(B, f"link {b}->{a}")
        return True

    def attach_file(self, nid: str, path: str) -> Optional[tuple]:
        note = self.get(nid)
        p = Path(path)
        if not note or not p.is_file():
            return None
        entry = (str(p), sha256_file(p), today())
        note.files = [f for f in note.files if f[0] != entry[0]] + [entry]
        self.save(note, "file-attach")
        return entry

    # ---- journal (raw, append-only record of conversations = original source)
    def journal_append(self, user_name: str, assistant_name: str, user: str, reply: str) -> str:
        day = today()
        f = self.journal_dir / f"{day}.md"
        n = (f.read_text(encoding="utf-8").count("\n### turn ") if f.exists() else 0) + 1
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(f"\n### turn {n} - {datetime.now():%H:%M}\n**{user_name}:** {user}\n\n**{assistant_name}:** {reply}\n")
        return f"chat:{day}#{n}"

    # ---- organisation (folders), graph views, core digest, journal reading
    def set_summary(self, nid: str, summary: str, reason: str = "summary") -> bool:
        note = self.get(nid)
        summary = one_line(summary)[:240]
        if not note or note.summary == summary:
            return False
        note.summary = summary
        self.save(note, reason)
        return True

    def move_note(self, nid: str, folder: str) -> bool:
        """Moves a note's file to another folder. The id (= file name) does not change, so every [[wikilink]],
        parent, source reference and the note's history stay valid: Obsidian resolves links by file name."""
        nid = safe_id(nid)
        folder = safe_folder(folder)
        src = self.path(nid)
        if not nid or nid == "core" or not src.is_file():
            return False
        dest = self.notes_dir / folder / f"{nid}.md" if folder else self.notes_dir / f"{nid}.md"
        if dest == src or dest.exists():
            return False
        with self._lock:
            self._snapshot(nid)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dest)
            self._paths[nid] = dest
            self._cache.pop(nid, None)
            self._touch(nid)
            self._log(f"moved {nid} -> {folder or '(root)'}")
        return True

    def backlinks(self, nid: str) -> list:
        """[(note id, relation)]: notes that link to `nid` or have it as parent."""
        nid = safe_id(nid)
        out = []
        for n in self.cached_notes():
            if n.id == nid:
                continue
            rels = [r for t, r, _ in n.links if t == nid]
            if rels:
                out.append((n.id, rels[0]))
            elif n.parent == nid:
                out.append((n.id, "child"))
        return sorted(out)

    def core_digest(self, max_chars: int = 900) -> str:
        """Compact text of the core note, built once and kept in RAM; rebuilt only after the core note changes."""
        c = self._core_cache
        if c and c[0] == max_chars:
            return c[1]
        note = self.peek("core")
        if not note:
            return ""
        lines = [f"[core] {note.summary}"]
        for f in note.facts:
            lines.append(f"- {strip_tags(f.text)}" + (" (unverified)" if f.conf == "inferred" else ""))
        branches = [t for t, r, _ in note.links if r == "branch"]
        if branches:
            lines.append("branches: " + ", ".join(branches))
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0]
        self._core_cache = (max_chars, text)
        self.core_builds += 1
        return text

    _TURN_RE = re.compile(r"\n### turn (\d+) - (\d{2}:\d{2})\n\*\*(.+?):\*\* (.*?)\n\n\*\*(.+?):\*\* (.*?)(?=\n### turn |\Z)", re.S)

    def read_journal(self, day: str) -> list:
        f = self.journal_dir / f"{day}.md"
        if not f.is_file():
            return []
        text = "\n" + f.read_text(encoding="utf-8", errors="replace")
        return [{"src": f"chat:{day}#{m.group(1)}", "day": day, "n": int(m.group(1)), "time": m.group(2),
                 "user": m.group(4).strip(), "assistant": m.group(6).strip()}
                for m in self._TURN_RE.finditer(text)]

    def journal_turns(self, srcs) -> list:
        """The journal turns named by source ids like 'chat:2026-11-19#4' (the journal is the original record)."""
        want, days = {}, set()
        for s_ in srcs:
            m = re.match(r"^chat:(\d{4}-\d{2}-\d{2})#(\d+)", str(s_))
            if m:
                want.setdefault(m.group(1), set()).add(int(m.group(2)))
        out = []
        for day in sorted(want):
            out += [t for t in self.read_journal(day) if t["n"] in want[day]]
        return out

    def journal_since(self, since: datetime, until: Optional[datetime] = None) -> list:
        """Turns written at or after `since` (naive local time, as the journal records it)."""
        out, day = [], since.date()
        last = (until or datetime.now()).date()
        while day <= last:
            for t in self.read_journal(day.isoformat()):
                hh, mm = map(int, t["time"].split(":"))
                stamp = datetime(day.year, day.month, day.day, hh, mm)
                if stamp >= since.replace(second=0, microsecond=0) and (until is None or stamp <= until):
                    out.append(t)
            day += timedelta(days=1)
        return out

    def last_turn_src(self) -> str:
        day = today()
        turns = self.read_journal(day)
        return turns[-1]["src"] if turns else ""

    # ---- views
    def tree_text(self) -> str:
        notes = {n.id: n for n in self.all_notes()}
        kids: dict = {}
        for n in notes.values():
            kids.setdefault(n.parent, []).append(n.id)
        seen, lines = set(), []

        def walk(nid, depth):
            seen.add(nid)
            n = notes[nid]
            fol = self.folder_of(nid)
            lines.append("  " * depth + f"- {nid} [{n.type}, {len(n.facts)} facts]" + (f" - {n.summary}" if n.summary else "")
                         + (f"  ({fol}/)" if fol else ""))
            for c in sorted(kids.get(nid, [])):
                if c not in seen:
                    walk(c, depth + 1)

        if "core" in notes:
            walk("core", 0)
        loose = [i for i in notes if i not in seen]
        if loose:
            lines.append("(not attached to core: " + ", ".join(loose) + ")")
        return "\n".join(lines) or "(memory is empty)"

    def map_text(self, limit: int = 1200) -> str:
        notes = self.all_notes()
        rows = [f"- {n.id}: {n.summary}" for n in sorted(notes, key=lambda n: n.updated, reverse=True)
                if n.id != "core" and n.type != "episodic"]
        out, used = [], 0
        for r in rows:
            if used + len(r) > limit:
                out.append(f"- ...and {len(rows) - len(out)} more notes (use memory.search)")
                break
            out.append(r)
            used += len(r) + 1
        return "\n".join(out) or "(no notes yet)"

    # ---- integrity
    def doctor(self, fix: bool = False) -> list:
        issues: list = []
        notes = {n.id: n for n in self.all_notes(copy_=True)}
        for tmp in self.notes_dir.rglob("*.tmp"):
            issues.append(Issue("leftover-temp", tmp.name, "interrupted write", False))
            if fix:
                tmp.unlink(missing_ok=True)
                issues[-1].fixed = True
        for n in notes.values():
            dirty = False

            def rec(kind, detail, fixed=False):
                issues.append(Issue(kind, n.id, detail, fixed))

            if n.id_mismatch:
                rec("id-mismatch", "frontmatter id differs from file name", fix)
                dirty = dirty or fix
            if n.id != "core" and (not n.parent or n.parent not in notes):
                rec("bad-parent", f"parent '{n.parent}' missing", fix)
                if fix:
                    n.parent, dirty = ("core" if "core" in notes else ""), True
            good = [l for l in n.links if l[0] in notes]
            if len(good) != len(n.links):
                for l in n.links:
                    if l[0] not in notes:
                        rec("broken-link", f"-> [[{l[0]}]] does not exist", fix)
                if fix:
                    n.links, dirty = good, True
            seen_norm = {}
            for f in list(n.facts):
                if f.conf not in CONFIDENCE:
                    rec("bad-confidence", f"'{f.conf}' on '{f.text[:40]}'", fix)
                    if fix:
                        f.conf, dirty = "inferred", True
                if not f.fid:
                    rec("missing-fact-id", f"'{f.text[:40]}'", fix)
                    if fix:
                        f.fid, dirty = self._new_fid(n, f.text), True
                k = norm(f.text)
                if k in seen_norm:
                    rec("duplicate-fact", f"'{f.text[:50]}'", fix)
                    if fix:
                        n.facts.remove(f)
                        f.reason = f"duplicate removed {today()}"
                        n.history.append(f)
                        dirty = True
                seen_norm[k] = f
            for path, sha, _ in n.files:
                p = Path(path)
                if not p.is_file():
                    rec("file-missing", path)
                elif sha:
                    try:
                        if sha256_file(p) != sha:
                            rec("file-changed", f"{path} differs from recorded fingerprint")
                    except Exception:
                        pass
            if dirty:
                self.save(n, "doctor")
        reach, stack = set(), ["core"] if "core" in notes else []
        while stack:
            cur = stack.pop()
            if cur in reach or cur not in notes:
                continue
            reach.add(cur)
            stack += [l[0] for l in notes[cur].links] + [c for c, x in notes.items() if x.parent == cur]
        for nid, n in notes.items():
            if nid not in reach and "core" in notes:
                issues.append(Issue("orphan", nid, "not reachable from core", fix))
                if fix:
                    fresh = self.get(nid)
                    fresh.parent = "core"
                    self.save(fresh, "doctor-reattach")
                    self.link("core", nid, "branch", fresh.summary)
        return issues


def _locked(fn):
    def wrapper(self, *a, **kw):
        with self._lock:
            return fn(self, *a, **kw)
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


for _name in ("create", "delete_note", "purge", "add_fact", "correct_fact", "retract_fact", "flag_review", "link",
              "attach_file", "journal_append", "doctor", "restore_version", "set_summary"):
    setattr(MemoryStore, _name, _locked(getattr(MemoryStore, _name)))

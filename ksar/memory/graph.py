"""
Graph structure helpers: branches (navigation notes), the core note's budget, link vocabulary and graph validation.

The structure is Obsidian-friendly by construction: one note per file, [[wikilinks]] by file name (= note id), plain
folders for broad physical grouping, YAML front matter, and meaningful relations in the ## Links section.
Branch notes are created ON DEMAND (only when a real note goes under them) so the vault never fills up with empty pages.
"""
from __future__ import annotations

import re

from .store import MemoryStore, Issue, norm, safe_folder, safe_id, strip_tags

# key -> (id, title, summary, parent key, folder)
BRANCHES = {
    "user": ("user", "User", "What I know about my owner: profile, preferences, rules and goals", "", "User"),
    "preferences": ("preferences", "Preferences", "The owner's stated preferences and habits", "user", "User/Preferences"),
    "rules": ("rules", "Rules", "Restrictions and permissions the owner has set", "user", "User/Rules"),
    "goals": ("goals", "Goals", "The owner's goals and plans", "user", "User/Goals"),
    "projects": ("projects", "Projects", "The owner's projects: requirements, decisions, status", "", "Projects"),
    "knowledge": ("knowledge", "Knowledge", "Topics and how-tos I have learned", "", "Knowledge"),
    "people": ("people", "People", "People in the owner's life", "", "People"),
    "episodes": ("episodes", "Episodes", "Daily log of what happened", "", "Episodes"),
}
# which branch a kind of fact belongs to when the model does not say
KIND_BRANCH = {"rule": "rules", "preference": "preferences", "goal": "goals", "decision": "projects",
               "environment": "user", "incident": "episodes"}
RELATIONS = ("parent", "child", "explains", "caused_by", "related_to", "contradicts", "supersedes", "belongs_to",
             "originated_from", "branch", "related", "derived_from")
AUTO_MARK = " | contains: "
CORE_MAX_FACTS = 16
CORE_MAX_CHARS = 1600


def clean_rel(rel: str) -> str:
    r = re.sub(r"[^a-z_]", "", str(rel or "related_to").lower().replace("-", "_"))
    return r if r in RELATIONS else "related_to"


def branch_key(name: str) -> str:
    k = safe_id(name)
    return k if k in BRANCHES else k


def ensure_branch(store: MemoryStore, key: str):
    """Returns (branch note id, folder for its children). Creates the branch (and its parent branch) if needed."""
    key = safe_id(key) or "knowledge"
    if key in BRANCHES:
        bid, title, summary, parent_key, folder = BRANCHES[key]
        parent = ensure_branch(store, parent_key)[0] if parent_key else "core"
        if not store.exists(bid):
            store.create(bid, title, "index", summary, parent if store.exists(parent) else "core",
                         folder=safe_folder(folder))
        return bid, folder
    title = key.replace("-", " ").title()
    if not store.exists(key):
        store.create(key, title, "index", f"Notes about {title.lower()}", "core", folder=safe_folder(title))
    return key, store.folder_of(key) or safe_folder(title)


def refresh_branch_summaries(store: MemoryStore) -> list:
    """Keeps each auto-managed branch summary ('<description> | contains: a, b, c') current. Writes only when the text
    really changed. Branches the owner rewrote by hand (no marker, not a standard branch) are left alone."""
    changed = []
    notes = store.cached_notes()
    kids: dict = {}
    for n in notes:
        if n.parent:
            kids.setdefault(n.parent, []).append(n)
    for n in notes:
        if n.type != "index" or n.id in ("core", "review"):
            continue
        managed = n.id in {b[0] for b in BRANCHES.values()} or AUTO_MARK in n.summary
        if not managed:
            continue
        base = n.summary.split(AUTO_MARK)[0].strip()
        titles = sorted({k.title for k in kids.get(n.id, []) if k.type != "episodic"})[:6]
        desired = base + (AUTO_MARK + ", ".join(titles) if titles else "")
        if desired != n.summary and store.set_summary(n.id, desired, "branch-summary"):
            changed.append(n.id)
    return changed


def core_reference(store: MemoryStore, text: str, note_id: str, src: str, tag: str = "") -> bool:
    """Adds one short pointer line to the core note, e.g. 'Never delete files without asking [[file-rules]]'.
    The core stays small: at most CORE_MAX_FACTS lines and CORE_MAX_CHARS characters; otherwise the detail simply
    stays in its own note, which retrieval finds anyway."""
    core = store.get("core")
    if not core:
        return False
    short = strip_tags(text)[:140].rstrip(" .")
    line = f"{short} [[{note_id}]]" + (f" #{tag}" if tag else "")
    if any(f"[[{note_id}]]" in f.text and norm(strip_tags(f.text).replace(f"[[{note_id}]]", "")) == norm(short)
           for f in core.facts):
        return False
    if len(core.facts) >= CORE_MAX_FACTS or sum(len(f.text) for f in core.facts) + len(line) > CORE_MAX_CHARS:
        return False
    return store.add_fact("core", line, "stated", src, allow_supersede=False).status == "added"


def organize_plan(store: MemoryStore, limit: int = 5, min_age_s: int = 600) -> list:
    """Notes lying in the vault root whose parent branch has a folder: [(note id, folder)]. Moves are never automatic
    unless the owner enabled memory_auto_organize; '/memory organize' runs them on request."""
    import time
    out = []
    for n in store.cached_notes():
        if n.id == "core" or store.folder_of(n.id) or n.id in {b[0] for b in BRANCHES.values()}:
            continue
        if n.type == "index" and n.parent == "core":
            continue
        folder = store.folder_of(n.parent) if n.parent else ""
        if not folder and n.parent in {b[0]: b[4] for b in BRANCHES.values()}:
            folder = {b[0]: b[4] for b in BRANCHES.values()}[n.parent]
        if not folder:
            continue
        try:
            if time.time() - store.path(n.id).stat().st_mtime < min_age_s:
                continue
        except OSError:
            continue
        out.append((n.id, folder))
        if len(out) >= limit:
            break
    return out


WIKI_RE = re.compile(r"\[\[([^\]|#]+)")


def validate_graph(store: MemoryStore) -> list:
    """Graph-level checks on top of doctor(): broken inline wikilinks, duplicate titles, links without a reason."""
    issues = []
    notes = store.cached_notes()
    ids = {n.id for n in notes}
    titles: dict = {}
    for n in notes:
        for f in n.facts:
            for t in WIKI_RE.findall(f.text):
                if safe_id(t) not in ids:
                    issues.append(Issue("broken-wikilink", n.id, f"fact links to [[{t.strip()}]] which does not exist"))
        if n.type not in ("core", "index"):
            titles.setdefault(norm(n.title), []).append(n.id)
    for t, owners in titles.items():
        if len(owners) > 1 and t:
            issues.append(Issue("duplicate-title", ", ".join(owners), f"several notes are titled '{t}': merge them?"))
    return issues

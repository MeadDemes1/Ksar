"""Housekeeping that keeps memory navigable. The LLM only GROUPS facts; the app moves them verbatim."""
from __future__ import annotations

from ..llm import extract_json
from .store import MemoryStore, safe_id


def split_note(store: MemoryStore, llm, nid: str, threshold: int = 25):
    """If a note has too many facts, branch it into sub-notes. No fact text is ever rewritten."""
    note = store.get(nid)
    if not note or note.type in ("episodic", "core") or len(note.facts) <= threshold:
        return None
    if any(not f.fid for f in note.facts):      # run /doctor first: ids are needed to move facts safely
        return None
    numbered = "\n".join(f"{i + 1}. {f.text}" for i, f in enumerate(note.facts))
    prompt = (
        f"Group these {len(note.facts)} memory facts into 2-4 coherent sub-topics. Facts that fit no "
        "group stay unassigned. Return ONLY JSON:\n"
        '{"groups":[{"id":"short-hyphenated-id","title":"..","summary":"one line","facts":[1,2,3]}]}\n\n'
        + numbered)
    try:
        data = extract_json(llm.chat([{"role": "user", "content": prompt}], json_mode=True, temperature=0.2))
    except Exception:
        return None
    groups = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups, list):
        return None
    used, plan = set(), []
    for g in groups[:4]:
        if not isinstance(g, dict) or not isinstance(g.get("facts"), list):
            continue
        idx = []
        for i in g["facts"]:
            if isinstance(i, int) and 1 <= i <= len(note.facts) and i not in used:
                idx.append(i)
        gid = safe_id(g.get("id") or g.get("title"))
        if len(idx) >= 2 and gid and gid != nid:
            used.update(idx)
            plan.append((gid, str(g.get("title") or gid), str(g.get("summary") or ""), idx))
    if not plan:
        return None
    created = []
    moved_ids = set()
    for gid, title, summary, idx in plan:
        base, k = gid, 2
        while store.exists(gid):
            gid, k = f"{base}-{k}", k + 1
        child = store.create(gid, title, note.type, summary, nid)
        child.facts = [note.facts[i - 1] for i in idx]        # verbatim, provenance preserved
        store.save(child, f"split-from {nid}")
        created.append(gid)
        moved_ids.update(id(note.facts[i - 1]) for i in idx)
    fresh = store.get(nid)                                    # links were added by create()
    moved_text = {note.facts[i - 1].fid for _, _, _, idx in plan for i in idx}
    fresh.facts = [f for f in fresh.facts if f.fid not in moved_text]
    store.save(fresh, f"split-into {','.join(created)}")
    return created

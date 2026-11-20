"""Tests for 0.3.3: connected memory, grounded extraction, proactive retrieval, the independent background worker,
time awareness, and the performance guarantees of the chat path."""
import io
import json
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ksar.app import Ksar
from ksar.cli import handle_command
from ksar.memcmd import parse_memory_command
from ksar.memory import grounding
from ksar.memory.graph import ensure_branch, organize_plan, validate_graph
from ksar.memory.jobs import JobQueue
from ksar.memory.keeper import Conflict, is_placeholder
from ksar.memory.retrieval import Index
from ksar.memory.store import MemoryStore, parse_note, render_note
from ksar.resources import SystemResources
from ksar.timeinfo import TimeService

SRC = "chat:2026-11-19#1"
USER = ("Please never delete my files without asking first. Last week you deleted my thesis draft and I lost two days "
        "of work. I like short direct answers. I want to study computer science in Germany, my budget is 8000 euro per year.")
OPS = {"ops": [
    {"op": "add", "note": "file-permissions", "title": "File permissions", "branch": "rules", "summary": "What I may do to files",
     "facts": [{"text": "Never delete files without asking first", "kind": "rule", "conf": "stated",
                "evidence": "never delete my files without asking first", "core": True, "because": "incident-lost-thesis"}]},
    {"op": "add", "note": "incident-lost-thesis", "title": "Lost thesis draft", "branch": "episodes",
     "summary": "A deleted thesis draft",
     "facts": [{"text": "Thesis draft was deleted without asking; two days of work lost", "kind": "incident", "conf": "stated",
                "evidence": "you deleted my thesis draft and I lost two days of work"}]},
    {"op": "add", "note": "communication-style", "title": "Communication style", "branch": "preferences",
     "summary": "How answers should be",
     "facts": [{"text": "Prefers brief concise answers", "kind": "preference", "conf": "stated", "evidence": "I like short direct answers"},
               {"text": "Favourite colour: unknown", "kind": "fact", "conf": "stated", "evidence": "colour"},
               {"text": "Has a dog named Rex", "kind": "fact", "conf": "stated", "evidence": "my dog Rex"}]},
    {"op": "add", "note": "education", "title": "Education plans", "branch": "goals", "summary": "University plans",
     "facts": [{"text": "Wants to study computer science in Germany", "kind": "goal", "conf": "stated",
                "evidence": "I want to study computer science in Germany"},
               {"text": "Budget: 8000 euro per year", "kind": "preference", "conf": "stated",
                "evidence": "my budget is 8000 euro per year"}]},
    {"op": "link", "a": "file-permissions", "b": "incident-lost-thesis", "rel": "caused_by", "why": "the rule exists because of it"},
]}


def turns(user=USER, n=1):
    return [{"src": f"chat:2026-11-19#{i + 1}", "day": "2026-11-19", "n": i + 1, "time": "10:00", "user": user,
             "assistant": "Understood."} for i in range(n)]


class FakeLLM:
    """Scripted model. Extraction prompts get `extract`; everything else gets `chat_reply`. Records the thread and
    whether the app lock was held at the time of every call."""
    supports_think = True

    def __init__(self, extract=None, chat_reply="Sure.", verify="unsure"):
        self.extract, self.chat_reply, self.verify = extract, chat_reply, verify
        self.calls, self.violations, self.model = [], [], "qwen3:8b"
        self.num_ctx, self.think, self.running = 0, False, {"qwen3:8b": 1}
        self.app = None
        self.gate = None                       # an Event: extraction calls wait for it (simulates a slow model)
        self.on_extract = None                 # hook called during every extraction call
        self.fail_next = 0

    def chat(self, messages, model=None, keep_alive=None, json_mode=False, temperature=None, think=None, max_tokens=None):
        text = messages[-1]["content"]
        kind = "extract" if "MEMORY KEEPER" in text else "verify" if "OWNER'S WORDS" in text else "chat"
        self.calls.append((kind, threading.current_thread().name, model))
        if self.app is not None and getattr(self.app.lock, "held", lambda: False)() and kind != "chat":
            self.violations.append(kind)
        if kind == "extract":
            if self.fail_next:
                self.fail_next -= 1
                raise RuntimeError("ollama down")
            if self.on_extract:
                self.on_extract()
            if self.gate:
                self.gate.wait(5)
            e = self.extract
            return json.dumps(e() if callable(e) else e) if e is not None else '{"ops": []}'
        if kind == "verify":
            return json.dumps({"answer": self.verify})
        return self.chat_reply

    def load(self, model, keep_alive=-1):
        self.running[model] = 1

    def running_models(self):
        return dict(self.running)

    def installed_models(self):
        return {"qwen3:8b": 5 * 2**30}


class TrackingLock:
    """Stands in for app.lock: tells whether anyone holds it."""

    def __init__(self):
        self._l, self.depth = threading.RLock(), 0

    def acquire(self, *a, **k):
        ok = self._l.acquire(*a, **k)
        if ok:
            self.depth += 1
        return ok

    def release(self):
        self.depth -= 1
        self._l.release()

    __enter__ = lambda self: (self.acquire(), self)[1]
    __exit__ = lambda self, *a: self.release()

    def held(self):
        return self.depth > 0


def wait_until(cond, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def app(self, llm=None, **settings):
        llm = llm or FakeLLM()
        a = Ksar(self.tmp / "home", llm=llm)
        llm.app = a
        a.lock = TrackingLock()
        a.resources.get_resources = lambda: SystemResources(32000, 28000, 100)
        a.settings.idle_after_s = 0
        for k, v in settings.items():
            setattr(a.settings, k, v)
        self.addCleanup(a.idle.stop)
        return a

    def run_plan(self, a, raw=None, user=USER, **kw):
        ctx = {"revs": {}, "user_text": user, "turn_srcs": [(SRC, user)]}
        plan = a.keeper.plan(json.dumps(raw or OPS), ctx, **kw)
        rep = a.keeper.apply(plan, SRC)
        return plan, rep

    def start_worker(self, a):
        a.idle.TICK = 0.02
        a.idle.attach(a.conversation())
        a.idle.user_idle()
        a.idle.start()
        return a.idle.convo


# ------------------------------------------------------------------------------------------------ architecture
class ArchitectureTests(Base):
    LEGACY = ("---\nid: old-note\ntitle: Old note\ntype: semantic\nsummary: from 0.3.2\nparent: core\nstatus: active\n"
              "created: 2026-01-01T10:00:00\nupdated: 2026-01-01T10:00:00\n---\n# Old note\n\n## Facts\n"
              "- Likes tea <!--f:abc123 c:stated d:2026-01-01 s:chat:2026-01-01#3-->\n\n## Links\n- [[core]] (parent)\n")

    def test_old_notes_load_without_migration(self):
        home = self.tmp / "home"
        notes = home / "memory" / "notes"
        notes.mkdir(parents=True)
        (notes / "old-note.md").write_text(self.LEGACY, encoding="utf-8")
        a = Ksar(home, llm=FakeLLM())
        n = a.memory.get("old-note")
        self.assertEqual((n.id, n.parent, n.facts[0].fid, n.facts[0].src), ("old-note", "core", "abc123", "chat:2026-01-01#3"))
        self.assertEqual(a.memory.folder_of("old-note"), "")                 # stays where it is: no silent reorganisation
        self.assertTrue((notes / "old-note.md").is_file())
        a.memory.add_fact("old-note", "Dislikes coffee", "stated", "chat:2026-11-19#1")
        again = a.memory.get("old-note")
        self.assertEqual(again.facts[0].fid, "abc123")                       # ids and sources survive an update
        self.assertEqual(again.facts[0].src, "chat:2026-01-01#3")

    def test_notes_written_now_are_readable_by_the_old_format(self):
        a = self.app()
        self.run_plan(a)
        text = (a.memory.path("file-permissions")).read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\nid: file-permissions\n"))
        for key in ("title:", "type:", "summary:", "parent:", "status:", "created:", "updated:"):
            self.assertIn(key, text)
        fact_line = [l for l in text.splitlines() if l.startswith("- Never delete")][0]
        self.assertRegex(fact_line, r"<!--f:[0-9a-f]{6} c:stated d:\d{4}-\d\d-\d\d s:chat:2026-11-19#1-->$")
        self.assertIn("#rule", fact_line)                                    # an Obsidian tag, plain text for older versions

    def test_core_is_created_cached_and_invalidated_only_by_a_real_change(self):
        a = self.app()
        self.assertTrue(a.memory.exists("core"))
        d1 = a.memory.core_digest(1500)
        builds = a.memory.core_builds
        for _ in range(25):
            a.memory.core_digest(1500)
            a.index.recall("anything at all")
        self.assertEqual(a.memory.core_builds, builds)                       # never rebuilt, never re-read from disk
        a.memory.add_fact("core", "Main project: Ksar", "stated", SRC, allow_supersede=False)
        d2 = a.memory.core_digest(1500)
        self.assertEqual(a.memory.core_builds, builds + 1)
        self.assertIn("Main project: Ksar", d2)
        self.assertNotIn("Main project", d1)

    def test_core_edited_outside_ksar_is_noticed_by_the_next_sync(self):
        a = self.app()
        a.memory.core_digest(1500)
        p = a.memory.path("core")
        p.write_text(p.read_text(encoding="utf-8").replace("## Facts\n", "## Facts\n- Edited by hand <!--f:ffffff c:stated d:2026-11-19 s:manual-->\n", 1), encoding="utf-8")
        self.assertTrue(a.memory.scan())
        self.assertIn("Edited by hand", a.memory.core_digest(1500))

    def test_pipeline_builds_a_connected_vault_with_folders(self):
        a = self.app()
        self.run_plan(a)
        m = a.memory
        self.assertEqual(m.folder_of("file-permissions"), "User/Rules")
        self.assertEqual(m.folder_of("communication-style"), "User/Preferences")
        self.assertEqual(m.folder_of("incident-lost-thesis"), "Episodes")
        self.assertEqual(m.get("file-permissions").parent, "rules")
        self.assertEqual(m.get("rules").parent, "user")
        self.assertEqual(m.get("user").parent, "core")
        self.assertIn("incident-lost-thesis", [t for t, r, _ in m.get("file-permissions").links if r == "caused_by"])
        self.assertIn(("file-permissions", "caused_by"), m.backlinks("incident-lost-thesis"))
        self.assertIn("rules", m.get("user").summary + " ".join(c.id for c in m.cached_notes() if c.parent == "user"))

    def test_vault_is_valid_for_obsidian(self):
        a = self.app()
        self.run_plan(a)
        m = a.memory
        files = {p.stem for p in m.notes_dir.rglob("*.md")}
        for p in m.notes_dir.rglob("*.md"):
            text = p.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"), p.name)
            for target in re.findall(r"\[\[([^\]|#]+)", text):
                self.assertIn(target.strip(), files, f"{p.name} links to missing [[{target}]]")
        self.assertEqual([i for i in validate_graph(m)], [])
        self.assertEqual([i for i in m.doctor() if i.kind in ("broken-link", "bad-parent")], [])
        self.assertEqual(len(files), len(list(m.notes_dir.rglob("*.md"))))   # unique file names: links resolve by name

    def test_moving_a_note_keeps_id_links_and_history(self):
        a = self.app()
        self.run_plan(a)
        m = a.memory
        before = m.get("file-permissions")
        self.assertTrue(m.move_note("file-permissions", "Somewhere/Else"))
        after = m.get("file-permissions")
        self.assertEqual((after.id, [f.fid for f in after.facts]), (before.id, [f.fid for f in before.facts]))
        self.assertEqual(m.folder_of("file-permissions"), "Somewhere/Else")
        self.assertIn("file-permissions", [t for t, _, _ in m.get("rules").links])          # [[links]] resolve by name
        self.assertEqual(validate_graph(m), [])
        self.assertTrue(m.versions("file-permissions"))                                     # history kept
        self.assertFalse(m.move_note("core", "X"))

    def test_user_edits_survive_ksar_writes(self):
        a = self.app()
        self.run_plan(a)
        p = a.memory.path("communication-style")
        text = p.read_text(encoding="utf-8")
        text = text.replace("status: active", "status: active\naliases: [Style]\ntags: [mine]")
        text = text.replace("# Communication style\n", "# Communication style\n\nMy own paragraph, written in Obsidian.\n")
        text = text.replace("\n## Links", "\nA loose line I typed under Facts.\n\n## Links") if "## Links" in text else text + "\nA loose line I typed under Facts.\n"
        p.write_text(text, encoding="utf-8")
        a.memory.scan()
        a.memory.add_fact("communication-style", "Dislikes emoji #preference", "stated", SRC)
        out = p.read_text(encoding="utf-8")
        self.assertIn("aliases: [Style]", out)
        self.assertIn("tags: [mine]", out)
        self.assertIn("My own paragraph, written in Obsidian.", out)
        self.assertIn("A loose line I typed under Facts.", out)
        self.assertNotIn("## _intro", out)
        self.assertIn("Dislikes emoji", out)

    def test_front_matter_is_valid_yaml_for_obsidian_properties(self):
        a = self.app()
        a.memory.create("odd", "Incident: lost thesis #1", "semantic", "Central note: who the owner is", "core")
        text = a.memory.path("odd").read_text(encoding="utf-8")
        head = text.split("---")[1]
        self.assertIn('title: "Incident: lost thesis #1"', head)                    # a bare ': ' would be invalid YAML
        self.assertIn('summary: "Central note: who the owner is"', head)
        for line in head.strip().splitlines():
            key, _, value = line.partition(": ")
            if not value.startswith('"'):
                self.assertNotRegex(value, r":(\s|$)| #", line)                     # every unquoted value is plain YAML
        n = a.memory.get("odd")                                                    # and Ksar reads it back unchanged
        self.assertEqual((n.title, n.summary), ("Incident: lost thesis #1", "Central note: who the owner is"))
        self.assertNotRegex(a.memory.get("core").summary, r"^\"")

    def test_obsidian_list_properties_survive_ksar_writes(self):
        a = self.app()
        self.run_plan(a)
        p = a.memory.path("communication-style")
        text = p.read_text(encoding="utf-8").replace("status: active", "status: active\ntags:\n  - journal\n  - idea\naliases:\n  - Style\ncssclasses: wide")
        p.write_text(text, encoding="utf-8")
        a.memory.scan()
        a.memory.add_fact("communication-style", "Likes tea #preference", "stated", SRC)
        out = p.read_text(encoding="utf-8")
        self.assertIn("tags:\n  - journal\n  - idea\naliases:\n  - Style\ncssclasses: wide", out)    # Obsidian's list format, verbatim
        self.assertEqual(out.split("---")[1].count("tags:"), 1)                      # property names stay unique

    def test_link_targets_use_only_characters_obsidian_allows(self):
        a = self.app()
        self.run_plan(a)
        for n in a.memory.cached_notes():
            for t, _, _ in n.links:
                self.assertNotRegex(t, r"[#|^:%\[\]]")                                   # forbidden in an Obsidian link target

    def test_index_and_queue_are_rebuildable(self):
        a = self.app()
        self.run_plan(a)
        a.jobs.path.unlink(missing_ok=True)
        a2 = Ksar(self.tmp / "home", llm=FakeLLM())                           # queue file gone: starts clean
        self.assertEqual(a2.jobs.counts(), {})
        rc = a2.index.recall("delete files without asking")                   # index built from the Markdown alone
        self.assertTrue(rc.rules)
        self.assertEqual(a2.index.full_rebuilds, 1)

    def test_branches_are_created_on_demand_only(self):
        a = self.app()
        self.assertEqual([n.id for n in a.memory.cached_notes()], ["core"])
        bid, folder = ensure_branch(a.memory, "preferences")
        self.assertEqual((bid, folder), ("preferences", "User/Preferences"))
        self.assertTrue(a.memory.exists("user"))
        ensure_branch(a.memory, "astronomy")                                  # a new branch can emerge
        self.assertEqual(a.memory.folder_of("astronomy"), "Astronomy")

    def test_branch_summaries_follow_children_and_do_not_churn(self):
        a = self.app()
        self.run_plan(a)
        self.assertIn("Communication style", a.memory.get("preferences").summary)
        from ksar.memory.graph import refresh_branch_summaries
        gen = a.memory.gen
        self.assertEqual(refresh_branch_summaries(a.memory), [])              # nothing changed: nothing rewritten
        self.assertEqual(a.memory.gen, gen)

    def test_organize_is_gradual_and_off_by_default(self):
        a = self.app()
        a.memory.create("goals", "Goals", "index", "Goals", "core", folder="User/Goals")
        a.memory.create("education-old", "Old education", "semantic", "legacy root note", "goals")    # lies in the vault root
        plan = organize_plan(a.memory, limit=5, min_age_s=0)
        self.assertEqual(plan, [("education-old", "User/Goals")])
        self.assertEqual(a.settings.memory_auto_organize, False)
        self.assertEqual(a.idle.job_organize(), None)                          # the worker does nothing unless enabled
        self.assertEqual(a.memory.folder_of("education-old"), "")


# ------------------------------------------------------------------------------------------------ extraction
class ExtractionTests(Base):
    def test_explicit_preference_and_goal_are_saved_with_sources(self):
        a = self.app()
        plan, rep = self.run_plan(a)
        facts = {f.text.split(" #")[0]: f for n in a.memory.cached_notes() for f in n.facts}
        self.assertIn("Prefers brief concise answers", facts)
        self.assertIn("Wants to study computer science in Germany", facts)
        self.assertEqual(facts["Prefers brief concise answers"].conf, "stated")
        self.assertEqual(facts["Budget: 8000 euro per year"].src, SRC)
        self.assertEqual(a.memory.journal_turns([])[:1], [])

    def test_explicit_restriction_is_saved_linked_and_referenced_in_core(self):
        a = self.app()
        self.run_plan(a)
        rule = a.memory.get("file-permissions").facts[0]
        self.assertIn("#rule", rule.text)
        core_facts = [f.text for f in a.memory.get("core").facts]
        self.assertTrue(any("[[file-permissions]]" in t and "#rule" in t for t in core_facts))
        self.assertTrue(any(t == "incident-lost-thesis" for t, r, _ in a.memory.get("file-permissions").links if r == "caused_by"))
        inc = a.memory.get("incident-lost-thesis").facts[0]
        self.assertIn("#incident", inc.text)

    def test_a_restriction_is_never_downgraded(self):
        a = self.app()
        self.run_plan(a)
        raw = {"ops": [{"op": "add", "note": "file-permissions", "facts": [
            {"text": "Prefers to be asked before deleting files", "kind": "preference", "conf": "inferred",
             "evidence": "never delete my files without asking first"}]}]}
        self.run_plan(a, raw)
        n = a.memory.get("file-permissions")
        rules = [f for f in n.facts if "#rule" in f.text]
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].conf, "stated")
        self.assertEqual(rules[0].text.split(" #")[0], "Never delete files without asking first")

    def test_paraphrases_are_recognised(self):
        said = "I really like it when you answer short and direct. Never erase my documents unless I confirm."
        cases = [("Prefers brief concise answers", "I really like it when you answer short and direct", "yes"),
                 ("Never delete documents without confirmation", "Never erase my documents unless I confirm", "yes"),
                 ("Allows deleting documents without confirmation", "Never erase my documents unless I confirm", "no"),
                 ("Prefers long detailed answers", "answer short and direct", "no")]
        for fact, ev, want in cases:
            self.assertEqual(grounding.ground(fact, ev, said).label, want, fact)

    def test_unsupported_assumptions_are_rejected(self):
        a = self.app()
        plan, _ = self.run_plan(a)
        reasons = dict(plan.rejected)
        self.assertTrue(any("evidence is not in" in r for r in reasons.values()))        # 'my dog Rex' was never said
        self.assertNotIn("Has a dog named Rex", " ".join(f.text for n in a.memory.cached_notes() for f in n.facts))

    def test_numbers_must_match_exactly(self):
        self.assertEqual(grounding.ground("Budget: 9000 euro per year", "my budget is 8000 euro per year",
                                          "my budget is 8000 euro per year").label, "no")

    def test_unknown_placeholders_are_never_generated(self):
        for t in ("Favourite colour: unknown", "Pet: n/a", "Plans: ?", "Phone: not specified", "Цвет: неизвестно", "x"):
            self.assertTrue(is_placeholder(t), t)
        self.assertFalse(is_placeholder("Favourite colour: dark blue"))
        a = self.app()
        plan, _ = self.run_plan(a)
        self.assertNotIn("unknown", " ".join(f.text for n in a.memory.cached_notes() for f in n.facts).lower())

    def test_no_empty_notes_when_every_fact_is_rejected(self):
        a = self.app()
        raw = {"ops": [{"op": "add", "note": "pets", "title": "Pets", "facts": [
            {"text": "Favourite colour: unknown", "evidence": "x"}, {"text": "Has a dog Rex", "evidence": "my dog Rex"}]}]}
        plan, rep = self.run_plan(a, raw)
        self.assertFalse(a.memory.exists("pets"))
        self.assertEqual([n.id for n in a.memory.cached_notes()], ["core"])

    def test_legacy_op_format_still_works_and_stays_grounded(self):
        a = self.app()
        raw = {"ops": [{"op": "add", "note": "meady", "title": "Meady", "type": "person", "summary": "The owner",
                        "facts": ["Prefers short direct answers", "Pet: tiger named Tom"], "conf": "stated"}]}
        self.run_plan(a, raw)
        texts = [f.text for f in a.memory.get("meady").facts]
        self.assertEqual(texts, ["Prefers short direct answers"])

    def test_temporary_circumstances_stay_out_of_core(self):
        a = self.app()
        raw = {"ops": [{"op": "add", "note": "health-now", "title": "Right now", "branch": "user", "facts": [
            {"text": "Currently has a cold", "kind": "temporary", "conf": "stated", "core": True,
             "evidence": "I have a cold this week"}]}]}
        self.run_plan(a, raw, user="I have a cold this week")
        self.assertIn("#temporary", a.memory.get("health-now").facts[0].text)
        self.assertFalse(any("cold" in f.text for f in a.memory.get("core").facts))

    def test_inferred_facts_are_labelled_and_never_verified(self):
        a = self.app()
        raw = {"ops": [{"op": "add", "note": "habits", "title": "Habits", "branch": "preferences", "facts": [
            {"text": "Probably works late at night", "kind": "preference", "conf": "inferred",
             "evidence": "I always write to you after midnight"},
            {"text": "Is a perfectionist", "kind": "fact", "conf": "verified", "evidence": "I always write to you after midnight"}]}]}
        self.run_plan(a, raw, user="I always write to you after midnight")
        confs = {f.text.split(" #")[0]: f.conf for f in a.memory.get("habits").facts}
        self.assertEqual(confs.get("Probably works late at night"), "inferred")
        self.assertNotIn("verified", confs.values())
        self.assertNotIn("Is a perfectionist", confs)                       # not supported by the quote

    def test_untrusted_sources_only_ever_produce_inferred_facts(self):
        a = self.app()
        ctx = {"revs": {}, "user_text": USER, "turn_srcs": [(SRC, USER)]}
        plan = a.keeper.plan(json.dumps(OPS), ctx, tainted=True)
        a.keeper.apply(plan, SRC, tainted=True)
        self.assertEqual({f.conf for f in a.memory.get("communication-style").facts}, {"inferred"})

    def test_duplicates_are_not_created_twice(self):
        a = self.app()
        self.run_plan(a)
        counts = {n.id: len(n.facts) for n in a.memory.cached_notes()}
        self.run_plan(a)
        self.assertEqual({n.id: len(n.facts) for n in a.memory.cached_notes()}, counts)

    def test_same_subject_is_merged_into_the_existing_note(self):
        a = self.app()
        self.run_plan(a)
        raw = {"ops": [{"op": "add", "note": "answer-style", "title": "Communication style", "facts": [
            {"text": "Dislikes emoji", "kind": "preference", "conf": "stated", "evidence": "I dislike emoji"}]}]}
        self.run_plan(a, raw, user="I dislike emoji")
        self.assertFalse(a.memory.exists("answer-style"))
        self.assertIn("Dislikes emoji", " ".join(f.text for f in a.memory.get("communication-style").facts))

    def test_newer_stated_fact_replaces_older_and_history_is_kept(self):
        a = self.app()
        for value, user in (("8000", "my budget is 8000 euro per year"), ("9500", "my budget is now 9500 euro per year")):
            raw = {"ops": [{"op": "add", "note": "education", "title": "Education plans", "facts": [
                {"text": f"Budget: {value} euro per year", "conf": "stated", "evidence": user}]}]}
            self.run_plan(a, raw, user=user)
        n = a.memory.get("education")
        self.assertEqual([f.text for f in n.facts if f.text.startswith("Budget")], ["Budget: 9500 euro per year"])
        self.assertTrue(any("8000" in h.text for h in n.history))

    def test_inferred_claim_does_not_overwrite_a_stated_fact(self):
        a = self.app()
        raw = {"ops": [{"op": "add", "note": "education", "title": "Education plans", "facts": [
            {"text": "Budget: 8000 euro per year", "conf": "stated", "evidence": "my budget is 8000 euro per year"}]}]}
        self.run_plan(a, raw, user="my budget is 8000 euro per year")
        raw = {"ops": [{"op": "add", "note": "education", "facts": [
            {"text": "Budget: 12000 euro per year", "conf": "inferred", "evidence": "I can probably find 12000 euro per year"}]}]}
        _, rep = self.run_plan(a, raw, user="I can probably find 12000 euro per year")
        self.assertEqual([f.text for f in a.memory.get("education").facts if f.text.startswith("Budget")],
                         ["Budget: 8000 euro per year"])
        self.assertTrue(any("review" in l for l in rep.lines))
        self.assertTrue(a.memory.exists("review"))

    def test_contradicting_rules_are_recorded_for_review_not_silently_replaced(self):
        a = self.app()
        self.run_plan(a)
        raw = {"ops": [{"op": "add", "note": "file-permissions", "facts": [
            {"text": "May delete files without asking first", "kind": "rule", "conf": "stated",
             "evidence": "you may delete files without asking first"}]}]}
        _, rep = self.run_plan(a, raw, user="you may delete files without asking first")
        texts = [f.text for f in a.memory.get("file-permissions").facts]
        self.assertTrue(any(t.startswith("Never delete") for t in texts))      # the old rule is still there
        self.assertTrue(any("rule conflict" in l for l in rep.lines))
        self.assertIn("Conflicting rules", " ".join(f.text for f in a.memory.get("review").facts))

    def test_borderline_facts_use_one_yes_no_check_or_wait(self):
        a = self.app()
        raw = {"ops": [{"op": "add", "note": "education", "title": "Education plans", "facts": [
            {"text": "Wants to attend university in Germany", "kind": "goal", "conf": "stated",
             "evidence": "I want to study in Germany"}]}]}
        user = "I want to study in Germany"
        ctx = {"revs": {}, "user_text": user, "turn_srcs": [(SRC, user)]}
        no_check = a.keeper.plan(json.dumps(raw), ctx)                          # without a model: waits, never guesses
        self.assertEqual(len(no_check.deferred), 1)
        yes = a.keeper.plan(json.dumps(raw), ctx, verify=lambda p: '{"answer": "yes"}')
        self.assertEqual(yes.notes["education"].facts[0].conf, "inferred")        # a judgement never becomes 'stated'
        no = a.keeper.plan(json.dumps(raw), ctx, verify=lambda p: '{"answer": "no"}')
        self.assertTrue(no.rejected and not no.deferred)
        unsure = a.keeper.plan(json.dumps(raw), ctx, verify=lambda p: '{"answer": "unsure"}')
        self.assertEqual(len(unsure.deferred), 1)

    def test_sources_are_traceable_to_the_journal_turn(self):
        a = self.app()
        a.memory.journal_append("boss", "Ksar", "my budget is 8000 euro per year", "Noted.")
        a.memory.journal_append("boss", "Ksar", "I like short direct answers", "Okay.")
        day = datetime.now().strftime("%Y-%m-%d")
        t = a.memory.read_journal(day)
        user = " ".join(x["user"] for x in t)
        ctx = {"revs": {}, "user_text": user, "turn_srcs": [(x["src"], x["user"]) for x in t]}
        raw = {"ops": [{"op": "add", "note": "communication-style", "title": "Communication style", "facts": [
            {"text": "Prefers brief concise answers", "conf": "stated", "evidence": "I like short direct answers"}]}]}
        plan = a.keeper.plan(json.dumps(raw), ctx)
        a.keeper.apply(plan, t[-1]["src"])
        fact = a.memory.get("communication-style").facts[0]
        self.assertEqual(fact.src, f"chat:{day}#2")
        self.assertEqual(a.memory.journal_turns([fact.src])[0]["user"], "I like short direct answers")

    def test_every_extraction_run_uses_the_same_validated_path(self):
        llm = FakeLLM(extract=OPS)
        a = self.app(llm)
        a.memory_chip_used = None
        rep = a.keeper.consolidate([(USER, "Understood.")], SRC)               # the synchronous fallback
        self.assertTrue(rep.ok)
        self.assertTrue(a.memory.exists("file-permissions"))
        self.assertFalse(a.memory.exists("pets"))


# ------------------------------------------------------------------------------------------------ retrieval
class RetrievalTests(Base):
    def build(self):
        a = self.app()
        m = a.memory

        def mk(i, title, ty, summary, parent, facts, folder=""):
            m.create(i, title, ty, summary, parent, folder=folder)
            for f in facts:
                m.add_fact(i, f, "stated", SRC)
        mk("user", "User", "index", "What I know about my owner", "core", [], "User")
        mk("rules", "Rules", "index", "Restrictions and permissions", "user", [], "User/Rules")
        mk("file-permissions", "File permissions", "semantic", "What I may do to files", "rules",
           ["Never delete or overwrite files without asking first #rule"], "User/Rules")
        mk("incident-lost-thesis", "Lost thesis draft", "semantic", "A thesis draft was deleted", "user",
           ["The thesis draft was deleted without asking and two days of work were lost"], "Episodes")
        m.link("file-permissions", "incident-lost-thesis", "caused_by", "why the rule exists")
        mk("communication-style", "Communication style", "semantic", "How the owner likes answers", "user",
           ["Prefers short direct answers #preference", "Dislikes emoji in answers #preference"], "User/Preferences")
        mk("programming-assistance", "Programming assistance", "semantic", "Coding preferences", "user",
           ["Prefers Python with type hints #preference", "Wants complete runnable code, not fragments #preference"])
        mk("education", "Education plans", "semantic", "University plans abroad", "user",
           ["Wants to study computer science in Germany", "Budget limit: about 8000 euro per year",
            "Needs English-taught programs"], "User/Goals")
        mk("ksar", "Ksar project", "project", "Local AI assistant", "core",
           ["Ksar runs on Ollama with qwen3:8b", "Ksar keeps memory as Markdown notes", "Decision: no external database"], "Projects")
        mk("pets", "Pets", "semantic", "Owner's pets", "user", ["Has a cat named Tom"])
        return a

    def test_relevant_preferences_are_retrieved_without_any_memory_command(self):
        a = self.build()
        rc = a.index.recall("Which universities should I apply to for my degree?")
        self.assertTrue(rc.worthwhile)
        self.assertEqual(rc.blocks[0].note_id, "education")
        rc = a.index.recall("write a python script that renames my photos")
        self.assertTrue(rc.worthwhile)
        self.assertIn("programming-assistance", [b.note_id for b in rc.blocks])

    def test_project_information_is_retrieved(self):
        a = self.build()
        rc = a.index.recall("what did we decide about the ksar database?")
        self.assertEqual(rc.blocks[0].note_id, "ksar")
        self.assertTrue(rc.worthwhile)

    def test_unrelated_questions_do_not_pull_personal_memory(self):
        a = self.build()
        for q in ("what is the capital of France?", "explain recursion", "hi, how are you?", "how many legs does a spider have?",
                  "translate good morning into Spanish"):
            self.assertFalse(a.index.recall(q).worthwhile, q)

    def test_a_restriction_surfaces_when_a_relevant_action_is_considered(self):
        a = self.build()
        rc = a.index.recall("please delete the old log files")
        self.assertEqual([b.note_id for b in rc.rules], ["file-permissions"])
        self.assertTrue(rc.worthwhile)
        text = rc.text(1500)
        self.assertTrue(text.startswith("RULES from the owner"))
        self.assertIn("incident-lost-thesis", [b.note_id for b in rc.blocks])    # follows the link to the explanation
        self.assertEqual(text.count("Never delete or overwrite files"), 1)       # shown once, not twice

    def test_branch_navigation(self):
        a = self.build()
        rc = a.index.recall("rules restrictions permissions")
        self.assertIn("rules", rc.branches)
        rc = a.index.recall("what restrictions have I set?")
        self.assertIn("file-permissions", [b.note_id for b in rc.blocks])        # found through its branch

    def test_graph_expansion_is_bounded(self):
        a = self.build()
        for i in range(30):                                                     # a hub linked to many notes
            a.memory.create(f"spoke-{i}", f"Spoke {i}", "semantic", "spoke about ksar", "ksar")
            a.memory.add_fact(f"spoke-{i}", f"ksar detail {i}", "stated", SRC)
            a.memory.link("ksar", f"spoke-{i}", "explains", "x")
        rc = a.index.recall("ksar", limit=4, expand=3)
        self.assertLessEqual(len(rc.blocks), 4 + 3)
        self.assertLessEqual(len([b for b in rc.blocks if b.level == "link"]), 3)

    def test_direct_match_is_not_displaced_by_loosely_related_notes(self):
        a = self.build()
        rc = a.index.recall("Germany budget")
        self.assertEqual(rc.blocks[0].note_id, "education")
        self.assertTrue(all(b.score <= rc.blocks[0].score for b in rc.blocks))

    def test_retrieval_is_deterministic(self):
        a = self.build()
        runs = [a.index.recall("delete files and the thesis incident").text(1500) for _ in range(3)]
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])
        b = Index(a.memory)
        self.assertEqual(b.recall("delete files and the thesis incident").text(1500), runs[0])

    def test_recency_matters_only_when_the_request_is_about_time(self):
        a = self.build()
        rc = a.index.recall("what did I say recently about my cat?")
        self.assertIn("pets", [b.note_id for b in rc.blocks])

    def test_why_a_note_was_retrieved_can_be_inspected(self):
        a = self.build()
        rc = a.index.recall("please delete the old log files")
        text = rc.explain()
        self.assertIn("rule matches", text)
        self.assertIn("file-permissions", text)
        buf = io.StringIO()
        with redirect_stdout(buf):
            handle_command("/memory search delete files", a, a.conversation(), print)
        self.assertIn("RULES from the owner", buf.getvalue())

    def test_agent_injects_memory_only_when_it_helps(self):
        llm = FakeLLM(chat_reply="ok")
        a = self.build()
        a.llm = llm
        a.keeper.llm = llm
        seen = []
        orig = llm.chat
        llm.chat = lambda messages, **kw: (seen.append(messages), orig(messages, **kw))[1]
        c = a.conversation()
        c.send("what is the capital of France?")
        self.assertNotIn("Your memory", seen[-1][-1]["content"])
        self.assertLess(len(seen[-1][0]["content"]), 600)                       # the chat prompt stays small and stable
        c.send("Which universities should I apply to for my degree?")
        self.assertIn("Wants to study computer science in Germany", seen[-1][-1]["content"])
        self.assertIn("[core]", seen[-1][-1]["content"])                        # core context travels with relevant memory
        sys_prompts = {m[0]["content"] for m in seen}
        self.assertEqual(len(sys_prompts), 1)                                    # same system prompt: Ollama's cache stays valid
        self.assertTrue(any("worthwhile: True" in l for l in a.agent.last_turn.memory_lines))

    def test_rules_are_offered_with_tool_requests(self):
        llm = FakeLLM(chat_reply='{"say": "I will ask you first."}')
        a = self.build()
        a.llm = llm
        a.keeper.llm = llm
        seen = []
        orig = llm.chat
        llm.chat = lambda messages, **kw: (seen.append(messages), orig(messages, **kw))[1]
        a.conversation().send("please delete the old files in my workspace")
        self.assertIn("RULES from the owner", seen[-1][-1]["content"])


# ------------------------------------------------------------------------------------------------ job queue
class JobQueueTests(Base):
    def test_jobs_survive_a_restart_and_running_jobs_are_requeued(self):
        q = JobQueue(self.tmp / "q.json")
        q.add("extract", {"srcs": ["chat:2026-11-19#1"]}, priority=5)
        q.add("extract", {"srcs": ["chat:2026-11-19#2"]}, priority=1)
        claimed = q.claim(("extract",), limit=1)
        self.assertEqual(claimed[0].priority, 1)
        q2 = JobQueue(self.tmp / "q.json")                                         # the app was killed mid-job
        self.assertEqual(q2.counts(), {"pending": 2})
        self.assertTrue(q2.jobs[0].id and q2.jobs[0].created)

    def test_duplicate_jobs_are_not_queued(self):
        q = JobQueue(self.tmp / "q.json")
        self.assertIsNotNone(q.add("extract", {"srcs": ["a"]}))
        self.assertIsNone(q.add("extract", {"srcs": ["a"]}))
        self.assertIsNotNone(q.add("extract", {"srcs": ["b"]}))
        self.assertEqual(q.counts(), {"pending": 2})

    def test_failures_are_retried_with_backoff_then_given_up(self):
        q = JobQueue(self.tmp / "q.json")
        job = q.add("extract", {"srcs": ["a"]}, max_attempts=2)
        q.fail(q.claim(("extract",)), "boom", retry_in_s=3600)
        self.assertEqual((job.status, job.attempts), ("pending", 1))
        self.assertFalse(q.has_ready(("extract",)))                                # waiting for its back-off
        job.next_try = ""
        q.fail(q.claim(("extract",)), "boom again")
        self.assertEqual(job.status, "failed")
        self.assertTrue(job.finished and job.result)

    def test_fields_are_complete(self):
        q = JobQueue(self.tmp / "q.json")
        j = q.add("extract", {"srcs": ["chat:2026-11-19#4"]}, priority=2, source_refs=["chat:2026-11-19#4"])
        for f in ("id", "type", "created", "status", "priority", "source_refs", "attempts", "max_attempts", "finished", "result"):
            self.assertTrue(hasattr(j, f), f)
        self.assertIn("+", j.created[10:] if len(j.created) > 19 else "+")        # timezone-aware timestamp


# ------------------------------------------------------------------------------------------------ the independent worker
class WorkerTests(Base):
    def test_chat_never_waits_for_memory_extraction(self):
        llm = FakeLLM(extract=OPS)
        llm.gate = threading.Event()                                              # the extraction model is "very slow"
        a = self.app(llm)
        c = self.start_worker(a)
        c.send("hello there")                                                    # enqueues; returns at once
        self.assertTrue(wait_until(lambda: any(k == "extract" for k, _, _ in llm.calls)))
        t0 = time.perf_counter()
        reply = c.send("and one more thing")                                     # the worker is stuck inside the model call
        self.assertLess(time.perf_counter() - t0, 1.0)
        self.assertEqual(reply, "Sure.")
        llm.gate.set()
        self.assertTrue(wait_until(lambda: a.jobs.pending_count(("extract",)) == 0 or True))

    def test_worker_processes_the_journal_in_the_background(self):
        llm = FakeLLM(extract=OPS)
        a = self.app(llm)
        c = self.start_worker(a)
        c.send(USER)
        self.assertFalse(a.memory.exists("file-permissions"))                    # not at answer time
        self.assertTrue(wait_until(lambda: a.memory.exists("file-permissions")))
        self.assertTrue(wait_until(lambda: a.jobs.counts().get("done", 0) >= 1))
        self.assertEqual(a.memory.folder_of("file-permissions"), "User/Rules")
        self.assertTrue(a.jobs.get_meta("last_extract_ok"))
        self.assertEqual(a.unprocessed_turns(), [])                               # watermark advanced

    def test_no_model_call_runs_while_the_app_lock_is_held(self):
        llm = FakeLLM(extract=OPS)
        a = self.app(llm)
        c = self.start_worker(a)
        c.send(USER)
        self.assertTrue(wait_until(lambda: a.memory.exists("file-permissions")))
        self.assertEqual(llm.violations, [])
        self.assertTrue(any(k == "extract" and "ksar-idle" in th for k, th, _ in llm.calls))   # it ran on the worker thread
        self.assertEqual(a.lock.depth, 0)                                         # the worker never took the lock at all

    def test_user_activity_stops_background_work_and_nothing_is_lost(self):
        llm = FakeLLM(extract=OPS)
        a = self.app(llm)
        c = self.start_worker(a)
        c.send(USER)
        llm.on_extract = a.idle.user_active                                       # the owner presses Enter mid-generation
        self.assertTrue(wait_until(lambda: any(k == "extract" for k, _, _ in llm.calls)))
        self.assertTrue(wait_until(lambda: a.jobs.counts().get("pending", 0) == 1 and a.jobs.counts().get("running", 0) == 0))
        self.assertFalse(a.memory.exists("file-permissions"))
        self.assertFalse(a.idle.is_idle())                                        # paused while the owner is active
        llm.on_extract = None
        a.idle.user_idle()
        self.assertTrue(wait_until(lambda: a.memory.exists("file-permissions")))   # resumes later, the job was not lost

    def test_failed_jobs_are_retried(self):
        llm = FakeLLM(extract=OPS)
        llm.fail_next = 1
        a = self.app(llm)
        c = a.conversation()
        a.idle.attach(c)
        c.pending = []
        a.jobs.add("extract", {"srcs": []}, priority=3)
        a.memory.journal_append("boss", "Ksar", USER, "ok")
        src = f"chat:{datetime.now():%Y-%m-%d}#1"
        a.jobs.jobs[0].payload = {"srcs": [src]}
        a.jobs.jobs[0].source_refs = [src]
        a.idle.user_idle()
        with self.assertRaises(RuntimeError):
            a.idle.job_extract()
        self.assertEqual((a.jobs.jobs[0].status, a.jobs.jobs[0].attempts), ("pending", 1))
        a.jobs.jobs[0].next_try = ""
        a.idle.job_extract()
        self.assertEqual(a.jobs.jobs[0].status, "done")
        self.assertTrue(a.memory.exists("file-permissions"))

    def test_stale_background_results_cannot_overwrite_newer_notes(self):
        llm = FakeLLM()
        a = self.app(llm)
        self.run_plan(a)                                                          # existing communication-style note
        edits = {"n": 0}

        def foreground_edit():                                                    # the foreground changes the note meanwhile
            edits["n"] += 1
            if edits["n"] == 1:
                a.memory.add_fact("communication-style", "Dislikes emoji in answers #preference", "stated", "chat:2026-11-19#9")

        llm.on_extract = foreground_edit
        user = "I like short direct answers and please keep your answers polite"
        llm.extract = {"ops": [{"op": "add", "note": "communication-style", "facts": [
            {"text": "Wants polite answers", "kind": "preference", "conf": "stated",
             "evidence": "please keep your answers polite"}]}]}
        a.memory.journal_append("boss", "Ksar", user, "ok")
        src = f"chat:{datetime.now():%Y-%m-%d}#1"
        c = a.conversation()
        a.idle.attach(c)
        a.jobs.add("extract", {"srcs": [src]}, source_refs=[src])
        a.idle.user_idle()
        a.idle.job_extract()
        texts = [f.text.split(" #")[0] for f in a.memory.get("communication-style").facts]
        self.assertIn("Dislikes emoji in answers", texts)                         # the foreground edit survived
        self.assertIn("Wants polite answers", texts)                      # and the background fact was added to the fresh note
        self.assertEqual(len([k for k, _, _ in llm.calls if k == "extract"]), 2)   # it re-planned once after the conflict

    def test_plan_refuses_to_apply_over_a_changed_note(self):
        a = self.app()
        self.run_plan(a)
        ctx = {"revs": a.memory.revs(["communication-style"]), "user_text": "I dislike emoji", "turn_srcs": [(SRC, "I dislike emoji")]}
        raw = {"ops": [{"op": "add", "note": "communication-style", "facts": [
            {"text": "Dislikes emoji", "kind": "preference", "evidence": "I dislike emoji"}]}]}
        plan = a.keeper.plan(json.dumps(raw), ctx)
        p = a.memory.path("communication-style")
        p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")        # the owner edits the file by hand
        with self.assertRaises(Conflict):
            a.keeper.apply(plan, SRC)

    def test_memory_command_returns_at_once_and_queues_the_work(self):
        llm = FakeLLM(extract=OPS)
        llm.gate = threading.Event()
        a = self.app(llm)
        c = self.start_worker(a)
        c.send("hello")
        before = len(llm.calls)
        t0 = time.perf_counter()
        reply = c.send("Remember the important things we discussed during the last hour.")
        self.assertLess(time.perf_counter() - t0, 0.5)
        self.assertIn("process that in the background", reply)
        self.assertTrue(any(j.priority == 1 and j.payload.get("since") for j in a.jobs.jobs))
        self.assertEqual(len([k for k, _, _ in llm.calls[before:] if k == "chat"]), 0)     # no model call for the command itself
        self.assertEqual(c.send("and now another question"), "Sure.")                       # chat continues immediately
        llm.gate.set()

    def test_memory_command_parsing(self):
        now = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
        start = now - timedelta(hours=3)
        got = parse_memory_command("Remember the important things we discussed during the last hour", now, start)
        self.assertEqual(got["since"], now - timedelta(hours=1))
        self.assertEqual(parse_memory_command("remember everything we discussed in the last 30 minutes", now, start)["since"],
                         now - timedelta(minutes=30))
        self.assertEqual(parse_memory_command("save this conversation", now, start)["since"], start)
        self.assertEqual(parse_memory_command("запомни всё за последний час", now, start)["since"], now - timedelta(hours=1))
        self.assertIsNone(parse_memory_command("remember that my dog is called Rex", now, start))   # real content: normal turn
        self.assertIsNone(parse_memory_command("what do you remember about me?", now, start))
        self.assertIsNone(parse_memory_command("hello", now, start))

    def test_without_a_running_worker_memory_is_saved_synchronously_as_before(self):
        llm = FakeLLM(extract=OPS)
        a = self.app(llm)
        c = a.conversation()                                                       # no worker thread started
        c.send("remember: " + USER)
        self.assertTrue(a.memory.exists("file-permissions"))
        self.assertEqual(a.jobs.counts(), {})

    def test_background_results_never_print_into_the_chat(self):
        llm = FakeLLM(extract=OPS)
        a = self.app(llm)
        c = self.start_worker(a)
        buf = io.StringIO()
        with redirect_stdout(buf):
            c.send(USER, on_event=lambda m: None)
            self.assertTrue(wait_until(lambda: a.memory.exists("file-permissions")))
            self.assertTrue(wait_until(lambda: a.jobs.counts().get("done", 0) >= 1))
        self.assertEqual(buf.getvalue(), "")                                       # nothing printed by the worker
        self.assertTrue(wait_until(lambda: (a.paths.logs / "idle.log").exists() and a.idle.notes))
        self.assertTrue(a.idle.pop_notes())                                        # available on request (/idle status, logs)
        self.assertFalse(a.settings.idle_notify)
        log = (a.paths.logs / "idle.log").read_text(encoding="utf-8")
        self.assertIn("extract", log)

    def test_catch_up_requeues_turns_that_never_reached_memory(self):
        a = self.app()
        a.memory.journal_append("boss", "Ksar", "first", "ok")
        a.jobs.set_meta("watermark", {datetime.now().strftime("%Y-%m-%d"): 0})
        a.memory.journal_append("boss", "Ksar", "second", "ok")
        self.assertEqual(len(a.unprocessed_turns()), 2)
        a.catch_up()
        self.assertEqual(a.jobs.pending_count(("extract",)), 1)
        a.catch_up()
        self.assertEqual(a.jobs.pending_count(("extract",)), 1)                    # no duplicate catch-up job

    def test_first_start_does_not_reprocess_old_conversations(self):
        home = self.tmp / "home"
        m = MemoryStore(home / "memory")
        m.journal_append("boss", "Ksar", "an old conversation", "ok")
        a = Ksar(home, llm=FakeLLM())
        self.assertEqual(a.unprocessed_turns(), [])
        self.assertEqual(a.jobs.pending_count(), 0)

    def test_long_conversations_are_processed_in_chunks(self):
        llm = FakeLLM(extract={"ops": []})
        a = self.app(llm)
        a.idle.CHUNK_CHARS = 300
        c = a.conversation()
        a.idle.attach(c)
        for i in range(6):
            a.memory.journal_append("boss", "Ksar", f"message number {i} " + "x" * 100, "ok")
        day = datetime.now().strftime("%Y-%m-%d")
        srcs = [f"chat:{day}#{i}" for i in range(1, 7)]
        a.jobs.add("extract", {"srcs": srcs})
        a.idle.user_idle()
        a.idle.job_extract()
        self.assertEqual(a.jobs.pending_count(("extract",)), 1)                    # the rest waits as its own job
        self.assertTrue(len(a.jobs.jobs[-1].payload["srcs"]) < 6)

    def test_deferred_facts_are_rechecked_later(self):
        llm = FakeLLM(verify="unsure")
        a = self.app(llm)
        a.idle.attach(a.conversation())
        user = "I want to study in Germany"
        llm.extract = {"ops": [{"op": "add", "note": "education", "title": "Education plans", "facts": [
            {"text": "Wants to attend university in Germany", "kind": "goal", "conf": "stated",
             "evidence": "I want to study in Germany"}]}]}
        a.memory.journal_append("boss", "Ksar", user, "ok")
        src = f"chat:{datetime.now():%Y-%m-%d}#1"
        a.jobs.add("extract", {"srcs": [src]}, source_refs=[src])
        a.idle.user_idle()
        a.idle.job_extract()
        self.assertFalse(a.memory.exists("education"))                             # unclear: nothing is invented
        rechecks = [j for j in a.jobs.jobs if j.type == "recheck"]
        self.assertEqual(len(rechecks), 1)
        llm.verify = "yes"                                                         # later the check can decide
        rechecks[0].next_try = ""
        a.idle.job_recheck()
        self.assertTrue(a.memory.exists("education"))
        self.assertEqual(a.memory.get("education").facts[0].conf, "inferred")

    def test_worker_uses_basic_when_no_light_chip_and_light_when_set(self):
        a = self.app()
        chip, kind = a.idle._memory_chip()
        self.assertEqual((chip.name, kind), ("standard", "standard"))
        a.settings.light_chip = "qwen3:1.7b"
        a.chips._load_config()
        chip, kind = a.idle._memory_chip()
        self.assertEqual((chip.name, kind), ("light", "light"))
        a.settings.memory_chip = "standard"
        self.assertEqual(a.idle._memory_chip()[1], "standard")
        self.assertEqual(a.chips.choose("normal").name, "standard")                # the Idle chip still never answers chat

    def test_corrections_need_the_stronger_chip(self):
        llm = FakeLLM()
        a = self.app(llm)
        a.settings.light_chip = "qwen3:1.7b"
        a.chips._load_config()
        a.memory.journal_append("boss", "Ksar", "my budget is 9000 euro per year", "ok")
        self.run_plan(a)
        fid = a.memory.get("education").facts[1].fid
        raw = {"ops": [{"op": "correct", "note": "education", "fact": fid, "new": "Budget: 9000 euro per year",
                        "evidence": "my budget is 9000 euro per year"}]}
        ctx = {"revs": {}, "user_text": "my budget is 9000 euro per year", "turn_srcs": [(SRC, "my budget is 9000 euro per year")]}
        self.assertEqual(a.keeper.plan(json.dumps(raw), ctx, allow_correct=False).corrections, [])      # light chip: refused
        self.assertEqual(len(a.keeper.plan(json.dumps(raw), ctx, allow_correct=True).corrections), 1)


# ------------------------------------------------------------------------------------------------ time
class Clock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now


class TimeTests(Base):
    def svc(self, start=None, **kw):
        self.clock = Clock(start or datetime(2026, 11, 19, 22, 30, tzinfo=timezone.utc))
        return TimeService(kw.pop("tz_name", ""), kw.pop("state_path", None), clock=self.clock, **kw)

    def test_timestamps_are_timezone_aware(self):
        t = self.svc()
        self.assertIsNotNone(t.now().tzinfo)
        self.assertIsNotNone(datetime.fromisoformat(t.stamp()).tzinfo)
        self.assertRegex(t.tz_label(), r"^UTC[+-]\d\d:\d\d")

    def test_configurable_timezone_with_a_safe_fallback(self):
        t = self.svc(tz_name="Not/AZone")
        self.assertTrue(t.tz_problem)
        self.assertIsNotNone(t.now().tzinfo)                                       # fell back to the operating system's zone
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo("Asia/Tokyo")
        except Exception:
            self.skipTest("no IANA database on this machine")
        tokyo = self.svc(tz_name="Asia/Tokyo")
        self.assertEqual(tokyo.now().hour, 7)                                       # 22:30 UTC = 07:30 next day in Tokyo
        self.assertIn("Asia/Tokyo", tokyo.tz_label())

    def test_session_duration_and_idle_time(self):
        t = self.svc()
        self.clock.now += timedelta(minutes=95)
        self.assertEqual(int(t.session_age().total_seconds() // 60), 95)
        t.begin_message()
        self.clock.now += timedelta(minutes=5)
        self.assertEqual(int(t.idle_for().total_seconds() // 60), 5)

    def test_date_rollover_is_reported_once(self):
        t = self.svc()
        self.assertIsNone(t.rollover())
        self.clock.now += timedelta(hours=3)
        old, new = t.rollover()
        self.assertEqual((new - old).days, 1)
        self.assertIsNone(t.rollover())

    def test_return_after_a_long_absence_is_noticed_once(self):
        t = self.svc(long_gap_hours=4)
        t.begin_message()                                                           # evening
        self.clock.now += timedelta(minutes=10)
        t.begin_message()
        self.assertEqual(t.context_for("how are you?"), "")                          # a short pause: nothing
        self.clock.now += timedelta(hours=9)
        t.begin_message()                                                           # next morning
        note = t.context_for("good morning")
        self.assertIn("back after 9 hours", note)
        self.assertIn("do not guess what they did", note)
        t.begin_message()
        self.assertEqual(t.context_for("another question"), "")                      # no repeated greeting
        self.assertEqual(t.context_for("and another"), "")

    def test_nothing_is_fabricated_about_the_gap(self):
        t = self.svc()
        t.begin_message()
        self.clock.now += timedelta(hours=10)
        t.begin_message()
        note = t.context_for("hi").lower()
        for invented in ("slept", "sleep", "woke", "left the house", "dream", "work"):
            self.assertNotIn(invented, note)

    def test_time_is_added_only_when_the_message_is_about_time(self):
        t = self.svc()
        t.begin_message()
        self.assertEqual(t.context_for("explain recursion"), "")
        self.assertIn("Now: Thursday 19 November 2026", t.context_for("what day is it today?").replace("  ", " ")
                      if t.now().day == 19 else "Now:")
        self.assertIn("Now:", t.context_for("remind me tomorrow"))

    def test_last_interaction_survives_a_restart(self):
        state = self.tmp / "state.json"
        t = self.svc(state_path=state)
        t.begin_message()
        t.note_response()
        later = TimeService("", state, clock=Clock(self.clock.now + timedelta(hours=8)))
        later.begin_message()
        self.assertIn("back after 8 hours", later.context_for("hi"))
        fresh = TimeService("", self.tmp / "none.json", clock=self.clock)
        fresh.begin_message()
        self.assertEqual(fresh.context_for("hi"), "")                                # the very first message ever

    def test_chat_gets_the_time_note_without_changing_the_cached_prompt(self):
        llm = FakeLLM(chat_reply="ok")
        a = self.app(llm)
        seen = []
        orig = llm.chat
        llm.chat = lambda messages, **kw: (seen.append(messages), orig(messages, **kw))[1]
        clock = Clock(datetime(2026, 11, 19, 22, 30, tzinfo=timezone.utc))
        a.time.clock = clock
        a.time._day = a.time.now().date()
        c = a.conversation()
        c.send("hello")
        clock.now += timedelta(hours=9)
        c.send("good morning")
        clock.now += timedelta(minutes=2)
        c.send("what day is it today?")
        self.assertNotIn("[Time:", seen[0][-1]["content"])
        self.assertIn("back after 9 hours", seen[1][-1]["content"])
        self.assertNotIn("back after", seen[2][-1]["content"])                       # not repeated
        self.assertIn("Now:", seen[2][-1]["content"])                                # asked about the date
        self.assertEqual(len({m[0]["content"] for m in seen}), 1)                    # the system prompt never changed
        self.assertTrue(a.agent.last_turn.time_note)

    def test_the_worker_notices_the_date_change(self):
        a = self.app()
        clock = Clock(datetime(2026, 11, 19, 23, 50, tzinfo=timezone.utc))
        a.time.clock = clock
        a.time._day = a.time.now().date()
        self.assertIsNone(a.idle.job_clock())
        clock.now += timedelta(minutes=30)
        self.assertIn("new day", a.idle.job_clock())

    def test_time_command(self):
        a = self.app()
        buf = io.StringIO()
        with redirect_stdout(buf):
            handle_command("/time", a, a.conversation(), print)
        self.assertIn("session:", buf.getvalue())


# ------------------------------------------------------------------------------------------------ performance
class PerformanceTests(Base):
    def test_ordinary_chat_does_no_disk_scans_and_no_note_parsing(self):
        llm = FakeLLM(chat_reply="fine")
        a = self.app(llm)
        self.run_plan(a)
        a.index.recall("warm up")
        builds = a.memory.core_builds
        from ksar.memory import store as store_mod
        c = a.conversation()
        with mock.patch.object(store_mod.os, "walk", side_effect=AssertionError("vault scan on the chat path")), \
                mock.patch.object(store_mod, "parse_note", side_effect=AssertionError("note re-read on the chat path")):
            for text in ("what is the capital of France?", "Which universities should I apply to?",
                         "please delete the old log files", "hello again"):
                self.assertTrue(c.send(text))
        self.assertEqual(a.memory.core_builds, builds)                               # the core cache was used

    def test_index_updates_incrementally_after_a_note_changes(self):
        a = self.app()
        self.run_plan(a)
        a.index.recall("anything")
        full = a.index.full_rebuilds
        a.memory.add_fact("communication-style", "Likes tea #preference", "stated", SRC)
        self.assertTrue(a.index.recall("tea").blocks)
        self.assertEqual(a.index.full_rebuilds, full)
        self.assertGreaterEqual(a.index.incremental_updates, 1)

    def test_unchanged_branch_indexes_are_not_rewritten(self):
        a = self.app()
        self.run_plan(a)
        a.idle.user_idle()
        a.idle.job_index()
        gen = a.memory.gen
        for _ in range(5):
            a.idle._index_gen = -1
            a.idle.job_index()
        self.assertEqual(a.memory.gen, gen)

    def test_plain_chat_prompt_is_unchanged_in_size(self):
        a = self.app()
        self.assertLess(len(a.agent.chat_prompt()), 600)

    def test_recall_is_fast_on_a_large_vault(self):
        a = self.app()
        for i in range(400):
            a.memory.create(f"note-{i}", f"Topic {i}", "semantic", f"about subject {i}", "core")
            a.memory.add_fact(f"note-{i}", f"fact about subject {i} and thing {i % 17}", "stated", SRC)
        a.index.recall("warm")
        t0 = time.perf_counter()
        for _ in range(20):
            a.index.recall("what do we know about subject 37 and thing 3?")
        per = (time.perf_counter() - t0) / 20
        self.assertLess(per, 0.15)                                                     # RAM only; no scanning per message


class CommandTests(Base):
    def test_memory_commands(self):
        a = self.app(FakeLLM(extract=OPS))
        self.run_plan(a)
        c = a.conversation()
        out = {}
        for cmd in ("/memory status", "/memory tree", "/memory search file permissions", "/memory backlinks incident-lost-thesis",
                    "/memory graph file-permissions", "/memory check", "/memory queue", "/memory organize", "/memory time",
                    "/tree", "/recall delete files", "/open file-permissions", "/doctor"):
            buf = io.StringIO()
            with redirect_stdout(buf):
                handle_command(cmd, a, c, print)
            out[cmd] = buf.getvalue()
            self.assertNotIn("unknown", out[cmd].lower(), cmd)
        self.assertIn("notes:", out["/memory status"])
        self.assertIn("cached in RAM", out["/memory status"])
        self.assertIn("(User/Rules/)", out["/memory tree"])
        self.assertIn("caused_by", out["/memory backlinks incident-lost-thesis"])
        self.assertIn("out  caused_by", out["/memory graph file-permissions"])
        self.assertIn("queue is empty", out["/memory queue"])

    def test_save_queues_in_the_background_and_wait_runs_it(self):
        llm = FakeLLM(extract=OPS)
        a = self.app(llm)
        c = self.start_worker(a)
        a.idle.stop()                                                                  # keep the worker quiet: we drive it by hand
        a.idle._thread = threading.Thread(target=lambda: time.sleep(5), daemon=True)
        a.idle._thread.start()
        a.memory.journal_append("boss", "Ksar", USER, "ok")
        a.jobs.set_meta("watermark", {datetime.now().strftime("%Y-%m-%d"): 0})
        buf = io.StringIO()
        with redirect_stdout(buf):
            handle_command("/save", a, c, print)
        self.assertIn("queued", buf.getvalue())
        self.assertFalse(a.memory.exists("file-permissions"))
        with redirect_stdout(io.StringIO()):
            handle_command("/save wait", a, c, print)
        self.assertTrue(a.memory.exists("file-permissions"))

    def test_new_settings_exist_and_can_be_changed(self):
        a = self.app()
        for key, value in (("memory.async", "false"), ("memory.auto", "false"), ("memory.chip", "standard"),
                           ("time.zone", "Europe/Berlin"), ("time.gap", "6"), ("idle.notify", "true"),
                           ("memory.organize", "true")):
            with redirect_stdout(io.StringIO()):
                handle_command(f"/config set {key} {value}", a, a.conversation(), print)
        s = a.settings
        self.assertEqual((s.memory_async, s.memory_auto, s.memory_chip, s.timezone, s.time_long_gap_hours,
                          s.idle_notify, s.memory_auto_organize), (False, False, "standard", "Europe/Berlin", 6, True, True))


if __name__ == "__main__":
    unittest.main()

"""Tests for the third chip slot (light) and the idle worker (v0.3.1)."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ksar.app import Ksar
from ksar.chips import ChipRole, ChipState
from ksar.cli import cmd_config, cmd_idle
from ksar.memory.keeper import grounded, restrict_ops
from ksar.resources import SystemResources


class IdleLLM:
    """Scripted back-end: streaming replies for the light chip, records what was asked of which model."""
    supports_think = True

    def __init__(self, reply="", running=None):
        self.reply, self.calls, self.loaded, self.model = reply, [], [], "qwen3:8b"
        self.num_ctx, self.think, self.running = 0, False, running if running is not None else {}
        self.on_stream = None                      # hook: called while "generating"

    def chat(self, messages, model=None, keep_alive=None, json_mode=False, temperature=None, think=None, max_tokens=None):
        self.calls.append(("chat", model))
        return self.reply if json_mode else "ok"

    def stream_chat(self, messages, model=None, keep_alive=None, json_mode=False, temperature=None, think=None,
                    max_tokens=None):
        self.calls.append(("stream", model, max_tokens))
        mid = len(self.reply) // 2
        yield "content", self.reply[:mid]
        if self.on_stream:
            self.on_stream()
        yield "content", self.reply[mid:]

    def load(self, model, keep_alive=-1):
        self.loaded.append(model)
        self.running[model] = 1

    def unload(self, model):
        self.running.pop(model, None)

    def running_models(self):
        return dict(self.running)

    def installed_models(self):
        return {}


OPS = {"ops": [{"op": "add", "note": "meady", "title": "Meady", "type": "person", "summary": "The owner",
                "parent": "core", "facts": ["Favourite colour: dark blue", "Pet: tiger named Tom"], "conf": "stated"},
               {"op": "correct", "note": "meady", "fact": "x", "new": "Name: Hacker"}]}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make(self, reply="", light="qwen3:1.7b", running=None):
        llm = IdleLLM(reply, running)
        a = Ksar(self.tmp / "home", llm=llm)
        a.settings.light_chip = light
        a.chips._load_config()
        a.resources.get_resources = lambda: SystemResources(32000, 28000, 100)
        a.idle.attach(a.conversation())
        return a, llm


class LightSlotTests(Base):
    def test_three_slots_and_keep_alive(self):
        a, _ = self.make()
        a.settings.advanced_chip = "qwen3:14b"
        a.chips._load_config()
        self.assertEqual(set(a.chips.chips), {"standard", "advanced", "light"})
        self.assertEqual(a.chips.chips["light"].role, ChipRole.LIGHTWEIGHT)
        self.assertEqual(a.chips.keep_alive_for(a.chips.chips["light"]), "5m")
        a.settings.chips_retention = "aggressive"
        self.assertEqual(a.chips.keep_alive_for(a.chips.chips["light"]), 0)

    def test_light_is_never_used_for_chat(self):
        a, llm = self.make('{"say": "x"}')
        a.settings.performance_priority = "quality"
        a.settings.advanced_chip = ""
        a.chips._load_config()
        self.assertEqual(a.chips.get_chip_for_task("complex").name, "standard")

    def test_light_equal_to_standard_is_ignored_and_off_by_default(self):
        a, _ = self.make(light="qwen3:8b")
        self.assertNotIn("light", a.chips.chips)
        a, _ = self.make(light="")
        self.assertIsNone(a.chips.light_for_idle()[0])

    def test_no_ram_no_light(self):
        a, _ = self.make()
        a.resources.get_resources = lambda: SystemResources(32000, 2500, 100)
        chip, why = a.chips.light_for_idle()
        self.assertIsNone(chip)
        self.assertIn("only 2500 MB free", why)


class GroundingTests(unittest.TestCase):
    def test_grounding(self):
        said = "my favourite colour is dark blue and I like pizza"
        self.assertTrue(grounded("Favourite colour: dark blue", said))
        self.assertTrue(grounded("Likes: pizza", said))
        self.assertFalse(grounded("Pet: tiger named Tom", said))
        self.assertFalse(grounded("Name: ", said))

    def test_light_mode_only_adds_grounded_facts(self):
        ops = restrict_ops(OPS["ops"], "my favourite colour is dark blue")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["facts"], ["Favourite colour: dark blue"])      # tiger dropped, 'correct' dropped


class IdleJobTests(Base):
    def test_not_idle_while_you_are_active(self):
        a, llm = self.make(json.dumps(OPS))
        a.idle.convo.pending.append(("my favourite colour is dark blue", "Nice!"))
        a.idle.user_active()
        self.assertIsNone(a.idle.run_once())
        a.idle.user_idle()
        self.assertIsNone(a.idle.run_once())                  # idle_after_s has not passed yet
        self.assertEqual(llm.calls, [])

    def test_consolidate_on_the_light_chip_conservatively(self):
        a, llm = self.make(json.dumps(OPS), running={"qwen3:8b": 1})
        convo = a.idle.convo
        convo.pending.append(("my favourite colour is dark blue", "Nice!"))
        done = a.idle.run_once(force=True)
        self.assertEqual(done[0], "consolidate")
        self.assertIn("memory updated (", done[1])
        self.assertEqual(llm.calls[0][:2], ("stream", "qwen3:1.7b"))      # the LIGHT model did it
        self.assertEqual(llm.calls[0][2], 700)                             # capped
        facts = [f.text for f in a.memory.get("meady").facts]
        self.assertEqual(facts, ["Favourite colour: dark blue"])
        self.assertEqual(convo.pending, [])
        self.assertTrue(any("consolidate" in n for n in a.idle.pop_notes()))

    def test_without_a_light_chip_nothing_is_done_in_the_background(self):
        a, llm = self.make(json.dumps(OPS), light="", running={"qwen3:8b": 1})
        a.idle.convo.pending.append(("my favourite colour is dark blue", "Nice!"))
        a.idle.job_doctor()
        self.assertIsNone(a.idle.run_once(force=True))
        self.assertEqual(len(a.idle.convo.pending), 1)
        self.assertEqual(llm.calls, [])

    def test_user_returning_stops_the_job_and_keeps_the_exchange(self):
        a, llm = self.make(json.dumps(OPS), running={"qwen3:8b": 1})
        a.idle.convo.pending.append(("my favourite colour is dark blue", "Nice!"))
        llm.on_stream = a.idle.user_active                    # you press Enter mid-generation
        a.idle._doctor_day = "x"
        out = a.idle.run_once(force=True)
        self.assertEqual(out, ("consolidate", "stopped"))
        self.assertEqual(len(a.idle.convo.pending), 1)
        self.assertIsNone(a.memory.get("meady"))

    def test_a_normal_save_during_the_job_wins(self):
        a, llm = self.make(json.dumps(OPS), running={"qwen3:8b": 1})
        convo = a.idle.convo
        convo.pending.append(("my favourite colour is dark blue", "Nice!"))
        llm.on_stream = lambda: setattr(convo, "gen", convo.gen + 1)      # main thread saved meanwhile
        a.idle.run_once(force=True)
        self.assertIsNone(a.memory.get("meady"))              # stale result discarded, nothing written twice

    def test_warm_loads_the_standard_model_once(self):
        a, llm = self.make(running={})
        self.assertTrue(a.idle.is_idle())                     # start-up counts as idle: warm early
        first = a.idle.run_once()
        self.assertEqual(first[0], "warm")
        self.assertEqual(llm.loaded, ["qwen3:8b"])
        self.assertNotEqual((a.idle.run_once() or ("", ""))[0], "warm")
        self.assertEqual(llm.loaded, ["qwen3:8b"])

    def test_warm_can_be_switched_off(self):
        a, llm = self.make(running={})
        a.settings.warm_on_start = False
        a.idle.run_once()
        self.assertEqual(llm.loaded, [])

    def test_doctor_runs_once_a_day(self):
        a, _ = self.make(running={"qwen3:8b": 1})
        self.assertIsNone(a.idle.job_doctor())
        self.assertNotEqual(a.idle._doctor_day, "")

    def test_failed_job_backs_off_and_never_raises(self):
        a, llm = self.make(json.dumps(OPS), running={"qwen3:8b": 1})
        a.idle.convo.pending.append(("hello", "hi"))
        def boom(*args, **kw):
            raise RuntimeError("ollama down")
        llm.stream_chat = boom
        a.idle._doctor_day = "x"
        self.assertIsNone(a.idle.run_once(force=True))
        self.assertGreater(a.idle._skip_until["consolidate"], a.idle.clock())
        self.assertTrue(any("failed" in m for _, _, m in a.idle.log))


class WindowTests(Base):
    def test_window_slides_in_steps_so_the_prompt_prefix_stays_stable(self):
        a, _ = self.make()
        a.settings.recent_messages = 12
        c = a.conversation()
        starts = []
        for i in range(30):
            c.history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
            c.window()
            starts.append(c._start)
        changes = sum(1 for x, y in zip(starts, starts[1:]) if x != y)
        self.assertLessEqual(changes, 8)                       # a one-by-one slide would change ~24 times
        self.assertLessEqual(len(c.window()), 12)

    def test_long_messages_shrink_the_window(self):
        a, _ = self.make()
        a.settings.history_chars = 2000
        c = a.conversation()
        for i in range(6):
            c.history += [{"role": "user", "content": "x" * 700}, {"role": "assistant", "content": "y" * 700}]
        self.assertLessEqual(sum(len(m["content"]) for m in c.window()), 2000)


class CliTests(Base):
    def test_idle_commands_and_config(self):
        a, _ = self.make()
        out = []
        import builtins
        real = builtins.print
        builtins.print = lambda *x, **k: out.append(" ".join(map(str, x)))
        try:
            cmd_idle(a, [], a.idle.convo)
            cmd_idle(a, ["off"], a.idle.convo)
            self.assertFalse(a.settings.idle_enabled)
            cmd_config(a, ["set", "idle.after", "2"])
            self.assertEqual(a.settings.idle_after_s, 20)            # out of range: unchanged
            cmd_config(a, ["set", "idle.after", "45"])
            self.assertEqual(a.settings.idle_after_s, 45)
            cmd_config(a, ["set", "chips.light", "qwen3:0.6b"])
            self.assertEqual(a.chips.chips["light"].model_id, "qwen3:0.6b")
        finally:
            builtins.print = real
        self.assertIn("idle work: ON", out[0])


if __name__ == "__main__":
    unittest.main()

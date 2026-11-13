"""Tests for the LLM wrapper, resource estimates, chip routing and the v0.2.1 fixes."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ksar.agent import classify_complexity
from ksar.app import Ksar
from ksar.chips import ChipState
from ksar.config import Settings
from ksar.llm import LLM, LLMError, extract_json, strip_think
from ksar.resources import ResourceManager, SystemResources, estimate_model_ram_mb


class Rec:
    """LLM stand-in that records every call (and supports load/unload like the real one)."""
    def __init__(self, replies=()):
        self.replies, self.calls, self.loaded, self.model, self.num_ctx, self.think = list(replies), [], [], "x", 0, False

    def chat(self, messages, model=None, keep_alive=None, json_mode=False, temperature=None):
        self.calls.append({"model": model, "keep_alive": keep_alive})
        return self.replies.pop(0) if self.replies else '{"say": "ok"}'

    def load(self, model, keep_alive=-1):
        self.loaded.append((model, keep_alive))


class LLMWrapperTests(unittest.TestCase):
    def test_settings_fields_are_honoured(self):
        s = Settings(ollama_url="http://example:1234/", request_timeout=900, num_ctx=8192, standard_chip="qwen3:14b")
        llm = LLM(s)
        self.assertEqual((llm.base_url, llm.timeout, llm.num_ctx, llm.model), ("http://example:1234", 900, 8192, "qwen3:14b"))

    def test_payload_has_num_ctx_think_off_and_keep_alive(self):
        llm = LLM(Settings())
        with mock.patch.object(llm, "_request", return_value={"message": {"content": "hi"}}) as req:
            llm.chat([{"role": "user", "content": "x"}], keep_alive="30m", json_mode=True)
        payload = req.call_args[0][1]
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertIs(payload["think"], False)
        self.assertEqual((payload["keep_alive"], payload["format"]), ("30m", "json"))

    def test_think_blocks_stripped(self):
        self.assertEqual(strip_think("<think>hmm</think>\n{\"say\": \"a\"}"), '{"say": "a"}')
        self.assertEqual(strip_think("<think>never closed"), "")

    def test_extract_json_handles_braces_in_text_and_two_objects(self):
        self.assertEqual(extract_json('ok {"say": "use } and { freely"} bye'), {"say": "use } and { freely"})
        self.assertEqual(extract_json('{"a": 1} then {"b": 2}'), {"a": 1})
        self.assertEqual(extract_json("```json\n{\"say\": \"x\"}\n```"), {"say": "x"})
        self.assertIsNone(extract_json("no json here"))

    def test_http_404_gives_pull_hint(self):
        import io, urllib.error
        err = urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO(b'{"error":"model not found"}'))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(LLMError) as cm:
                LLM(Settings()).chat([{"role": "user", "content": "x"}], model="qwen3:32b")
        self.assertIn("ollama pull qwen3:32b", str(cm.exception))


class ResourceTests(unittest.TestCase):
    def test_estimates(self):
        self.assertTrue(5000 < estimate_model_ram_mb("qwen3:8b") < 8000)
        self.assertTrue(19000 < estimate_model_ram_mb("qwen3:32b") < 24000)
        self.assertGreater(estimate_model_ram_mb("llama3:70b"), 40000)
        self.assertLess(estimate_model_ram_mb("x", installed_bytes=5 * 2**30), 7000)   # real size wins

    def test_unknown_ram_is_not_silently_allowed(self):
        rm = ResourceManager(Path("."))
        with mock.patch.object(rm, "get_resources", return_value=SystemResources()):
            ok, why = rm.explain(1000)
        self.assertFalse(ok)
        self.assertIn("cannot read free RAM", why)

    def test_enough_and_not_enough(self):
        rm = ResourceManager(Path("."))
        with mock.patch.object(rm, "get_resources", return_value=SystemResources(32000, 20000)):
            self.assertTrue(rm.can_load_chip(10000))
            self.assertFalse(rm.can_load_chip(18000))        # 18000 + 3072 reserve > 20000


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def app(self, replies=(), advanced="qwen3:14b", **settings):
        llm = Rec(replies)
        a = Ksar(self.tmp / "home", llm=llm)
        a.settings.advanced_chip = advanced
        for k, v in settings.items():
            setattr(a.settings, k, v)
        a.chips._load_config()
        a.resources.get_resources = lambda: SystemResources(32000, 28000, 100)
        return a, llm

    def test_complexity_only_from_human_text(self):
        self.assertEqual(classify_complexity("my project folder"), "normal")       # 'project' no longer escalates
        self.assertEqual(classify_complexity("please debug this"), "complex")
        self.assertEqual(classify_complexity("сделай план"), "complex")
        self.assertEqual(classify_complexity("x" * 800), "complex")

    def test_normal_chat_uses_standard_and_passes_keep_alive(self):
        a, llm = self.app()
        a.conversation().send("hello")
        self.assertEqual(llm.calls[0]["model"], "qwen3:8b")
        self.assertEqual(llm.calls[0]["keep_alive"], "30m")
        self.assertEqual(llm.loaded, [])

    def test_complex_escalates_once_per_turn_and_loads_advanced(self):
        a, llm = self.app([json.dumps({"tool": "memory.search", "args": {"query": "x"}}), '{"say": "done"}'])
        a.conversation().send("analyze my notes")
        models = [c["model"] for c in llm.calls[:2]]
        self.assertEqual(models, ["qwen3:14b", "qwen3:14b"])           # same chip for every step of the turn
        self.assertEqual(llm.loaded[0][0], "qwen3:14b")

    def test_speed_priority_never_escalates(self):
        a, llm = self.app(performance_priority="speed")
        a.conversation().send("analyze everything")
        self.assertEqual(llm.calls[0]["model"], "qwen3:8b")

    def test_no_ram_means_standard_chip_and_a_reason(self):
        a, llm = self.app()
        a.resources.get_resources = lambda: SystemResources(32000, 4000, 100)
        a.conversation().send("analyze everything")
        self.assertEqual(llm.calls[0]["model"], "qwen3:8b")
        self.assertIn("only 4000 MB free", a.chips.chips["advanced"].last_error)

    def test_failed_load_is_not_retried_every_step(self):
        a, llm = self.app()
        def boom(model, keep_alive=-1):
            raise LLMError("not pulled")
        llm.load = boom
        a.chips.get_chip_for_task("complex")
        a.chips.get_chip_for_task("complex")
        self.assertEqual(a.chips.chips["advanced"].last_error, "not pulled")
        self.assertGreater(a.chips.chips["advanced"].failed_at, 0)

    def test_retention_maps_to_keep_alive(self):
        a, _ = self.app(chips_retention="persistent")
        self.assertEqual(a.chips.keep_alive_for(a.chips.chips["standard"]), -1)
        a.settings.chips_retention = "aggressive"
        self.assertEqual(a.chips.keep_alive_for(a.chips.chips["advanced"]), 0)

    def test_old_settings_file_keeps_its_model(self):
        home = self.tmp / "old"
        (home / "config").mkdir(parents=True)
        (home / "config" / "settings.json").write_text(json.dumps({"model": "llama3.1:8b"}))
        a = Ksar(home, llm=Rec())
        self.assertEqual(a.settings.standard_chip, "llama3.1:8b")
        saved = json.loads((home / "config" / "settings.json").read_text())
        self.assertIn("advanced_chip", saved)                         # new keys written back

    def test_permissions_file_not_rewritten_every_call(self):
        a, _ = self.app()
        f = a.perms.file
        before = f.stat().st_mtime_ns
        for _ in range(5):
            a.tools.call("sys.info", {})
        self.assertEqual(f.stat().st_mtime_ns, before)


if __name__ == "__main__":
    unittest.main()

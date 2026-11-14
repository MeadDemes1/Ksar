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
from ksar.llm import LLM, LLMError, SayExtractor, ThinkSplitter, extract_json, strip_think
from ksar.resources import ResourceManager, SystemResources, estimate_model_ram_mb


class Rec:
    """LLM stand-in that records every call (and supports load/unload like the real one)."""
    def __init__(self, replies=()):
        self.replies, self.calls, self.loaded, self.model, self.num_ctx, self.think = list(replies), [], [], "x", 0, False

    def chat(self, messages, model=None, keep_alive=None, json_mode=False, temperature=None, think=None):
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
        a, llm = self.app([json.dumps({"tool": "fs.list", "args": {}}), '{"say": "done"}'])
        a.conversation().send("analyze the files in my workspace")
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


class StreamLLM(Rec):
    """Streams a scripted list of (kind, text) pieces, like Ollama does."""
    supports_think = True

    def __init__(self, pieces):
        super().__init__()
        self.pieces, self.seen = pieces, []

    def stream_chat(self, messages, model=None, keep_alive=None, json_mode=False, temperature=None, think=None):
        self.seen.append({"messages": messages, "json_mode": json_mode, "think": think})
        yield from self.pieces
        yield "stats", {"eval_count": 10, "eval_duration": 1_000_000_000}


class FakeOllama:
    """A tiny local HTTP server that streams NDJSON like Ollama's /api/chat."""

    def __init__(self, lines):
        import http.server, threading
        outer = self
        outer.lines, outer.requests = lines, []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for ln in outer.lines:
                    self.wfile.write((json.dumps(ln) + "\n").encode())
                    self.wfile.flush()

            def log_message(self, *a):
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class StreamingHttpTests(unittest.TestCase):
    def _run(self, lines, **kw):
        srv = FakeOllama(lines)
        self.addCleanup(srv.close)
        llm = LLM(Settings(ollama_url=srv.url))
        return list(llm.stream_chat([{"role": "user", "content": "hi"}], **kw)), srv

    def test_thinking_field_and_content(self):
        out, srv = self._run([{"message": {"thinking": "hm"}}, {"message": {"content": "Hel"}},
                              {"message": {"content": "lo"}}, {"done": True, "eval_count": 5, "eval_duration": 10**9}],
                             think=True)
        self.assertEqual([(k, t) for k, t in out if k != "stats"], [("thinking", "hm"), ("content", "Hel"), ("content", "lo")])
        self.assertEqual(out[-1][1]["eval_count"], 5)
        self.assertTrue(srv.requests[0]["stream"] and srv.requests[0]["think"] is True)

    def test_inline_think_tags_from_older_ollama(self):
        out, _ = self._run([{"message": {"content": "<think>pl"}}, {"message": {"content": "an</think>Yes"}},
                            {"done": True}])
        self.assertEqual([(k, t) for k, t in out if k != "stats"], [("thinking", "pl"), ("thinking", "an"), ("content", "Yes")])

    def test_error_line_raises(self):
        srv = FakeOllama([{"error": "boom"}])
        self.addCleanup(srv.close)
        with self.assertRaises(LLMError):
            list(LLM(Settings(ollama_url=srv.url)).stream_chat([{"role": "user", "content": "x"}]))


class SpeedDesignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_splitter_and_say_extractor(self):
        sp, out = ThinkSplitter(), []
        for ch in ["<th", "ink>hmm ", "ok</thi", "nk>\n\nHel", "lo"]:
            out += sp.feed(ch)
        out += sp.flush()
        self.assertEqual("".join(t for k, t in out if k == "thinking"), "hmm ok")
        self.assertEqual("".join(t for k, t in out if k == "content"), "\n\nHello")
        ex, acc, full = SayExtractor(), "", '{"say": "Hi \\"you\\"\\nok \\u0416"}'
        for i in range(0, len(full), 4):
            acc += ex.feed(full[i:i + 4])
        self.assertEqual(acc, 'Hi "you"\nok \u0416')
        self.assertEqual(SayExtractor().feed('{"tool": "fs.list", "args": {}}'), "")

    def test_routing_is_selective(self):
        from ksar.agent import route_message
        names = ["fs.list", "fs.read_text", "fs.write_text", "sys.info", "memory.search", "memory.remember"]
        r = lambda t: route_message(t, names)
        for chat in ("hi, how are you?", "write me a poem about the sea", "make a plan for my week",
                     "read a good book", "explain recursion"):
            self.assertEqual((r(chat).tools, r(chat).memory), ([], False), chat)
        self.assertTrue(r("what do you know about me?").memory)
        self.assertEqual(r("what do you know about me?").tools, [])        # memory text is injected, no tool round-trip
        self.assertEqual(r("use fs.list please").tools, ["fs.list"])        # a tool answers to its own name
        self.assertIn("fs.read_text", r("edit notes.txt with fs.write_text").tools)
        self.assertEqual(r("read notes.txt").tools[:1], ["fs.list"])
        self.assertFalse([t for t in r("read notes.txt").tools if t.startswith("memory.")])
        self.assertTrue(r("@memory anything").tools)
        self.assertEqual(r("сколько времени?").tools, ["sys.info"])

    def _app(self, llm):
        a = Ksar(self.tmp / "home", llm=llm)
        a.resources.get_resources = lambda: SystemResources(32000, 28000, 100)
        return a

    def test_plain_chat_streams_thoughts_and_answer_with_a_tiny_prompt(self):
        llm = StreamLLM([("thinking", "let me think"), ("content", "Hel"), ("content", "lo!")])
        a = self._app(llm)
        a.settings.think = True
        got = []
        reply = a.conversation().send("hi there", on_token=lambda k, t: got.append((k, t)))
        self.assertEqual(reply, "Hello!")
        self.assertEqual([k for k, _ in got], ["thinking", "content", "content", "stats"])
        seen = llm.seen[0]
        self.assertFalse(seen["json_mode"])
        self.assertTrue(seen["think"])
        system = seen["messages"][0]["content"]
        self.assertLess(len(system), 600)                    # was several thousand characters
        self.assertNotIn("TOOLS", system)
        self.assertNotIn("MEMORY", system)

    def test_memory_is_read_only_when_asked_for(self):
        llm = StreamLLM([("content", "ok")])
        a = self._app(llm)
        a.memory.create("pets", "Pets", "topic", "my pets")
        a.memory.add_fact("pets", "Cat: Tom", "stated", "chat:test")
        a.index.rebuild() if hasattr(a.index, "rebuild") else None
        tok = lambda k, t: None
        a.conversation().send("tell me a joke", on_token=tok)
        self.assertNotIn("Cat: Tom", json.dumps(llm.seen[-1]["messages"]))
        a.conversation().send("what do you know about my cat?", on_token=tok)
        self.assertIn("Cat: Tom", llm.seen[-1]["messages"][-1]["content"])

    def test_tool_mode_streams_only_the_say_text_and_limits_tools(self):
        llm = StreamLLM([("content", '{"tool": "memory.forget", "args": {"note": "x"}}')])
        a = self._app(llm)
        events = []
        a.conversation().send("list the files in my workspace", events.append, lambda k, t: None)
        self.assertTrue(any("not available" in e for e in events))          # not offered for this message
        system = llm.seen[0]["messages"][0]["content"]
        self.assertIn("fs.list", system)
        self.assertNotIn("memory.forget", system)

    def test_say_is_streamed_in_tool_mode(self):
        llm = StreamLLM([("content", '{"say": "He'), ("content", 'llo"}')])
        a = self._app(llm)
        got = []
        reply = a.conversation().send("list the files in my workspace", on_token=lambda k, t: got.append((k, t)))
        self.assertEqual(reply, "Hello")
        self.assertEqual("".join(t for k, t in got if k == "content"), "Hello")


if __name__ == "__main__":
    unittest.main()

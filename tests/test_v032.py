"""Tests for 0.3.2: calculator, manual chip selection (/use), structural routing, diagnostics, integration."""
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ksar.app import Ksar
from ksar.chips import ChipState
from ksar.cli import cmd_use, handle_command
from ksar.resources import SystemResources
from ksar.router import assess, find_expressions, route_message
from ksar.tools.base import Risk
from ksar.tools.calc import CalcError, calc_evaluate, evaluate, fmt


def ev(expr):
    return fmt(evaluate(expr))


class CalculatorTests(unittest.TestCase):
    def test_arithmetic_and_precedence(self):
        self.assertEqual(ev("2+3*4"), "14")
        self.assertEqual(ev("(2+3)*4"), "20")
        self.assertEqual(ev("17*23+45/9-3"), "393")
        self.assertEqual(ev("2**3**2"), "512")                 # right associative like Python
        self.assertEqual(ev("-3**2"), "-9")
        self.assertEqual(ev("(-3)**2"), "9")
        self.assertEqual(ev("10-4-3"), "3")

    def test_division_family_and_exactness(self):
        self.assertEqual(ev("7/2"), "7/2 (= 3.5)")
        self.assertEqual(ev("6/3"), "2")
        self.assertEqual(ev("7//2"), "3")
        self.assertEqual(ev("-7//2"), "-4")
        self.assertEqual(ev("-7%3"), "2")
        self.assertEqual(ev("2**-2"), "1/4 (= 0.25)")
        self.assertEqual(ev("1/3*3"), "1")                     # exact, no float drift

    def test_negative_decimal_and_friendly_syntax(self):
        self.assertEqual(ev("-2.5*4"), "-10")
        self.assertEqual(ev("0.5+0.25"), "0.75")
        self.assertEqual(ev("2^10"), "1024")                   # ^ means power, not XOR
        self.assertEqual(ev("3×4÷2"), "6")
        self.assertEqual(ev("1_000*3"), "3000")
        self.assertEqual(ev("5 - -3"), "8")

    def test_functions_and_constants(self):
        self.assertEqual(ev("sqrt(144)"), "12")
        self.assertEqual(ev("factorial(10)"), "3628800")
        self.assertEqual(ev("gcd(12,18)"), "6")
        self.assertEqual(ev("max(3,9,4)"), "9")
        self.assertEqual(ev("round(2.5)"), "2")
        self.assertTrue(ev("pi*2").startswith("6.28318530717959"))
        self.assertTrue(ev("sin(pi/2)").startswith("1"))
        self.assertEqual(ev("comb(10,3)"), "120")

    def test_big_integers_are_exact(self):
        self.assertEqual(ev("2**200"), str(2 ** 200))
        self.assertEqual(ev("factorial(30)"), "265252859812191058636308480000000")

    def test_errors_are_clear_not_crashes(self):
        for bad, word in [("1/0", "division by zero"), ("5%0", "modulo by zero"), ("5//0", "division by zero"),
                          ("0**-1", "division by zero"), ("(1+", "invalid syntax"), ("2*", "invalid syntax"),
                          ("", "empty"), ("sqrt(-1)", "domain"), ("ln(0)", "domain"), ("(-8)**0.5", "not a real number"),
                          ("1e999*1e999", "not finite"), ("x+1", "unknown name")]:
            with self.assertRaises(CalcError, msg=bad) as cm:
                evaluate(bad)
            self.assertIn(word, str(cm.exception), bad)

    def test_unsupported_python_is_rejected(self):
        for bad in ["lambda: 1", "1 if 1 else 2", "[1, 2]", "(1, 2)", "{1: 2}", "'a' * 3", "True + 1", "1j",
                    "a.b", "x := 5", "not 1", "1 < 2", "1 and 2", "f'{1}'", "...", "print(1)", "abs(-1, 2)",
                    "abs(x=1)", "sqrt(*[4])", "[x for x in (1, 2)]"]:
            with self.assertRaises(CalcError, msg=bad):
                evaluate(bad)

    def test_malicious_constructs_never_execute(self):
        for bad in ["__import__('os').system('echo pwned')", "().__class__.__bases__[0].__subclasses__()",
                    "open('/etc/passwd').read()", "exec('x=1')", "eval('1+1')", "(lambda: 1)()",
                    "getattr(1, '__class__')", "globals()", "[c for c in ().__class__.__mro__]",
                    "__builtins__", "abs.__call__(1)", "compile('1', 'f', 'eval')"]:
            with self.assertRaises(CalcError, msg=bad):
                evaluate(bad)

    def test_excessive_calculations_are_refused(self):
        for bad in ["2**100000", "9**9**9", "10**10000*10**10000", "factorial(100000)", "comb(100000, 50000)",
                    "2*" * 300 + "2", "1+" * 300 + "1", "e**1000", "sin(1e400)", "7**7**7"]:
            with self.assertRaises(CalcError, msg=bad[:30]):
                evaluate(bad)
        with self.assertRaises(CalcError):                       # tree too big although the text is short
            evaluate("+".join(["1"] * 200) + "+" + "-".join(["1"] * 100))
        deep = "(" * 200 + "1" + ")" * 200                      # parser limit, must not crash Ksar
        try:
            evaluate(deep)
        except CalcError:
            pass

    def test_tool_wrapper(self):
        r = calc_evaluate(None, {"expression": "12*(3+4)/5"})
        self.assertTrue(r.ok)
        self.assertIn("12*(3+4)/5 = 84/5 (= 16.8)", r.data)
        self.assertFalse(r.untrusted)                            # must not taint the session
        r = calc_evaluate(None, {"expressions": "1+1; 2*3\n4**2"})
        self.assertTrue(r.ok)
        self.assertIn("2*3 = 6", r.data)
        self.assertIn("4**2 = 16", r.data)
        r = calc_evaluate(None, {"expressions": "1+1; 1/0"})
        self.assertFalse(r.ok)
        self.assertIn("ERROR: division by zero", r.data)
        self.assertIn("1+1 = 2", r.data)                         # the good one is still answered
        self.assertFalse(calc_evaluate(None, {}).ok)
        self.assertFalse(calc_evaluate(None, {"expressions": ";".join(["1"] * 11)}).ok)

    def test_registered_as_read_risk_without_approval(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        a = Ksar(tmp / "home", llm=Scripted([]))
        self.assertEqual(a.tools.tools["calc.evaluate"].risk, Risk.READ)
        a.tools.approver = lambda req: self.fail("calc.evaluate must never ask for approval")
        res = a.tools.call("calc.evaluate", {"expression": "2+2"})
        self.assertTrue(res.ok)
        self.assertFalse(a.session.tainted)


# ------------------------------------------------------------------------------------------ fake back-end
class Scripted:
    """Chat model stand-in: replays scripted replies and records every call and message list."""
    supports_think = True

    def __init__(self, replies, installed=None, running=None):
        self.replies, self.calls, self.sent = list(replies), [], []
        self.installed = installed if installed is not None else {"qwen3:8b": 5 * 2**30, "qwen3:14b": 9 * 2**30}
        self.running, self.loaded = running if running is not None else {"qwen3:8b": 1}, []
        self.model, self.num_ctx, self.think = "qwen3:8b", 0, False

    def chat(self, messages, model=None, keep_alive=None, json_mode=False, temperature=None, think=None, max_tokens=None):
        self.calls.append(model)
        self.sent.append(list(messages))                          # a snapshot: the agent keeps appending to its list
        return self.replies.pop(0) if self.replies else '{"say": "ok"}'

    def load(self, model, keep_alive=-1):
        self.loaded.append(model)
        self.running[model] = 1

    def unload(self, model):
        self.running.pop(model, None)

    def running_models(self):
        return dict(self.running)

    def installed_models(self):
        return dict(self.installed)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make(self, replies=(), advanced="qwen3:14b", ram=(32000, 28000), **kw):
        llm = Scripted(replies, **kw)
        a = Ksar(self.tmp / "home", llm=llm)
        a.settings.advanced_chip = advanced
        a.chips._load_config()
        self.ram = list(ram)
        a.resources.get_resources = lambda: SystemResources(self.ram[0], self.ram[1], 100)
        return a, llm


class ManualChipTests(Base):
    def test_use_basic_and_advanced_and_auto(self):
        a, _ = self.make()
        cm = a.chips
        self.assertEqual(cm.set_override("basic")[0], True)
        self.assertEqual(cm.override, "standard")
        self.assertTrue(cm.set_override("advanced")[0])
        self.assertEqual(cm.override, "advanced")
        self.assertTrue(cm.set_override("AUTO")[0])
        self.assertIsNone(cm.override)
        self.assertTrue(cm.set_override("standard")[0])          # internal name works too
        self.assertEqual(cm.override, "standard")

    def test_override_applies_to_exactly_one_message(self):
        a, llm = self.make()
        a.chips.set_override("advanced")
        c = a.conversation()
        c.send("hello there")
        self.assertEqual(llm.calls[-1], "qwen3:14b")
        self.assertIsNone(a.chips.override)                      # used up
        c.send("and again")
        self.assertEqual(llm.calls[-1], "qwen3:8b")               # back to automatic routing
        self.assertEqual(a.agent.last_turn.mode, "auto")

    def test_commands_do_not_consume_the_override(self):
        a, _ = self.make()
        c = a.conversation()
        a.chips.set_override("advanced")
        buf = io.StringIO()
        with redirect_stdout(buf):
            for cmd in ("/tools", "/why", "/chip status", "/chip active", "/use status", "/think", "/idle",
                        "/perms", "/config show", "/fs.list"):
                handle_command(cmd, a, c, print)
        self.assertEqual(a.chips.override, "advanced")

    def test_manual_beats_complexity_and_priority(self):
        a, llm = self.make(performance_priority="speed") if False else self.make()
        a.settings.performance_priority = "speed"                # speed never escalates automatically...
        a.chips.set_override("advanced")
        a.conversation().send("hi")                              # ...but a manual choice wins, even for a trivial message
        self.assertEqual(llm.calls[-1], "qwen3:14b")
        a.settings.performance_priority = "quality"              # quality always escalates automatically...
        a.chips.set_override("basic")
        a.conversation().send("please debug this architecture")  # ...but a manual Basic wins even for a hard one
        self.assertEqual(llm.calls[-1], "qwen3:8b")
        self.assertEqual(a.agent.last_turn.mode, "manual")

    def test_unavailable_advanced_never_creates_an_override(self):
        a, _ = self.make(advanced="")
        ok, msg = a.chips.set_override("advanced")
        self.assertFalse(ok)
        self.assertIn("no Advanced chip is configured", msg)
        self.assertIsNone(a.chips.override)
        a, _ = self.make(installed={"qwen3:8b": 1})
        ok, msg = a.chips.set_override("advanced")
        self.assertFalse(ok)
        self.assertIn("ollama pull qwen3:14b", msg)
        self.assertIsNone(a.chips.override)

    def test_insufficient_ram_is_reported_and_not_bypassed(self):
        a, llm = self.make(running={"qwen3:8b": 1}, ram=(32000, 6000))
        ok, msg = a.chips.set_override("advanced")
        self.assertFalse(ok)
        self.assertIn("cannot be loaded safely", msg)
        self.assertIn("only 6000 MB free", msg)
        self.assertIsNone(a.chips.override)
        self.assertEqual(llm.loaded, [])                         # nothing was loaded

    def test_ram_can_vanish_between_command_and_message(self):
        a, llm = self.make(replies=["fine"])
        self.assertTrue(a.chips.set_override("advanced")[0])
        self.ram[1] = 5000                                       # memory gets used by something else
        c = a.conversation()
        reply = c.send("hello")
        self.assertIn("I did not answer", reply)
        self.assertIn("cannot be loaded safely", reply)
        self.assertEqual(llm.calls, [])                          # NOT silently answered by Basic
        self.assertEqual(c.history, [])                          # nothing recorded
        self.assertIsNone(a.chips.override)                      # used up: no broken override stays behind
        self.assertEqual(llm.loaded, [])
        c.send("try again")
        self.assertEqual(llm.calls[-1], "qwen3:8b")              # automatic routing works again

    def test_light_chip_is_never_a_chat_chip(self):
        a, llm = self.make()
        a.settings.light_chip = "qwen3:1.7b"
        a.chips._load_config()
        for word in ("light", "idle"):
            ok, msg = a.chips.set_override(word)
            self.assertFalse(ok)
            self.assertIn("reserved for background work", msg)
        a.settings.performance_priority = "quality"
        for text in ("hi", "please debug this architecture", "x" * 900):
            self.assertNotEqual(a.chips.choose("complex" if len(text) > 5 else "normal").name, "light")
        a.chips.override = "light"                               # even a forced bad state cannot reach the light chip
        choice = a.chips.choose("normal")
        self.assertNotEqual(choice.chip and choice.chip.name, "light")

    def test_unknown_word_and_status(self):
        a, _ = self.make()
        self.assertFalse(a.chips.set_override("turbo")[0])
        self.assertIn("auto", a.chips.override_status())
        a.chips.set_override("advanced")
        self.assertIn("next message goes to Advanced", a.chips.override_status())
        buf = io.StringIO()
        with redirect_stdout(buf):
            cmd_use(a, ["status"])
            cmd_use(a, ["turbo"])                                # failed request leaves the pending one alone
        self.assertEqual(a.chips.override, "advanced")
        self.assertIn("(unchanged)", buf.getvalue())

    def test_auto_routing_reports_why_advanced_was_not_used(self):
        a, llm = self.make(advanced="")
        a.conversation().send("please analyze and debug the architecture")
        t = a.agent.last_turn
        self.assertEqual((t.chip, t.mode), ("standard", "auto"))
        self.assertIn("no Advanced chip is configured", t.note)
        a, llm = self.make(ram=(32000, 5000), running={"qwen3:8b": 1})
        a.conversation().send("please analyze and debug the architecture")
        self.assertIn("Advanced not available", a.agent.last_turn.note)
        self.assertEqual(llm.calls[-1], "qwen3:8b")              # automatic mode keeps working with Basic


class RoutingTests(unittest.TestCase):
    TOOLS = ["calc.evaluate", "fs.list", "fs.read_text", "fs.write_text", "sys.info", "memory.search"]

    def r(self, text):
        return route_message(text, self.TOOLS)

    def test_short_mathematical_problems_can_be_complex(self):
        for text in ["A train travels 120 km in 1.5 hours, then 90 km in 2 hours. What is its average speed?",
                     "Solve 3x + 7 = 22 then compute x^2 * 4 - 1",
                     "compute 2*3+4*5-6/2+7*8-1"]:
            self.assertEqual(self.r(text).complexity.level, "complex", text)

    def test_numerical_programming_problem(self):
        text = ("Write a function that returns the n-th Fibonacci number modulo 1e9+7, where n <= 10^5. "
                "For n=10 it should return 55.")
        r = self.r(text)
        self.assertEqual(r.complexity.level, "complex")
        self.assertTrue(r.calc_reason)
        self.assertIn("calc.evaluate", r.tools)

    def test_code_with_input_output_examples(self):
        r = self.r("Input:\n3\n1 2 3\nOutput:\n6\nwrite code that sums the numbers")
        self.assertEqual(r.complexity.level, "complex")
        self.assertIn("calc.evaluate", r.tools)
        self.assertIn("input/output examples", r.complexity.why())

    def test_debugging_code_is_complex_explaining_code_is_not(self):
        code = "```python\ndef f(a):\n    return a + 1\n```\n"
        self.assertEqual(self.r(code + "this gives a wrong output, fix the bug").complexity.level, "complex")
        self.assertEqual(self.r(code + "what does this do?").complexity.level, "normal")

    def test_ordinary_messages_with_numbers_stay_basic(self):
        for text in ["I'm 25 years old and live in room 12", "Meeting at 14:30 on 2026-11-20 in room 5, ok?",
                     "call 555-1234 or 555-9876 tomorrow?", "The price is 10-20 dollars", "Is 10-20 dollars ok?",
                     "version 1.2.3 was released on 2026-11-18", "I have 2 cats, 3 dogs and 4 fish, how many animals?",
                     "Tomorrow I have to buy milk and the budget is 25 dollars", "hi, how are you?"]:
            r = self.r(text)
            self.assertEqual(r.complexity.level, "normal", text)
            self.assertEqual(r.preeval, [], text)
            self.assertNotIn("calc.evaluate", [t for t in r.tools if r.calc_reason == ""], text)

    def test_simple_programming_questions_stay_basic(self):
        for text in ["how do I reverse a list in python?", "write a python function that reverses a string",
                     "explain recursion", "what is a decorator?"]:
            self.assertEqual(self.r(text).complexity.level, "normal", text)

    def test_complex_without_the_usual_keywords(self):
        text = ("First open the box, then count the items, then label them, finally report the total. "
                "It must be exactly 12 and at most 20 and only if nothing is broken.")
        self.assertEqual(self.r(text).complexity.level, "complex")
        self.assertEqual(self.r(text).tools, [])                 # complex does not mean tools
        numbered = ("Please do this:\n1. collect the data\n2. clean it, but only if it has at least 10 rows\n"
                    "3. summarise it in exactly 5 lines\n4. the report must be at most one page")
        self.assertEqual(self.r(numbered).complexity.level, "complex")
        self.assertEqual(self.r("Please:\n1. buy milk\n2. buy bread\n3. call mum").complexity.level, "normal")

    def test_calculator_is_offered_only_when_relevant(self):
        self.assertEqual(self.r("hello").tools, [])
        self.assertEqual(self.r("write me a poem about autumn").tools, [])
        self.assertIn("calc.evaluate", self.r("What is 17*23+45/9-3?").tools)
        self.assertIn("calc.evaluate", self.r("what is 15% of 240?").tools)
        self.assertEqual(self.r("use calc.evaluate please").tools, ["calc.evaluate"])
        self.assertEqual(self.r("@calc 2+2").tools, ["calc.evaluate"])
        self.assertNotIn("calc.evaluate", route_message("17*23", ["fs.list"]).tools)     # not registered: not offered

    def test_pre_evaluation_is_exact_and_careful(self):
        self.assertEqual(self.r("What is 17*23+45/9-3?").preeval, [("17*23+45/9-3", "393")])
        self.assertEqual(self.r("What is 100-37?").preeval, [("100-37", "63")])
        self.assertEqual(self.r("Solve 3x + 7 = 22 then compute x^2 * 4 - 1").preeval, [])   # never the tail of x^2*4-1
        self.assertEqual(find_expressions("calculate sqrt(144) + 2**5"), [])                   # never half a function call
        self.assertEqual(find_expressions("on 2026-11-20 at 14:30"), [])

    def test_router_cost_is_bounded_on_hostile_input(self):
        import time
        nasty = ["\n" * 100000 + "1", "1." * 100000, "1 + " * 25000, "a.b " * 25000, "(" * 5000 + "1" + ")" * 5000,
                 "1" * 20000 + " + " + "2" * 20000, " " * 50000 + "1"]
        t0 = time.perf_counter()
        for text in nasty:
            self.r(text)
        self.assertLess(time.perf_counter() - t0, 3.0)           # was 30+ seconds for one of them before the fix

    def test_existing_behaviour_is_kept(self):
        self.assertEqual(assess("my project folder").level, "normal")
        self.assertEqual(assess("please debug this").level, "complex")
        self.assertEqual(assess("сделай план").level, "complex")
        self.assertEqual(assess("x" * 800).level, "complex")


class IntegrationTests(Base):
    def test_model_calls_the_calculator_and_uses_the_real_result(self):
        a, llm = self.make([
            json.dumps({"tool": "calc.evaluate", "args": {"expressions": "120+90; 1.5+2"}}),
            json.dumps({"say": "Total distance is 210 km in 3.5 hours: 60 km/h (calculated)."})])
        events = []
        reply = a.conversation().send("A train travels 120 km in 1.5 hours, then 90 km in 2 hours. "
                                      "What is its average speed?", events.append)
        self.assertEqual(reply, "Total distance is 210 km in 3.5 hours: 60 km/h (calculated).")
        second = llm.sent[1][-1]["content"]                       # what the model was told after the tool ran
        self.assertIn("120+90 = 210", second)
        self.assertIn("1.5+2 = 3.5", second)
        self.assertIn("calc.evaluate", llm.sent[0][0]["content"])  # the instructions + tool were offered
        self.assertIn("calculated", llm.sent[0][0]["content"])
        t = a.agent.last_turn
        self.assertEqual([c["tool"] for c in t.tool_calls], ["calc.evaluate"])
        self.assertTrue(t.tool_calls[0]["ok"])
        self.assertFalse(a.session.tainted)
        audit = (a.paths.logs / "audit.jsonl").read_text(encoding="utf-8")
        self.assertIn("calc.evaluate", audit)
        self.assertTrue(any("[tool] ok" in e for e in events))

    def test_expression_in_the_message_is_calculated_before_the_model_runs(self):
        a, llm = self.make(['{"say": "393"}'])
        a.conversation().send("What is 17*23+45/9-3?")
        user_msg = llm.sent[0][-1]["content"]
        self.assertIn("[Calculator results", user_msg)
        self.assertIn("17*23+45/9-3 = 393", user_msg)
        self.assertEqual(a.agent.last_turn.preeval, 1)

    def test_invalid_tool_requests_fail_safely(self):
        a, llm = self.make([
            json.dumps({"tool": "calc.evaluate", "args": {"expression": "__import__('os').system('echo hi')"}}),
            json.dumps({"tool": "calc.evaluate", "args": {"expression": "1/0"}}),
            json.dumps({"tool": "calc.evaluate", "args": {"nonsense": 1}}),
            json.dumps({"tool": "os.system", "args": {"cmd": "calc"}}),
            json.dumps({"say": "I could not calculate that."})])
        reply = a.conversation().send("calculate 5 + 5 for me")
        self.assertEqual(reply, "I could not calculate that.")
        calls = a.agent.last_turn.tool_calls
        self.assertEqual([c["ok"] for c in calls], [False, False, False, False])
        self.assertIn("unsupported function", llm.sent[1][-1]["content"])
        self.assertIn("division by zero", llm.sent[2][-1]["content"])
        self.assertIn("unknown argument", llm.sent[3][-1]["content"])
        self.assertIn("not available for this message", llm.sent[4][-1]["content"])

    def test_calculator_not_offered_in_ordinary_chat(self):
        a, llm = self.make(["Hello! Nice to meet you."])
        reply = a.conversation().send("hello, I'm 25 and my room is 12")
        self.assertEqual(reply, "Hello! Nice to meet you.")      # plain chat: raw model text, no JSON, no tools
        system = llm.sent[0][0]["content"]
        self.assertNotIn("calc", system)
        self.assertNotIn("TOOLS", system)
        self.assertLess(len(system), 600)

    def test_tool_use_is_bounded(self):
        a, llm = self.make([json.dumps({"tool": "calc.evaluate", "args": {"expression": "1+1"}})] * 20)
        a.settings.max_agent_steps = 3
        a.conversation().send("calculate 1+1 again and again")
        self.assertLessEqual(len(llm.calls), 4)                  # 3 steps + the final "answer now" request

    def test_diagnostics_footer_why_and_log(self):
        a, llm = self.make([json.dumps({"tool": "calc.evaluate", "args": {"expression": "2+2"}}), '{"say": "4"}'])
        a.conversation().send("calculate 2+2")
        t = a.agent.last_turn
        foot = t.footer({"eval_count": 20, "eval_duration": 2 * 10**9})
        for piece in ("Basic qwen3:8b", "auto", "10.0 tok/s", "calc:", "1/1 calls ok"):
            self.assertIn(piece, foot)
        text = "\n".join(t.lines())
        for piece in ("Basic", "why:", "complexity:", "calc.evaluate", "ok"):
            self.assertIn(piece, text)
        buf = io.StringIO()
        with redirect_stdout(buf):
            handle_command("/why", a, a.conversation(), print)
        self.assertIn("chip:       Basic (qwen3:8b), auto", buf.getvalue())
        rows = [json.loads(l) for l in (a.paths.logs / "routing.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["chip"], "standard")
        self.assertEqual(rows[-1]["tool_calls"][0]["tool"], "calc.evaluate")

    def test_manual_choice_is_shown_in_diagnostics(self):
        a, llm = self.make(['{"say": "ok"}'])
        a.chips.set_override("advanced")
        a.conversation().send("hello")
        t = a.agent.last_turn
        self.assertEqual((t.chip, t.mode), ("advanced", "manual"))
        self.assertIn("Advanced qwen3:14b · manual", t.footer())
        self.assertIn("you asked for Advanced", t.reason)

    def test_settings_files_from_older_versions_still_load(self):
        home = self.tmp / "old"
        (home / "config").mkdir(parents=True)
        (home / "config" / "settings.json").write_text(json.dumps({"model": "llama3.1:8b", "think": False}))
        a = Ksar(home, llm=Scripted([]))
        self.assertEqual(a.settings.standard_chip, "llama3.1:8b")
        self.assertIn("calc.evaluate", a.tools.tools)


if __name__ == "__main__":
    unittest.main()

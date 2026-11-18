"""
Router: decides, WITHOUT any model call, how a message is handled.

  * complexity   - a deterministic score from structural signals (not just keywords or length)
  * tools        - which tools are offered for this one message (none for ordinary chat)
  * memory       - whether to look something up in memory
  * pre-evaluated arithmetic found in the message itself (done by the calculator, instantly)

The result is a small structured Route, so later versions can improve routing (or switch chips in the middle of a
task) without touching the rest of the code. Complexity threshold: a score of 3 or more is "complex".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

COMPLEX_THRESHOLD = 3
MAX_ANALYSED_CHARS = 6000          # the router only reads the start of a very long message: bounded cost, no regex blow-ups

# ----------------------------------------------------------------------------- signal patterns
STRONG_RE = re.compile(
    r"\b(plan|planning|analy[sz]e|analysis|debug|architecture|refactor|design|compare|investigate|step[- ]by[- ]step"
    r"|optimi[sz]e|prove|derive|algorithm|trade-?offs?|time complexity|figure out why)\b"
    r"|план\w*|анализ\w*|проанализ\w*|отлад\w*|архитектур\w*|сравни\w*|оптимизир\w*|докажи\w*|алгоритм\w*", re.I)
MATH_VERB_RE = re.compile(
    r"\b(calculate|compute|evaluate|solve|determine|sum of|product of|percent(?:age)?|average|mean|median|probability"
    r"|square root|factorial|derivative|integral|equation|how many|how much|how long|how far|how fast|what is the "
    r"(?:sum|product|difference|total|value|result|area|volume|speed|rate))\b"
    r"|вычисли\w*|посчитай\w*|сколько|реши\w*|найди значение|процент\w*|вероятност\w*|уравнени\w*", re.I)
QUESTION_RE = re.compile(r"\?|\b(what is|how (?:many|much|long|far|fast)|find|determine|compute|calculate)\b|сколько|найди", re.I)
DATE_TIME_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[:.]\d{2}(?::\d{2})?\b|\b\d{1,3}(?:\.\d{1,3}){2,5}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
# an operator between two operands; the right operand is only looked at (lookahead) so chains like 1+2+3 count fully
OP_BETWEEN_RE = re.compile(r"(?<=[\d)])\s*(?:\*\*|//|[+*/^×÷%]|(?<=\))-)\s*(?=-?[\d(])|(?<=[\d)])\s+-\s+(?=[\d(])")
LOOSE_MINUS_RE = re.compile(r"(?<=\d)-(?=\d)")
MATH_CUE_RE = re.compile(r"\b(?:what is|what's|calculate|compute|evaluate|equals?|result of|value of)\b|=|сколько будет|посчитай|вычисли", re.I)
EQUATION_RE = re.compile(r"\b\d*\s*[a-z]\s*(?:[+\-*/^]\s*\d+\s*)?=\s*[-\d(]|\bsolve for\b|\bx\s*\^?\s*2\b", re.I)
FENCE_RE = re.compile(r"```")
CODE_LINE_RE = re.compile(r"(?m)^[ \t]*(?:def |class |import |from \S+ import |for .+:|while .+:|if .+:|return\b|#include|"
                          r"public |private |static |function |const |let |var |\}|\{)|;\s*$|=>|->|\bprint\(|\bconsole\.log")
IO_RE = re.compile(r"(?im)^[ \t]*(?:sample[ \t]+)?(?:input|output|example[ \t]*\d*)[ \t]*:?[ \t]*$|\binput\b[^.\n]{0,60}\boutput\b|\bexpected (?:output|result)\b")
CONSTRAINT_RE = re.compile(r"\b[a-z]\s*(?:≤|<=|<|>=|≥)\s*\d|\d\s*(?:≤|<=)\s*[a-z]|10\s*\^\s*\d|10\s*\*\*\s*\d|\b1e\d\b|\btest cases?\b|\btime limit\b|\bmemory limit\b", re.I)
CODING_TASK_RE = re.compile(r"\b(?:write|implement|create|code|make)\b[^.\n]{0,40}\b(?:function|program|script|class|algorithm|solution|code)\b|\bimplement\b|напиши\w* (?:функци|программ|код|скрипт)", re.I)
ALGO_RE = re.compile(r"\b(?:modulo|mod\s*\d|dynamic programming|recursion|recursive|binary search|permutations?|subsequence|subarray|prime|fibonacci|gcd|lcm|graph|matrix|big[- ]?o|O\([^)]*\))\b|\bO\(", re.I)
DEBUG_RE = re.compile(r"\b(?:error|exception|traceback|bug|crash(?:es|ed)?|doesn'?t work|not working|fails?|failing|wrong (?:output|answer|result)|incorrect|segfault|\w+Error)\b|ошибк\w*|не работает", re.I)
SEQUENCE_RE = re.compile(r"\b(?:first|then|after that|afterwards|next|finally|and then|before that|step \d+)\b|\bсначала\b|\bзатем\b|\bпотом\b|\bнаконец\b", re.I)
CONDITION_RE = re.compile(r"\b(?:must|at most|at least|exactly|only if|unless|given that|such that|provided that|no more than|constraint|requirement|except|both|neither)\b|\bдолжен\b|\bтолько если\b|\bне более\b|\bне менее\b", re.I)
NUMBERED_LINE_RE = re.compile(r"(?m)^[ \t]*(?:\d+[.)]|[-*•])[ \t]+\S")
MATH_WORD_RE = re.compile(
    r"\b(?:calculate|compute|evaluate|solve|sum of|product of|percent(?:age)?|average of|probability|square root|sqrt|"
    r"factorial|how much is)\b|\bwhat is \d|\d\s*%\s*of\b|вычисли\w*|посчитай\w*|сколько будет|реши\b|процент\w*", re.I)


@dataclass
class Complexity:
    score: int = 0
    signals: list = field(default_factory=list)      # human-readable reasons, strongest first

    @property
    def level(self) -> str:
        return "complex" if self.score >= COMPLEX_THRESHOLD else "normal"

    def why(self) -> str:
        return ", ".join(self.signals) if self.signals else "ordinary request"


def _strip(text: str) -> str:
    return DATE_TIME_RE.sub(" ", text)


def count_arithmetic(text: str) -> int:
    """Number of arithmetic operators between operands. '17-5' counts as subtraction only with a maths cue
    ('what is', 'calculate', '='), so '10-20 dollars' stays a range."""
    t = _strip(text)
    n = len(OP_BETWEEN_RE.findall(t))
    if MATH_CUE_RE.search(t):
        n += len(LOOSE_MINUS_RE.findall(t))
    return n


def assess(text: str) -> Complexity:
    """Deterministic complexity score from structure. Only the user's own message is ever examined."""
    text = text or ""
    n_chars, text = len(text), text[:MAX_ANALYSED_CHARS]
    c = Complexity()

    def add(points: int, why: str) -> None:
        c.score += points
        c.signals.append(f"{why} (+{points})")

    clean = _strip(text)
    if n_chars > 700:
        add(3, "long message")
    elif n_chars > 350:
        add(1, "fairly long message")
    if STRONG_RE.search(text):
        add(3, "analysis/design/debug wording")
    fences = len(FENCE_RE.findall(text)) // 2
    code_lines = len(CODE_LINE_RE.findall(text))
    has_code = fences > 0 or code_lines >= 3
    if has_code:
        add(2, "contains code")
    if has_code and DEBUG_RE.search(text):
        add(2, "debugging code")
    if IO_RE.search(text):
        add(2, "input/output examples")
    if CONSTRAINT_RE.search(text):
        add(2, "problem constraints")
    if CODING_TASK_RE.search(text):
        add(1, "coding task")
    if ALGO_RE.search(text):
        add(1, "algorithmic topic")
    ops = count_arithmetic(text)
    if ops >= 5:
        add(3, f"{ops} arithmetic operations")
    elif ops >= 3:
        add(2, f"{ops} arithmetic operations")
    elif ops >= 1:
        add(1, "arithmetic")
    if EQUATION_RE.search(clean):
        add(2, "equation")
    numbers = NUMBER_RE.findall(clean)
    if ops == 0 and len(numbers) >= 3 and QUESTION_RE.search(text) and MATH_VERB_RE.search(text):
        add(3 if len(numbers) >= 4 else 2, f"word problem with {len(numbers)} values")
    if len(NUMBERED_LINE_RE.findall(text)) >= 3 or len(SEQUENCE_RE.findall(text)) >= 3:
        add(2, "multi-step instructions")
    if len(CONDITION_RE.findall(text)) >= 3:
        add(2, "several conditions")
    c.signals.sort(key=lambda s: -int(re.search(r"\+(\d+)\)", s).group(1)))
    return c


def classify_complexity(text: str) -> str:
    """'complex' or 'normal' (kept for compatibility: tasks and tests use it)."""
    return assess(text).level


# ----------------------------------------------------------------------------- arithmetic found in the message
_EXPR_SPAN_RE = re.compile(r"[\d(][\d\s.+\-*/^%()×÷]*[\d)]")


def find_expressions(text: str, limit: int = 5) -> list:
    """Arithmetic written out in the message (e.g. '17*23+45/9'). Dates, times, versions, phone-like numbers and
    pieces of function calls are skipped. Returns expression strings the calculator can evaluate."""
    text = (text or "")[:MAX_ANALYSED_CHARS]
    masked = DATE_TIME_RE.sub(lambda m: " " * len(m.group()), text)
    out = []
    for m in _EXPR_SPAN_RE.finditer(masked):
        s = m.group().strip()
        start, end = m.start(), m.end()
        # the span must stand alone: not the tail of 'x^2 * 4' or 'sqrt(...)', not the head of '2(3)' or '4x'
        near_b = masked[start - 1] if start > 0 else " "
        near_a = masked[end] if end < len(masked) else " "
        prev_c = masked[:start].rstrip()[-1:] or " "
        next_c = masked[end:].lstrip()[:1] or " "
        if near_b.isalnum() or near_b in "_." or near_a.isalnum() or near_a in "_." \
                or prev_c in "^*/+-%" or next_c in "^*/+-%(":
            continue
        loose = bool(MATH_CUE_RE.search(text)) and LOOSE_MINUS_RE.search(s)
        if not re.search(r"\d", s) or not (re.search(r"(?:\*\*|//|[+*/^%×÷])|\s-\s|\)-|-\s*\(", s) or loose):
            continue                                   # needs a real operator ('10-20' alone is a range, not maths)
        if re.fullmatch(r"10\s*\^\s*\d+", s) or re.fullmatch(r"10\s*\*\*\s*\d+", s):
            continue                                   # '10^5' in a constraint is a size, not a question
        if s.count("(") != s.count(")"):
            continue
        if re.search(r"\d\s+\d", s):                   # '3 4' is two numbers, not an expression
            continue
        if re.search(r"\d,\d{3}", s):
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------------------- tool and memory routing
MEM_RE = re.compile(
    r"\b(remember|recall|remind\w*|memor(?:y|ies)|about me|did i|i told you|i said|last time|earlier|yesterday"
    r"|previous(?:ly)?|what do you know)\b|запомни\w*|помн\w+|вспомни\w*|памят\w+|что ты знаешь|я говорил\w*"
    r"|ранее|вчера|в прошлый раз", re.I)
FS_RE = re.compile(
    r"\b(files?|folders?|director(?:y|ies)|workspace|desktop|downloads?)\b|файл\w*|папк\w*|каталог\w*|рабоч\w+ стол"
    r"|[\w.-]{1,80}\.(?:txt|md|py|json|csv|docx?|xlsx?|pdf|png|jpe?g|zip)\b|[A-Za-z]:\\|~/|\./", re.I)
SYS_RE = re.compile(r"\b(what time|current time|time is it|disk space|free space|storage left|which os|operating system)\b"
                    r"|сколько времени|который час|место на диске", re.I)
MEMTOOL_RE = re.compile(r"\b(forget|search|check|look up|open|read|correct|link|attach)\b.{0,40}\b(memory|notes?\b(?!\.\w))"
                        r"|\bforget (that|about)\b|забудь|найди в памяти|исправь в памяти", re.I)
FAMILIES = {"fs": "fs.", "memory": "memory.", "sys": "sys.", "calc": "calc."}


@dataclass
class Route:
    tools: list = field(default_factory=list)       # tool names offered for this message (empty = plain chat)
    memory: bool = False                            # inject relevant memory text into this message
    complexity: Complexity = field(default_factory=Complexity)
    preeval: list = field(default_factory=list)     # [(expression, result text)] calculated before the model runs
    calc_reason: str = ""                           # why the calculator was offered ('' = not offered)

    def label(self) -> str:
        bits = []
        if self.memory:
            bits.append("memory")
        if self.tools:
            bits.append("tools: " + ", ".join(self.tools))
        return "; ".join(bits)


def route_message(text: str, tool_names: list) -> Route:
    """Cheap, instant routing (no model call). A tool is offered when the user names it (fs.read_text, @fs) or asks
    for something only it can do; otherwise Ksar just talks."""
    text = (text or "")[:MAX_ANALYSED_CHARS * 2]
    low = text.lower()
    names: list = []

    def add(n):
        if n in tool_names and n not in names:
            names.append(n)

    for n in tool_names:                                       # 1. explicit: the tool's own name
        if re.search(r"(?<![\w.])" + re.escape(n) + r"(?![\w])", low):
            add(n)
    for fam, prefix in FAMILIES.items():                       # 2. explicit family: @fs, @memory, @sys, @calc
        if re.search(r"(?<!\w)@" + fam + r"\b", low):
            for n in tool_names:
                if n.startswith(prefix):
                    add(n)
    if FS_RE.search(text):                                     # 3. natural requests
        for n in tool_names:
            if n.startswith("fs."):
                add(n)
    if SYS_RE.search(text):
        add("sys.info")
    if MEMTOOL_RE.search(text):
        for n in tool_names:
            if n.startswith("memory."):
                add(n)
    if "fs.write_text" in names:
        add("fs.read_text")                                    # editing needs a read first

    cx = assess(text)
    calc_reason = ""
    if "calc.evaluate" in names:
        calc_reason = "named by you"
    elif "calc.evaluate" in tool_names:
        nums = NUMBER_RE.findall(_strip(text))
        ops = count_arithmetic(text)
        if ops >= 1:
            calc_reason = "arithmetic in the message"
        elif MATH_WORD_RE.search(text) and nums:
            calc_reason = "calculation requested"
        elif any(s.startswith(("input/output", "problem constraints")) for s in cx.signals) and len(nums) >= 2:
            calc_reason = "numerical programming problem"
        elif any("values" in s for s in cx.signals):
            calc_reason = "several values to combine"
        if calc_reason:
            add("calc.evaluate")

    pre = []
    if "calc.evaluate" in names:
        from .tools.calc import CalcError, evaluate, fmt     # local import: router stays importable on its own
        for e in find_expressions(text):
            try:
                pre.append((e, fmt(evaluate(e))))
            except CalcError:
                continue
    return Route(tools=names, memory=bool(MEM_RE.search(text)), complexity=cx, preeval=pre, calc_reason=calc_reason)

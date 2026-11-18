"""
calc.evaluate - a deterministic calculator the model can call instead of doing arithmetic "in its head".

Safety: the expression is parsed with ast.parse(mode="eval") and walked by a WHITELIST evaluator. There is no eval()
or exec(), no attribute access, no names except a few constants, no calls except the listed functions, no strings,
lists, comprehensions or lambdas. Sizes are bounded (expression length, tree size, depth, result size, factorial /
power arguments, wall-clock time), so a hostile or careless expression cannot hang or crash Ksar.
The tool is pure computation: no file, network or system access, so it needs no approval (Risk.READ).

Numbers: integers are exact (Python big ints, up to ~4000 digits); int / int gives an exact fraction (shown as a
fraction and a decimal); anything involving a float or a function like sqrt() is a float (15 significant digits).
"""
from __future__ import annotations

import ast
import math
import re
import time
from fractions import Fraction

from .base import Param, Risk, Tool, ToolResult

MAX_EXPR_CHARS = 500
MAX_NODES = 250
MAX_DEPTH = 40
MAX_BITS = 14_000                 # about 4,200 decimal digits: exact, printable, and fast
MAX_POW_EXPONENT = 10_000
MAX_FACTORIAL = 2_000
MAX_COMB = 5_000
MAX_BATCH = 10
TIME_BUDGET_S = 2.0


class CalcError(Exception):
    """A clear, user-facing reason why an expression was refused or failed."""


# ----------------------------------------------------------------------------- numbers
def _bits(x) -> int:
    if isinstance(x, bool):
        return 1
    if isinstance(x, int):
        return abs(x).bit_length()
    if isinstance(x, Fraction):
        return max(abs(x.numerator).bit_length(), x.denominator.bit_length())
    return 0


def _check(x):
    """Every intermediate result must be a real, finite, reasonably sized number."""
    if isinstance(x, bool) or isinstance(x, complex):
        raise CalcError("the result is not a real number")
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            raise CalcError("the result is not finite (overflow or undefined)")
        return x
    if _bits(x) > MAX_BITS:
        raise CalcError(f"the result is too large (limit about {MAX_BITS * 30103 // 100000} digits)")
    if isinstance(x, Fraction) and x.denominator == 1:
        return int(x.numerator)
    return x


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float, Fraction)):
        raise CalcError("only plain numbers are allowed")
    return x


def _is_exact(x) -> bool:
    return isinstance(x, (int, Fraction)) and not isinstance(x, bool)


def _to_float(x) -> float:
    try:
        return float(x)
    except OverflowError:
        raise CalcError("the number is too large for this function") from None


def _int_arg(x, name: str, lo: int = None, hi: int = None) -> int:
    if isinstance(x, Fraction) and x.denominator == 1:
        x = int(x)
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if not isinstance(x, int) or isinstance(x, bool):
        raise CalcError(f"{name} needs a whole number")
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise CalcError(f"{name} must be between {lo} and {hi}" if None not in (lo, hi) else f"{name} is out of range")
    return x


# ----------------------------------------------------------------------------- operators
def _pow(a, b):
    if _is_exact(a) and _is_exact(b):
        if isinstance(b, Fraction) and b.denominator != 1:
            if a < 0:
                raise CalcError("a negative number to a fractional power is not a real number")
            return _check(_to_float(a) ** _to_float(b))
        e = int(b)
        if a == 0 and e < 0:
            raise CalcError("division by zero (0 to a negative power)")
        if abs(e) > MAX_POW_EXPONENT:
            raise CalcError(f"exponent too large (limit {MAX_POW_EXPONENT})")
        if a not in (0, 1, -1) and _bits(a) * abs(e) > MAX_BITS:
            raise CalcError(f"the result would be too large (limit about {MAX_BITS * 30103 // 100000} digits)")
        return _check(Fraction(a) ** e if e < 0 or isinstance(a, Fraction) else a ** e)
    try:
        r = _to_float(a) ** _to_float(b)
    except ZeroDivisionError:
        raise CalcError("division by zero") from None
    except OverflowError:
        raise CalcError("the result is too large (overflow)") from None
    return _check(r)


def _binop(op, a, b):
    a, b = _num(a), _num(b)
    try:
        if isinstance(op, ast.Add):
            return _check(a + b)
        if isinstance(op, ast.Sub):
            return _check(a - b)
        if isinstance(op, ast.Mult):
            return _check(a * b)
        if isinstance(op, ast.Div):
            if b == 0:
                raise CalcError("division by zero")
            return _check(Fraction(a, 1) / Fraction(b, 1) if _is_exact(a) and _is_exact(b) else a / b)
        if isinstance(op, ast.FloorDiv):
            if b == 0:
                raise CalcError("division by zero")
            return _check(a // b)
        if isinstance(op, ast.Mod):
            if b == 0:
                raise CalcError("modulo by zero")
            return _check(a % b)
        if isinstance(op, ast.Pow):
            return _pow(a, b)
    except OverflowError:
        raise CalcError("the result is too large (overflow)") from None
    except ZeroDivisionError:
        raise CalcError("division by zero") from None
    raise CalcError("unsupported operator")


# ----------------------------------------------------------------------------- functions and constants
def _f1(fn):
    def wrap(x):
        try:
            return _check(fn(_to_float(x)))
        except ValueError:
            raise CalcError("the argument is outside the function's domain") from None
        except OverflowError:
            raise CalcError("the result is too large (overflow)") from None
    return wrap


def _round(x, nd=None):
    if nd is None:
        return _check(round(x))
    return _check(round(x, _int_arg(nd, "round digits", -15, 15)))


def _log(x, base=None):
    try:
        if base is None:
            return _check(math.log(_to_float(x)))
        return _check(math.log(_to_float(x), _to_float(base)))
    except (ValueError, ZeroDivisionError):
        raise CalcError("the argument is outside the function's domain") from None


def _sqrt(x):
    if _is_exact(x) and x >= 0:                         # exact when the root is rational
        f = Fraction(x)
        n, d = math.isqrt(f.numerator), math.isqrt(f.denominator)
        if n * n == f.numerator and d * d == f.denominator:
            return _check(Fraction(n, d))
    return _f1(math.sqrt)(x)


def _cbrt(x):
    f = _to_float(x)
    return _check(math.copysign(abs(f) ** (1 / 3), f))


def _factorial(n):
    return _check(math.factorial(_int_arg(n, "factorial", 0, MAX_FACTORIAL)))


def _comb(n, k):
    return _check(math.comb(_int_arg(n, "comb n", 0, MAX_COMB), _int_arg(k, "comb k", 0, MAX_COMB)))


def _perm(n, k=None):
    n = _int_arg(n, "perm n", 0, MAX_COMB)
    return _check(math.perm(n) if k is None else math.perm(n, _int_arg(k, "perm k", 0, MAX_COMB)))


def _gcd(*a):
    return _check(math.gcd(*[_int_arg(x, "gcd") for x in a]))


def _lcm(*a):
    return _check(math.lcm(*[_int_arg(x, "lcm") for x in a]))


FUNCS = {
    "abs": (lambda x: _check(abs(x)), 1, 1), "fabs": (lambda x: _check(abs(x)), 1, 1),
    "sqrt": (_sqrt, 1, 1), "cbrt": (_cbrt, 1, 1),
    "exp": (_f1(math.exp), 1, 1), "log": (_log, 1, 2), "ln": (_log, 1, 1),
    "log10": (_f1(math.log10), 1, 1), "log2": (_f1(math.log2), 1, 1),
    "sin": (_f1(math.sin), 1, 1), "cos": (_f1(math.cos), 1, 1), "tan": (_f1(math.tan), 1, 1),
    "asin": (_f1(math.asin), 1, 1), "acos": (_f1(math.acos), 1, 1), "atan": (_f1(math.atan), 1, 1),
    "sinh": (_f1(math.sinh), 1, 1), "cosh": (_f1(math.cosh), 1, 1), "tanh": (_f1(math.tanh), 1, 1),
    "degrees": (_f1(math.degrees), 1, 1), "radians": (_f1(math.radians), 1, 1),
    "floor": (lambda x: _check(math.floor(x)), 1, 1), "ceil": (lambda x: _check(math.ceil(x)), 1, 1),
    "trunc": (lambda x: _check(math.trunc(x)), 1, 1), "int": (lambda x: _check(math.trunc(x)), 1, 1),
    "round": (_round, 1, 2),
    "min": (lambda *a: _check(min(a)), 1, 8), "max": (lambda *a: _check(max(a)), 1, 8),
    "hypot": (lambda *a: _check(math.hypot(*[_to_float(x) for x in a])), 2, 4),
    "atan2": (lambda y, x: _check(math.atan2(_to_float(y), _to_float(x))), 2, 2),
    "factorial": (_factorial, 1, 1), "comb": (_comb, 2, 2), "perm": (_perm, 1, 2),
    "gcd": (_gcd, 2, 8), "lcm": (_lcm, 2, 8),
}
CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


# ----------------------------------------------------------------------------- evaluator
class _Eval:
    def __init__(self):
        self.nodes = 0
        self.deadline = time.monotonic() + TIME_BUDGET_S

    def run(self, node, depth=0):
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise CalcError("the expression is too complicated")
        if depth > MAX_DEPTH:
            raise CalcError("the expression is nested too deeply")
        if time.monotonic() > self.deadline:
            raise CalcError("the calculation took too long")
        t = type(node)
        if t is ast.Expression:
            return self.run(node.body, depth + 1)
        if t is ast.Constant:
            v = node.value
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise CalcError("only numbers are allowed (no text, booleans or complex numbers)")
            return _check(v)
        if t is ast.UnaryOp and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = _num(self.run(node.operand, depth + 1))
            return _check(-v if isinstance(node.op, ast.USub) else v)
        if t is ast.BinOp and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            return _binop(node.op, self.run(node.left, depth + 1), self.run(node.right, depth + 1))
        if t is ast.Name:
            if node.id in CONSTS:
                return CONSTS[node.id]
            raise CalcError(f"unknown name '{node.id}' (constants: pi, e, tau)")
        if t is ast.Call:
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                raise CalcError("unsupported function (allowed: " + ", ".join(sorted(FUNCS)) + ")")
            if node.keywords:
                raise CalcError("named arguments are not supported")
            fn, lo, hi = FUNCS[node.func.id]
            if not lo <= len(node.args) <= hi:
                raise CalcError(f"{node.func.id}() takes {lo}" + (f" to {hi}" if hi != lo else "") + " argument(s)")
            args = [_num(self.run(a, depth + 1)) for a in node.args]
            try:
                return fn(*args)
            except CalcError:
                raise
            except (ValueError, TypeError, ZeroDivisionError, OverflowError):
                raise CalcError(f"{node.func.id}() cannot compute this") from None
        raise CalcError(f"unsupported syntax ({t.__name__}); only arithmetic and the listed functions are allowed")


_NORMALIZE = str.maketrans({"×": "*", "÷": "/", "−": "-", "–": "-", "—": "-", "^": "**"})


def normalize(expr: str) -> str:
    """Friendly input: ^ means power (not XOR), × ÷ − are accepted, 1_000 is allowed, '2x' is NOT guessed."""
    return expr.strip().translate(_NORMALIZE).replace("****", "**")


def evaluate(expr: str):
    """Returns an int, Fraction or float. Raises CalcError with a clear reason for anything else."""
    if not isinstance(expr, str) or not expr.strip():
        raise CalcError("empty expression")
    if len(expr) > MAX_EXPR_CHARS:
        raise CalcError(f"expression too long (limit {MAX_EXPR_CHARS} characters)")
    src = normalize(expr)
    if "**" in src and re.search(r"\*\*\s*\*", src):
        raise CalcError("invalid syntax")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        raise CalcError("invalid syntax") from None
    except (ValueError, RecursionError, MemoryError):
        raise CalcError("the expression cannot be read") from None
    try:
        return _Eval().run(tree)
    except RecursionError:
        raise CalcError("the expression is nested too deeply") from None


def fmt(x) -> str:
    """Exact text for ints, 'n/d (= decimal)' for fractions, 15 significant digits for floats."""
    if isinstance(x, int):
        return str(x)
    if isinstance(x, Fraction):
        return f"{x.numerator}/{x.denominator} (= {float(x):.15g})"
    return f"{x:.15g}"


# ----------------------------------------------------------------------------- the tool
def calc_evaluate(ctx, a) -> ToolResult:
    items = []
    if a.get("expression"):
        items.append(a["expression"])
    if a.get("expressions"):
        items += [p.strip() for p in re.split(r"[;\n]+", a["expressions"]) if p.strip()]
    if not items:
        return ToolResult(False, "give an expression, e.g. 12*(3+4)/5")
    if len(items) > MAX_BATCH:
        return ToolResult(False, f"at most {MAX_BATCH} expressions per call")
    lines, ok_n, errs = [], 0, 0
    for it in items:
        try:
            lines.append(f"{it} = {fmt(evaluate(it))}")
            ok_n += 1
        except CalcError as e:
            lines.append(f"{it} -> ERROR: {e}")
            errs += 1
    ok = errs == 0
    msg = lines[0] if len(items) == 1 else f"{ok_n} of {len(items)} calculated"
    return ToolResult(ok, msg, "\n".join(lines) + "\n(calculated by calc.evaluate)")


def tools() -> list:
    P = Param
    return [
        Tool("calc.evaluate",
             "Exact calculator. Give ONE expression, or several separated by ; or new lines. Supports + - * / // % ** ^ "
             "( ), decimals, pi e tau, and sqrt cbrt abs round floor ceil exp ln log log2 log10 sin cos tan asin acos "
             "atan atan2 sinh cosh tanh degrees radians min max hypot factorial comb perm gcd lcm. "
             "Use it for every non-trivial calculation; never guess arithmetic.",
             {"expression": P(desc="e.g. 17*23+45/9", required=False),
              "expressions": P(desc="several expressions separated by ; or new lines", required=False)},
             Risk.READ, calc_evaluate),
    ]

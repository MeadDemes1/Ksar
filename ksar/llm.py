"""
LLM interface: talks to the Ollama HTTP API, extracts JSON from model output, and gives the Chip Manager
real lifecycle control (load / unload / what is running / what is installed).
"""
from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request


class LLMError(Exception):
    """Raised when the LLM API request fails or returns an unusable response."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Removes Qwen3-style <think>...</think> blocks (also an unterminated one at the start)."""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    if text.lstrip().lower().startswith("<think>"):      # cut off before </think>
        return ""
    return text.strip()


def _balanced_objects(text: str):
    """Yields every top-level {...} substring, respecting strings/escapes (so braces inside text are safe)."""
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def extract_json(text: str):
    """Best-effort: the first valid JSON object in the model's output (handles ``` fences and stray text)."""
    if not text or not isinstance(text, str):
        return None
    text = strip_think(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for cand in _balanced_objects(text):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


class ThinkSplitter:
    """Splits a token stream into ('thinking', ...) and ('content', ...) when the model inlines <think> tags
    (older Ollama). Safe across chunk boundaries."""
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf, self.in_think = "", False

    @staticmethod
    def _partial_tail(buf: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(buf)), 0, -1):
            if tag.startswith(buf[-n:]):
                return n
        return 0

    def feed(self, text: str) -> list:
        self.buf += text
        out = []
        while True:
            tag = self.CLOSE if self.in_think else self.OPEN
            kind = "thinking" if self.in_think else "content"
            i = self.buf.find(tag)
            if i >= 0:
                if i:
                    out.append((kind, self.buf[:i]))
                self.buf = self.buf[i + len(tag):]
                self.in_think = not self.in_think
                continue
            keep = self._partial_tail(self.buf, tag)
            emit = self.buf[:len(self.buf) - keep]
            if emit:
                out.append((kind, emit))
            self.buf = self.buf[len(self.buf) - keep:]
            return out

    def flush(self) -> list:
        out = [("thinking" if self.in_think else "content", self.buf)] if self.buf else []
        self.buf = ""
        return out


_SAY_KEY = re.compile(r'"(?:say|finish)"\s*:\s*"')
_ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", '"': '"', "\\": "\\", "/": "/"}


def _decode_json_string_prefix(s: str) -> str:
    """Decodes the beginning of a JSON string literal (up to its closing quote) from a possibly cut-off stream."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == '"':
            break
        if c == "\\":
            if i + 1 >= len(s):
                break
            n = s[i + 1]
            if n == "u":
                if i + 6 > len(s):
                    break
                try:
                    out.append(chr(int(s[i + 2:i + 6], 16)))
                except ValueError:
                    pass
                i += 6
                continue
            out.append(_ESC.get(n, n))
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


class SayExtractor:
    """Feeds on the raw streamed JSON {"say": "..."} and returns the answer text as soon as it is readable,
    so the answer can appear live even though the protocol is JSON."""

    def __init__(self):
        self.buf, self.emitted = "", 0

    def feed(self, text: str) -> str:
        self.buf += text
        m = _SAY_KEY.search(self.buf)
        if not m:
            return ""
        decoded = _decode_json_string_prefix(self.buf[m.end():])
        new, self.emitted = decoded[self.emitted:], len(decoded)
        return new


class LLM:
    """Ollama chat wrapper. Accepts a Settings object (preferred) or explicit parameters."""

    supports_think = True       # chat()/stream_chat() accept think=...; the agent checks this flag

    def __init__(self, base_url_or_settings=None, model: str = None, temperature: float = 0.7, timeout: int = 120):
        s = base_url_or_settings
        self.think = False
        self.num_ctx = 0
        if s is not None and not isinstance(s, str) and hasattr(s, "model"):
            # Settings fields are ollama_url / request_timeout / standard_chip (old names kept as fallbacks).
            self.base_url = (getattr(s, "ollama_url", None) or getattr(s, "base_url", None)
                             or "http://localhost:11434").rstrip("/")
            self.model = getattr(s, "standard_chip", "") or s.model or "qwen3:8b"
            self.temperature = getattr(s, "temperature", temperature)
            self.timeout = getattr(s, "request_timeout", None) or getattr(s, "timeout", None) or timeout
            self.num_ctx = int(getattr(s, "num_ctx", 0) or 0)
            self.think = bool(getattr(s, "think", False))
        else:
            self.base_url = (s if isinstance(s, str) else "http://localhost:11434").rstrip("/")
            self.model = model or "qwen3:8b"
            self.temperature = temperature
            self.timeout = timeout

    # ------------------------------------------------------------------ transport
    def _map_http_error(self, e: urllib.error.HTTPError, model: str) -> LLMError:
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", "")
        except Exception:
            detail = ""
        if e.code == 404:
            return LLMError(f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}")
        return LLMError(f"Ollama error {e.code}: {detail or e.reason}")

    def _request(self, path: str, payload: dict = None, timeout: int = None, method: str = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                     method=method or ("POST" if data is not None else "GET"))
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                body = r.read().decode("utf-8")
                return json.loads(body) if body.strip() else {}
        except urllib.error.HTTPError as e:           # must come BEFORE URLError (it is a subclass)
            raise self._map_http_error(e, (payload or {}).get("model", "?")) from e
        except urllib.error.URLError as e:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}. Is it running? ({e.reason})") from e
        except (TimeoutError, socket.timeout) as e:
            raise LLMError(f"Ollama did not answer within {timeout or self.timeout}s "
                           "(model too slow or still loading; raise request_timeout)") from e
        except json.JSONDecodeError as e:
            raise LLMError(f"Ollama returned invalid JSON: {e}") from e
        except OSError as e:
            raise LLMError(f"Connection to Ollama failed: {e}") from e

    def _chat_payload(self, messages, model, keep_alive, json_mode, temperature, think, stream, max_tokens=None) -> dict:
        options = {"temperature": self.temperature if temperature is None else temperature}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx           # without this Ollama silently uses its small default
        if max_tokens:
            options["num_predict"] = int(max_tokens)    # hard cap (used for short background jobs)
        payload = {"model": model or self.model, "messages": messages, "stream": stream, "options": options}
        want_think = self.think if think is None else think
        if want_think:
            payload["think"] = True
        else:
            payload["think"] = False                    # Qwen3: skip the slow hidden reasoning
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if json_mode:
            payload["format"] = "json"
        return payload

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list, model: str = None, keep_alive=None,
             json_mode: bool = False, temperature: float = None, think: bool = None, max_tokens: int = None) -> str:
        """
        keep_alive: -1 = stay loaded, 0 = unload right after, "30m" = unload after idle, None = Ollama default.
        json_mode:  forces valid JSON output.
        think:      None = use the setting; True/False overrides it for this call.
        """
        payload = self._chat_payload(messages, model, keep_alive, json_mode, temperature, think, False, max_tokens)
        try:
            data = self._request("/api/chat", payload)
        except LLMError as e:
            if payload.get("think") and "think" in str(e).lower():      # model has no thinking mode
                payload["think"] = False
                data = self._request("/api/chat", payload)
            else:
                raise
        if data.get("error"):
            raise LLMError(str(data["error"]))
        return strip_think(data.get("message", {}).get("content", "") or "")

    def stream_chat(self, messages: list, model: str = None, keep_alive=None, json_mode: bool = False,
                    temperature: float = None, think: bool = None, max_tokens: int = None):
        """
        Generator of (kind, text):  ("thinking", ...) | ("content", ...) | ("stats", {...}) at the end.
        Tokens arrive as the model produces them, so the user sees progress instead of a silent minute.
        """
        payload = self._chat_payload(messages, model, keep_alive, json_mode, temperature, think, True, max_tokens)
        for attempt in (1, 2):
            req = urllib.request.Request(f"{self.base_url}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                resp = urllib.request.urlopen(req, timeout=self.timeout)
                break
            except urllib.error.HTTPError as e:
                err = self._map_http_error(e, payload["model"])
                if attempt == 1 and payload.get("think") and "think" in str(err).lower():
                    payload["think"] = False                 # model has no thinking mode: retry plain
                    continue
                raise err from e
            except urllib.error.URLError as e:
                raise LLMError(f"Cannot reach Ollama at {self.base_url}. Is it running? ({e.reason})") from e
            except (TimeoutError, socket.timeout) as e:
                raise LLMError("Ollama did not start answering in time (model loading?). "
                               "Raise request_timeout or try again.") from e
            except OSError as e:
                raise LLMError(f"Connection to Ollama failed: {e}") from e
        splitter = ThinkSplitter()
        try:
            for raw in resp:
                line = raw.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("error"):
                    raise LLMError(str(d["error"]))
                m = d.get("message") or {}
                if m.get("thinking"):
                    yield "thinking", m["thinking"]
                if m.get("content"):
                    yield from splitter.feed(m["content"])
                if d.get("done"):
                    yield from splitter.flush()
                    stats = {k: d.get(k) for k in ("eval_count", "eval_duration", "prompt_eval_count",
                                                   "prompt_eval_duration", "load_duration", "total_duration")}
                    yield "stats", stats
                    return
            yield from splitter.flush()
        except (TimeoutError, socket.timeout) as e:
            raise LLMError("Ollama stopped answering mid-reply (timeout)") from e
        except OSError as e:
            raise LLMError(f"Connection to Ollama lost: {e}") from e
        finally:
            resp.close()

    # ------------------------------------------------------------------ chip lifecycle
    def load(self, model: str, keep_alive=-1) -> None:
        """Loads a model into memory without generating anything (an empty /api/generate request)."""
        self._request("/api/generate", {"model": model, "prompt": "", "keep_alive": keep_alive, "stream": False},
                      timeout=max(self.timeout, 300))

    def unload(self, model: str) -> None:
        self._request("/api/generate", {"model": model, "prompt": "", "keep_alive": 0, "stream": False}, timeout=60)

    def running_models(self) -> dict:
        """{model name: size in bytes} for models currently loaded in Ollama."""
        data = self._request("/api/ps", timeout=10)
        return {m.get("name") or m.get("model"): int(m.get("size", 0)) for m in data.get("models", [])}

    def installed_models(self) -> dict:
        """{model name: size in bytes} for models that have been pulled."""
        data = self._request("/api/tags", timeout=10)
        return {m.get("name") or m.get("model"): int(m.get("size", 0)) for m in data.get("models", [])}


def no_think(llm) -> dict:
    """kwargs that switch reasoning off for background jobs (memory updates), if this LLM supports it."""
    return {"think": False} if getattr(llm, "supports_think", False) else {}


# Backward compatibility alias
OllamaLLM = LLM

"""
LLM interface: talks to the Ollama HTTP API, extracts JSON from model output, and gives the Chip Manager
real lifecycle control (load / unload / what is running / what is installed).
"""
from __future__ import annotations

import json
import re
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


class LLM:
    """Ollama chat wrapper. Accepts a Settings object (preferred) or explicit parameters."""

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
            try:
                detail = json.loads(e.read().decode("utf-8")).get("error", "")
            except Exception:
                detail = ""
            model = (payload or {}).get("model", "?")
            if e.code == 404:
                raise LLMError(f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}") from e
            raise LLMError(f"Ollama error {e.code}: {detail or e.reason}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}. Is it running? ({e.reason})") from e
        except TimeoutError as e:
            raise LLMError(f"Ollama did not answer within {timeout or self.timeout}s "
                           "(model too slow or still loading; raise request_timeout)") from e
        except json.JSONDecodeError as e:
            raise LLMError(f"Ollama returned invalid JSON: {e}") from e
        except OSError as e:
            raise LLMError(f"Connection to Ollama failed: {e}") from e

    # ------------------------------------------------------------------ streaming transport
    def _stream_chat(self, payload: dict, on_token) -> dict:
        """Read Ollama's newline-delimited chat stream and forward content chunks."""
        url = f"{self.base_url}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        pieces = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                for line in response:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as e:
                        raise LLMError(f"Ollama returned an invalid streaming chunk: {e}") from e
                    if item.get("error"):
                        raise LLMError(str(item["error"]))
                    content = (item.get("message") or {}).get("content") or ""
                    if content:
                        pieces.append(content)
                        on_token(content)
                    if item.get("done"):
                        break
            return {"message": {"content": "".join(pieces)}}
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("error", "")
            except Exception:
                detail = ""
            model = payload.get("model", "?")
            if e.code == 404:
                raise LLMError(f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}") from e
            raise LLMError(f"Ollama error {e.code}: {detail or e.reason}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}. Is it running? ({e.reason})") from e
        except TimeoutError as e:
            raise LLMError(f"Ollama did not answer within {self.timeout}s "
                           "(model too slow or still loading; raise request_timeout)") from e
        except OSError as e:
            raise LLMError(f"Connection to Ollama failed: {e}") from e

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list, model: str = None, keep_alive=None,
             json_mode: bool = False, temperature: float = None, on_token=None) -> str:
        """
        keep_alive: -1 = stay loaded, 0 = unload right after, "30m" = unload after idle, None = Ollama default.
        json_mode:  forces valid JSON output.
        """
        options = {"temperature": self.temperature if temperature is None else temperature}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx           # without this Ollama silently uses its small default
        payload = {"model": model or self.model, "messages": messages,
                   "stream": on_token is not None, "options": options}
        if not self.think:
            payload["think"] = False                    # Qwen3: skip hidden reasoning unless explicitly enabled
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if json_mode:
            payload["format"] = "json"
        data = self._stream_chat(payload, on_token) if on_token is not None else self._request("/api/chat", payload)
        if data.get("error"):
            raise LLMError(str(data["error"]))
        return strip_think(data.get("message", {}).get("content", "") or "")

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


# Backward compatibility alias
OllamaLLM = LLM

# Changelog

## 0.2.1 - bugfix and optimisation release

### Bugs fixed
- **Ollama URL and timeout were ignored.** `LLM` read `settings.base_url` / `settings.timeout`, but the real
  fields are `ollama_url` / `request_timeout`, so the timeout was 120 s instead of 900 s and a custom URL did nothing.
- **`num_ctx` was never sent to Ollama**, so the 8192-token window in settings was not used and memory context
  could be silently cut off. It is now sent with every request.
- **Advanced chip loaded on a wrong RAM estimate.** Every chip was assumed to need 4 GB; `qwen3:32b` needs about
  21 GB. The estimate now comes from the real model size reported by Ollama (or from the "32b" in the name).
- **RAM detection used `wmic`**, which is removed from newer Windows 11. It failed silently, so free RAM read as 0
  and the advanced chip could never load. Now uses `GlobalMemoryStatusEx` (ctypes). Unknown RAM is reported, not hidden.
- **Almost every message escalated to the advanced chip.** Routing looked at the last 3 messages including tool
  output and file contents, and the word "project" counted as complex. Now only what *you* wrote counts, with word
  matching, and the chip is chosen once per turn/task (no model switching between steps).
- **A failed advanced-chip load was retried at every step** (each retry can wait minutes). Failures now back off 5 min.
- **Retention settings did nothing for normal chat** (no `keep_alive` was sent). They now map to Ollama keep-alive:
  persistent = stay loaded, balanced = 30 min (advanced 10 min), aggressive = 5 min (advanced unloads at once).
- **Loading a chip generated text** ("ping" chat). It now uses an empty `/api/generate` request (no tokens).
- **`settings.model` and `standard_chip` could disagree.** Old `settings.json` files keep their model; new keys are
  written back to the file.
- `HTTPError` handler was unreachable (it is a subclass of `URLError`); 404 now says `ollama pull <model>`.
- `extract_json` used a greedy `{.*}` regex; now picks the first balanced object and ignores braces inside strings.
- Qwen3 `<think>` blocks are stripped from replies.

### Optimisations
- Qwen3 thinking is off by default (`/config set llm.think true` to enable): much faster on CPU.
- `permissions.json` is only re-read when it changed (it was rewritten on every tool call).
- `/chip status` reads what Ollama really has loaded (`/api/ps`) instead of a remembered guess.

### New
- `/chip` (alias of `/chips`) and `/chippref [speed|balanced|quality]`.
  `speed` never uses the advanced chip, `quality` always does when RAM allows.
- `/config set llm.think|llm.num_ctx`.
- `.gitignore`: `ksar_data/` and `memory/` hold your private conversations and are no longer meant to be committed.
- Tests for all of the above (`tests/test_chips_llm.py`, 17 new tests; suite is 39 tests).

### Needs
Python 3.10+, Ollama 0.9 or newer for the `think` switch (older versions ignore it), `ollama pull qwen3:8b`.

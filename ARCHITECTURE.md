# Ksar 0.3.2 - architecture

Local-first assistant. Python 3.10+, standard library only. The model runs in Ollama on the same computer.
Full specification: see `docs/` (per-version notes) and the 0.3.1 specification text; this file is the short map.

## Layers
```
console (cli.py)  --Enter-->  Conversation.send  -->  Agent.turn
                                  |                      |
                                  |        router.route_message(text)   (no model call)
                                  |           -> Route: tools offered, memory yes/no, complexity, pre-calculated sums
                                  |        ChipManager.choose(...)      (manual /use override > automatic)
                                  |           -> ChipChoice: chip, mode, reason, note, error
                                  |        LLM.stream_chat / chat       (Ollama /api/chat, streamed)
                                  |        ToolManager.call             (validate -> risk -> permissions -> approval ->
                                  |                                      run -> verify -> audit.jsonl)
                                  v
                       journal + memory (Keeper) + routing.jsonl
IdleWorker (thread): warm model, save memory on the light chip, split notes, daily check. Stops when you press Enter.
```

## Modules
| module | job |
|---|---|
| `app.py` | `Ksar`: wires settings, memory, tools, chips, agent, idle worker; `app.lock` (RLock) |
| `cli.py` | console loop, commands, live printer (thoughts grey, answer streamed, footer line) |
| `router.py` | complexity score, tool/memory routing, calculator offer, pre-calculation |
| `agent.py` | prompts, tool loop (bounded), `Conversation` (history window, journal, memory saves), `TurnInfo`, tasks |
| `chips.py` | three slots (standard=Basic, advanced=Advanced, light=Idle/Light), RAM-aware loading, `/use` override |
| `llm.py` | Ollama client: streaming, thinking split, JSON extraction, load/unload/ps/tags |
| `resources.py` | free RAM (ctypes / `/proc`), model size estimates |
| `idle.py` | background worker and its jobs |
| `tasks.py` | persistent task state machine |
| `memory/` | Markdown notes, journal, BM25+graph retrieval, Keeper (model proposes, code applies), maintenance |
| `tools/` | `base.py` (Tool, Risk, permissions, audit), `fs.py`, `memory_tools.py`, `calc.py` |

## Rules that hold everywhere
- The model proposes; the application decides. Permissions are enforced by code, not by the prompt.
- Plain chat is fast: ~100-token stable system prompt, streamed, no JSON. Tools and memory only when the message needs them.
- The router and the calculator never call a model. One chip per turn; a manual `/use` choice lasts one message and
  never bypasses the RAM check; the Idle/Light chip never answers chat.
- Calculator: whitelist AST evaluator, bounded, pure computation (Risk.READ, no approval, does not taint the session).
- Background work never holds the lock during a model call; memory is touched only under `app.lock`.
- Everything is reversible or logged: note history, trash, `audit.jsonl`, `idle.log`, `routing.jsonl`.

## Data (`ksar_data/`, override with `KSAR_HOME`)
`config/` (settings.json, permissions.json, profile.json), `memory/` (notes, journal, .history, .trash), `tasks/`,
`workspace/` (the only default write area), `logs/` (audit.jsonl, idle.log, routing.jsonl).

## Where to extend
- New tool: `Tool(name, description, params, Risk, handler)` in `tools/`, add to `registry.py`; add a family prefix to
  `router.FAMILIES` only if it should answer to `@family`.
- Routing: signals live in `router.assess`; the chip decision in `ChipManager._auto`; per-step chip in
  `Agent._chip_for_step` (future mid-task escalation).

# Ksar 0.3.3 - architecture

Local-first assistant. Python 3.10+, standard library only. The model runs in Ollama on the same computer.
Per-version notes with how-to and limits: `docs/` (v3.1, v3.2, v3.3). This file is the short map.

## Layers
```
console (cli.py) --Enter--> Conversation.send ----------------------------> journal (raw record)  ---> job queue
        |                        |                                                                      |
        |                 Agent.turn                                                          IdleWorker (thread)
        |                   router.route_message(text)     no model call                        warm model
        |                   ChipManager.choose(...)         /use override > automatic            extract (journal -> notes)
        |                   Index.recall(text)              RAM only: core / branch / notes      recheck pending facts
        |                   TimeService.context_for(text)   only when asked or after a long gap  index/sync, split, check
        |                   LLM.stream_chat -> Ollama                                             organize, clock
        |                   ToolManager.call (tool mode)
        v
   live printer: thoughts, answer, footer
```
The chat path never reads the vault from disk and never waits for memory work. The worker never takes `app.lock`.

## Modules
| module | job |
|---|---|
| `app.py` | `Ksar`: wires settings, memory, index, tools, chips, agent, time, job queue, worker; journal watermark/catch-up |
| `cli.py` | console loop, commands, live printer |
| `router.py` | complexity score, tool/memory routing, calculator offer |
| `agent.py` | prompts, bounded tool loop, `Conversation` (history window, journal, queueing memory work), `TurnInfo` |
| `chips.py` | slots standard (Basic) / advanced / light (Idle), RAM-aware loading, `/use` override |
| `llm.py` | Ollama client: streaming, thinking split, JSON extraction |
| `idle.py` | worker thread and jobs |
| `timeinfo.py` | time service |
| `memcmd.py` | recognises pure memory commands ("remember what we discussed during the last hour") |
| `memory/store.py` | Markdown notes (one file per note, folders allowed), journal, history, trash, change log, revisions, core digest cache |
| `memory/retrieval.py` | incremental in-RAM index, 3-level `recall`, rule priority, explanations |
| `memory/keeper.py` | extraction: prepare -> plan (validation, grounding) -> apply (writes) |
| `memory/grounding.py` | concept-level support check for proposed facts |
| `memory/graph.py` | branches, core budget, relations, graph validation, organization plan |
| `memory/jobs.py` | persistent job queue |
| `memory/maintenance.py` | splitting oversized notes |
| `tools/` | tools, permissions, audit (`calc.evaluate`, `fs.*`, `memory.*`, `sys.info`) |

## Memory model
- One Markdown file per note; front matter `id title type summary parent status created updated` (+ any keys the owner added);
  `## Facts` (`- text #tag <!--f:ID c:CONF d:DATE s:SOURCE-->`), `## Links` (`- [[id]] (relation)`), `## Files`, `## History`.
- Note id = file name = link target (unique slug); the folder is only physical grouping. Notes may live in sub-folders.
- Hierarchy: `core` -> branch notes (type `index`) -> focused notes; relations in `## Links`.
- Markdown is the source of truth. The RAM index, the core digest cache and `queue.json` are rebuildable.
- Vault for Obsidian: `ksar_data/memory/` (`notes/` and `journal/` visible; `.history`, `.trash`, `.jobs` hidden).

## Concurrency rules
- `MemoryStore` has one small RLock around single operations. Nothing slow (model, search, maintenance) runs under it.
- Background plans record note revisions; `apply` raises `Conflict` if a note changed and the job is planned again.
- Foreground has priority: Enter cancels the running background call at its next token; the job returns to the queue.
- `app.lock` is only for foreground entry points and is never held during a model call or memory work.

## Where to extend
- New tool: `Tool(...)` in `tools/`, register in `registry.py`. New branch: `memory/graph.BRANCHES` (or let the model name one).
- Routing signals: `router.assess`. Chip decision: `ChipManager._auto`. Retrieval gate: `Index.recall`. Grounding groups: `grounding._GROUPS`.

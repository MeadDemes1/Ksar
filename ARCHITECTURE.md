# Ksar 0.2.1 - architecture

Ksar is built from independent modules. The model (LLM) never touches the computer: it asks the
**Tool Manager**, and the application enforces permissions, not the prompt.

```
            you (console now; Telegram / desktop UI later = just another front-end)
                                   |
 Conversation / Agent  --JSON-->  LLM (Ollama, replaceable: anything with .chat())
        |   \
        |    +--> Task Manager   persistent state machine, checkpoints, resume, no repeated steps
        v
   Tool Manager  -> validate -> risk -> permissions -> owner approval -> run -> verify -> audit.jsonl
        |
   +----+--------------------+
   fs.* tools            memory.* tools            (docs.*, system.* come in later phases)
                              |
                      Memory (Markdown notes)  <--  Keeper (LLM proposes, app validates & stores)
```

## Memory

One note = one `.md` file in `ksar_data/memory/notes/` (Obsidian-compatible, hand-editable):

```
---
id: lab-3   title: Lab 3   type: project   summary: ...   parent: school
---
## Facts
- Deadline: 2026-11-20 <!--f:a1b2c3 c:stated d:2026-11-13 s:chat:2026-11-13#4-->
## Links      - [[lab-3-template]] (derived_from) - ...
## Files      - C:\path\lab3.pdf | sha256:ab12... | 2026-11-13
## History    - ~~Deadline: 2026-11-18~~ (superseded 2026-11-14) <!--...-->
```

| Requirement from the brief | How it is done |
|---|---|
| Episodic / semantic / project / relational | note `type`s (`episodic`, `semantic`, `project`, `person`, `document`, `task`, `index`, `core`); typed `[[links]]`; `parent` gives the branching tree rooted at `core` |
| Preserve sources | every fact has `s:` = the exact journal turn (`chat:DATE#N` -> `memory/journal/DATE.md`) or `task:ID`; files are recorded with a SHA-256 fingerprint |
| Verified vs assumed | `c:` = `verified` / `stated` / `inferred`. The model can never write `verified`; facts learned after reading untrusted content are forced to `inferred` |
| No duplicates / contradictions | normalised + similarity dedupe; `Key: value` facts with the same key **supersede** the old one, which moves to `## History`; a `verified` fact is never overwritten by a weaker one (goes to the `review` note instead) |
| Never silently overwrite | old facts live in History; a snapshot of the whole note is saved before every change (`.history/<note>/`); `/restore` brings back any version |
| Correction & deletion | `memory.correct`, `memory.forget` (approval), `/forget-note` (soft delete to `.trash`), `/purge` (permanent, explicit) |
| Deterministic retrieval | BM25 over weighted fields (title, id, summary, file names, facts) + graph expansion over links/parent/children + fact-level selection. No model, no randomness: same memory + same query = same result (`/recall <q>` shows scores) |
| Integrity | atomic writes, `doctor` finds broken links, orphans, bad parents, duplicates, missing ids, changed/missing files; `/doctor fix` repairs what is safe |
| LLM decides, app stores | the Keeper asks the model for small JSON operations (add / correct / link / episode); ids, sizes, provenance, injection filter and conflict rules are applied by code |
| Branches that don't pile up | a note over 25 facts is split: the model only *groups* fact numbers, the app moves the exact lines (nothing is rewritten or lost) |

## Tools and permissions

Permissions are in `ksar_data/config/permissions.json`. Only the owner edits them (`/allow`, `/revoke`);
no tool can. Default: read + write only inside `ksar_data/workspace/`.

| Risk | Examples | Policy |
|---|---|---|
| read | fs.list, fs.search, fs.read_text, memory.search | allowed inside readable folders |
| write | fs.write_text (new file), fs.copy, fs.move, fs.make_dir, memory.remember | allowed inside writable folders, never overwrites |
| destructive | overwrite, fs.trash, memory.forget | owner approves every time; with no approver (background) the task parks in `waiting_approval` |
| exec | (no tool yet) | reserved; will always need approval |

Always blocked, even inside an allowed folder: Ksar's own source, `config/`, `logs/`, `memory/`, `.ssh`,
`AppData`, `.env`, keys, password stores. Paths are resolved first, so `..` and symlinks cannot escape.
Every write is read back and checked, and the result lists **verified effects**. Every call is appended to
`logs/audit.jsonl` (time, task, tool, risk, arguments, decision, result).

## Tasks

`ksar_data/tasks/<id>.json`, rewritten after every step. States: pending, planning, in_progress,
waiting_approval, paused, failed, completed, cancelled (illegal transitions raise). After a crash the task
comes back `paused` and `/resume` continues where it stopped; an action that already succeeded is skipped, not
repeated. A task's "completed" report lists what the **tools confirmed**, not what the model claims.

## Safety model (short)

1. Model -> tools only; permissions enforced in code.  2. Owner-only permission changes.
3. Approval for anything destructive.  4. File contents and tool output are labelled untrusted data.
5. Reading untrusted content marks the session *tainted*: later memory is only `inferred`.
6. Instruction-like text ("ignore your rules...") is refused for memory.  7. Everything reversible
(versions, trash, history).  8. Everything logged.

## Not built yet (by design, next phases)

PDF/Excel tools (Phase 2, will need `pypdf`/`openpyxl`), folder watchers, Telegram (Phase 4: just another
front-end calling `Conversation.send` / `TaskManager`, with an approver that asks you in Telegram), desktop app
(Phase 5), model escalation, GUI automation. Tool calling relies on the model emitting JSON; Qwen3 8B is
decent but not perfect, so every layer validates and recovers instead of trusting it.

## Add a tool

Write `def handler(ctx, args) -> ToolResult` (check paths with `ctx.perms`, return `effects` you verified),
wrap it in `Tool(name, description, params, Risk.X, handler)` and add it to a `tools()` list in
`ksar/tools/`. It is then audited, validated and permission-gated like every other tool.

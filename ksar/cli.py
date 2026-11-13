"""
Console interface. Commands are accepted flat (/tree, /task ...) or grouped (/memory tree).
The console and any future GUI must use the same settings backend (config.py / settings.json).
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass

from .agent import Agent, Conversation
from .tools.base import ApprovalRequest

PRIORITIES = ("speed", "balanced", "quality")
RETENTIONS = ("aggressive", "balanced", "persistent")

CONFIG_KEYS = {
    "performance.priority": ("performance_priority", "choice", PRIORITIES),
    "chips.retention": ("chips_retention", "choice", RETENTIONS),
    "chips.auto_escalation": ("auto_escalation", "bool", None),
    "chips.standard": ("standard_chip", "str", None),
    "chips.advanced": ("advanced_chip", "str", None),
    "llm.think": ("think", "bool", None),
    "llm.num_ctx": ("num_ctx", "int", None),
}

HELP = """
chat      just type; risky actions ask you [y/N]
memory    /tree  /recall <q>  /open <note>  /history <note>  /restore <note> <version>
          /forget-note <note>  /purge <note>  /doctor [fix]  /reflect
tasks     /task <objective>  /tasks  /resume <id>  /cancel <id>  /report <id>
safety    /perms  /allow read|write <folder>  /revoke read|write <folder>  /audit [n]
config    /config show | get <key> | set <key> <value>
chips     /chip [status] | active | load <chip> | unload <chip>      (also /chips)
          /chippref [speed|balanced|quality]    how eagerly the advanced chip is used
other     /help  /exit
"""


# ----------------------------------------------------------------- helpers
def _save_settings(app) -> None:
    """Persist settings through the app's own backend if it has one."""
    for name in ("save_settings", "save_config"):
        fn = getattr(app, name, None)
        if callable(fn):
            fn()
            return
    s = app.settings
    if hasattr(s, "save") and callable(s.save):
        s.save()
        return
    data = asdict(s) if is_dataclass(s) else vars(s)
    (app.paths.config / "settings.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _chip_manager(app):
    cm = getattr(app, "chips", None)
    if cm is None:
        print("Chip manager is not installed (patch app.py to enable chips).")
    return cm


# ----------------------------------------------------------------- memory
def cmd_memory(app, args, convo, on_event):
    if not args:
        print("memory: /tree /recall /open /history /restore /forget-note /purge /doctor /reflect")
        return
    sub, rest = args[0].lstrip("/"), args[1:]
    m = app.memory

    if sub == "tree":
        print(m.tree_text())
    elif sub == "recall":
        q = " ".join(rest)
        if not q:
            print("Usage: /recall <query>")
        else:
            print(app.index.context(q, budget=app.settings.recall_chars))
    elif sub == "open":
        if not rest:
            print("Usage: /open <note>")
        else:
            from .memory.store import render_note
            n = m.get(rest[0])
            print(render_note(n) if n else f"no note '{rest[0]}'")
    elif sub == "history":
        if not rest:
            print("Usage: /history <note>")
        else:
            versions = m.versions(rest[0])
            if not versions:
                print("no saved versions for that note")
            else:
                for v in versions:
                    print(v)
                print(f"restore with: /restore {rest[0]} <version>")
    elif sub == "restore":
        if len(rest) < 2:
            print("Usage: /restore <note> <version>   (see /history <note>)")
        else:
            print("restored." if m.restore_version(rest[0], rest[1]) else "version not found.")
    elif sub == "forget-note":
        if not rest:
            print("Usage: /forget-note <note>")
        else:
            print("moved to .trash." if m.delete_note(rest[0]) else "not found, or 'core' cannot be deleted.")
    elif sub == "purge":
        if not rest:
            print("Usage: /purge <note>")
        elif input(f"Permanently erase '{rest[0]}'? This cannot be undone. [y/N]: ").strip().lower() != "y":
            print("cancelled.")
        else:
            print("erased." if m.purge(rest[0]) else "note must be deleted first (/forget-note).")
    elif sub == "doctor":
        issues = m.doctor(fix=bool(rest and rest[0] == "fix"))
        if not issues:
            print("no issues found.")
        for i in issues:
            print(f"[{i.kind}] {i.note}: {i.detail}" + (" (fixed)" if i.fixed else ""))
    elif sub == "reflect":
        convo.flush(on_event)          # <- convo is now passed in
    else:
        print(f"unknown memory command '{sub}'")


# ----------------------------------------------------------------- tasks
def cmd_tasks(app, args, on_event):
    sub = args[0].lstrip("/") if args else "tasks"
    rest = args[1:] if args else []
    tm, agent = app.tasks, Agent(app)

    if sub == "task":
        objective = " ".join(rest)
        if not objective:
            print("Usage: /task <objective>")
            return
        t = tm.create(objective[:40], objective)
        print(f"task {t.id} created.")
        agent.run_task(t, on_event)
        print(Agent.report(t))
    elif sub == "tasks":
        tasks = tm.list()
        if not tasks:
            print("no tasks.")
        for t in tasks:
            print(f"{t.id} | {t.state.upper():16} | {t.title}")
    elif sub in ("resume", "cancel", "report"):
        if not rest:
            print(f"Usage: /{sub} <id>")
            return
        t = tm.get(rest[0])
        if not t:
            print(f"no task '{rest[0]}'")
            return
        if sub == "resume":
            agent.run_task(t, on_event)
            print(Agent.report(t))
        elif sub == "cancel":
            tm.transition(t, "cancelled", "cancelled by owner")
            print(f"task {t.id} cancelled.")
        else:
            print(Agent.report(t))
    else:
        print(f"unknown task command '{sub}'")


# ----------------------------------------------------------------- safety
def cmd_safety(app, args):
    if not args:
        print("safety: /perms /allow /revoke /audit")
        return
    sub, rest = args[0].lstrip("/"), args[1:]
    perms = app.tools.ctx.perms

    if sub == "perms":
        for kind in ("read", "write"):
            print(f"{kind} folders:")
            for r in perms.roots(kind):
                print(f"  {r}")
        print("Ksar's source, config, logs and memory are always blocked.")
    elif sub in ("allow", "revoke"):
        if len(rest) < 2 or rest[0] not in ("read", "write"):
            print(f"Usage: /{sub} read|write <folder>")
            return
        kind, path = rest[0], " ".join(rest[1:])
        try:
            if sub == "allow":
                print(f"{kind} allowed: {perms.add_root(kind, path)}")
            else:
                perms.remove_root(kind, path)
                print(f"{kind} revoked: {path}")
        except Exception as e:
            print(f"failed: {e}")
    elif sub == "audit":
        try:
            n = int(rest[0]) if rest else 10
        except ValueError:
            print("Usage: /audit [number]")
            return
        try:
            lines = app.tools.audit_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            print("no audit log yet.")
            return
        for line in lines[-n:]:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            print(f"[{r.get('ts')}] {r.get('tool')} {r.get('risk', '')} "
                  f"{r.get('decision', '')} {'ok' if r.get('ok') else ''} {r.get('message', '')[:70]}")
    else:
        print(f"unknown safety command '{sub}'")


# ----------------------------------------------------------------- config
def cmd_config(app, args):
    sub = args[0] if args else "show"

    if sub == "show":
        s = app.settings
        print("current settings:")
        for key, (attr, _kind, _choices) in CONFIG_KEYS.items():
            print(f"  {key:24} = {getattr(s, attr, '(unset)')}")
        print(f"  {'model (llm fallback)':24} = {app.llm.model}")
        return

    if sub == "get":
        if len(args) < 2 or args[1] not in CONFIG_KEYS:
            print("known keys: " + ", ".join(CONFIG_KEYS))
            return
        attr = CONFIG_KEYS[args[1]][0]
        print(f"{args[1]} = {getattr(app.settings, attr, '(unset)')}")
        return

    if sub == "set":
        if len(args) < 3 or args[1] not in CONFIG_KEYS:
            print("Usage: /config set <key> <value>")
            print("known keys: " + ", ".join(CONFIG_KEYS))
            return
        key, raw = args[1], " ".join(args[2:]).strip()
        attr, kind, choices = CONFIG_KEYS[key]

        if kind == "bool":
            if raw.lower() not in ("true", "false", "yes", "no", "1", "0"):
                print(f"invalid value '{raw}': expected true or false. Nothing changed.")
                return
            value = raw.lower() in ("true", "yes", "1")
        elif kind == "int":
            try:
                value = int(raw)
                if not 512 <= value <= 131072:
                    raise ValueError
            except ValueError:
                print(f"invalid value '{raw}': expected a whole number between 512 and 131072. Nothing changed.")
                return
        elif kind == "choice":
            if raw.lower() not in choices:
                print(f"invalid value '{raw}': expected one of {', '.join(choices)}. Nothing changed.")
                return
            value = raw.lower()
        else:
            value = "" if raw.lower() in ("none", "disabled", '""') else raw

        setattr(app.settings, attr, value)
        try:
            _save_settings(app)
        except Exception as e:
            print(f"could not save settings: {e}")
            return
        print(f"{key} = {value if value != '' else '(disabled)'}")

        cm = getattr(app, "chips", None)
        if cm is not None and attr in ("standard_chip", "advanced_chip"):
            app.llm.model = app.settings.standard_chip or app.llm.model
            cm._load_config()
            print("chip manager reloaded.")
        if attr == "num_ctx":
            app.llm.num_ctx = app.settings.num_ctx
        if attr == "think":
            app.llm.think = app.settings.think
        return

    print("Usage: /config show | get <key> | set <key> <value>")


# ----------------------------------------------------------------- chips
def cmd_chips(app, args):
    cm = _chip_manager(app)
    if cm is None:
        return
    sub = args[0] if args else "status"
    rest = args[1:]

    if sub in ("status", "list"):
        for r in cm.status_rows():
            flag = "" if r["installed"] in (None, True) else "  NOT PULLED -> ollama pull " + r["model"]
            print(f"{r['name']:9} {r['model']:20} {r['state']:9} ~{r['est_ram_mb']} MB  keep-alive {r['keep_alive']}{flag}")
            if r["note"]:
                print(f"          note: {r['note']}")
        res = getattr(app, "resources", None)
        if res is not None:
            r = res.get_resources()
            ram = f"{r.ram_free_mb} MB free / {r.ram_total_mb} MB total" if r.ram_known else "unknown (cannot read RAM)"
            print(f"RAM: {ram}   priority: {app.settings.performance_priority}   "
                  f"retention: {app.settings.chips_retention}   escalation: {app.settings.auto_escalation}")
        if "advanced" not in cm.chips:
            print("advanced chip: off. Enable: /config set chips.advanced <model>")
    elif sub == "active":
        active = cm.active_chip
        print(f"active chip: {active.name} ({active.model_id})" if active else "no chip is active.")
    elif sub in ("load", "unload"):
        if not rest or rest[0] not in cm.chips:
            print(f"Usage: /chip {sub} <chip>   available: {', '.join(cm.chips) or 'none'}")
            return
        print(cm.load_chip(rest[0]) if sub == "load" else cm.unload_chip(rest[0]))
    else:
        print("Usage: /chip [status] | active | load <chip> | unload <chip>")


def cmd_chippref(app, args):
    if not args:
        print(f"priority: {app.settings.performance_priority}  (speed | balanced | quality)")
        return
    cmd_config(app, ["set", "performance.priority", args[0]])


# ----------------------------------------------------------------- dispatch
GROUPS = {"memory": cmd_memory, "tasks": cmd_tasks, "safety": cmd_safety}
MEMORY = {"tree", "recall", "open", "history", "restore", "forget-note", "purge", "doctor", "reflect"}
TASKS = {"task", "tasks", "resume", "cancel", "report"}
SAFETY = {"perms", "allow", "revoke", "audit"}


def handle_command(raw: str, app, convo, on_event) -> bool:
    """Returns False when the console should exit."""
    parts = raw.lstrip("/").split()
    if not parts:
        return True
    cmd, args = parts[0].lower(), parts[1:]

    if cmd in ("exit", "quit", "bye"):
        return False
    if cmd in ("help", "?"):
        print(HELP)
    elif cmd == "config":
        cmd_config(app, args)
    elif cmd in ("chips", "chip"):
        cmd_chips(app, args)
    elif cmd == "chippref":
        cmd_chippref(app, args)
    elif cmd in GROUPS:                       # grouped form: /memory tree
        handler = GROUPS[cmd]
        if handler is cmd_memory:
            handler(app, args, convo, on_event)
        elif handler is cmd_tasks:
            handler(app, args, on_event)
        else:
            handler(app, args)
    elif cmd in MEMORY:                       # flat form: /tree
        cmd_memory(app, [cmd] + args, convo, on_event)
    elif cmd in TASKS:
        cmd_tasks(app, [cmd] + args, on_event)
    elif cmd in SAFETY:
        cmd_safety(app, [cmd] + args)
    else:
        print(f"unknown command '/{cmd}'. /help lists the commands.")
    return True


# ----------------------------------------------------------------- main loop
def run(app):
    convo = Conversation(app)

    def on_event(msg):
        print(msg)

    def approver(req: ApprovalRequest) -> bool:
        print(f"\n  APPROVAL NEEDED: {req.tool} [{req.risk.name}]")
        print(f"  {req.summary}")
        return input("  allow? [y/N]: ").strip().lower() in ("y", "yes")

    app.tools.approver = approver

    recovered = app.tasks.recover()
    if recovered:
        print(f"{len(recovered)} interrupted task(s) recovered as paused. /tasks to see them.")
    print(f"{app.names.name} ready, {app.names.user}. /help for commands, /exit to quit.")

    while True:
        try:
            line = input(f"\n{app.names.user}> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not line:
            continue
        try:
            if line.startswith("/"):
                if not handle_command(line, app, convo, on_event):
                    break
            else:
                print()
                streamed = [False]
                answer_open = [False]

                def stream_token(token):
                    if not answer_open[0]:
                        print(f"{app.names.name}> ", end="", flush=True)
                        answer_open[0] = True
                        streamed[0] = True
                    print(token, end="", flush=True)

                def turn_event(message):
                    if answer_open[0]:
                        print()
                        answer_open[0] = False
                    on_event(message)

                reply = convo.send(line, turn_event, on_token=stream_token)
                if answer_open[0]:
                    print()
                elif not streamed[0]:
                    print(f"{app.names.name}> {reply}")
        except KeyboardInterrupt:
            print("\n(interrupted)")
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")

    try:
        convo.flush(on_event)
    except Exception as e:
        print(f"memory flush on exit failed: {e}")
    print("goodbye.")


def main():
    """Entry point for __main__.py. Constructs Ksar and starts the console."""
    from .app import Ksar
    ksar = Ksar()
    run(ksar)


run_cli = run
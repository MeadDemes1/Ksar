"""
Tool Manager: the ONLY way the model touches the computer.

Every call is: looked up -> validated -> risk-assessed -> permission-checked -> (approved by a human if
risky) -> executed -> audited. Permissions live in a file only the owner edits (config/permissions.json);
no tool can change them.
"""
from __future__ import annotations

import fnmatch
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import Paths, Settings


class Risk(IntEnum):
    READ = 0
    WRITE = 1
    DESTRUCTIVE = 2
    EXEC = 3


@dataclass
class Param:
    type: str = "string"            # string | integer | boolean
    desc: str = ""
    required: bool = True
    default: Any = None
    allow_empty: bool = False


@dataclass
class ToolResult:
    ok: bool
    message: str = ""
    data: str = ""
    effects: list = field(default_factory=list)   # tool-verified facts about what changed
    untrusted: bool = False                       # content came from outside (files, web...)
    needs_approval: bool = False


@dataclass
class Session:
    """State shared by every tool call of one chat session or one task."""
    task_id: str = "chat"
    src: str = "chat"
    tainted: bool = False        # True once untrusted content (e.g. a file's text) was read


@dataclass
class ToolContext:
    settings: Settings
    paths: Paths
    perms: "Permissions"
    memory: Any
    index: Any
    session: Session


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    risk: Risk
    handler: Callable[[ToolContext, dict], ToolResult]
    risk_fn: Optional[Callable[[ToolContext, dict], Risk]] = None


# ------------------------------------------------------------------ permissions
DENY_PARTS = {".ssh", ".gnupg", ".aws", ".azure", "appdata", ".kube"}
DENY_NAMES = ("*.pem", "*.key", "*.kdbx", ".env", ".env.*", "id_rsa*", "id_ed25519*", "login data",
              "cookies", "*.pfx", "wallet.dat")


def _inside(p: Path, root: Path) -> bool:
    try:
        a, b = os.path.normcase(str(p)), os.path.normcase(str(root))
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


class Permissions:
    DEFAULTS = {"read_roots": [], "write_roots": [], "approve_min_risk": int(Risk.DESTRUCTIVE)}

    def __init__(self, paths: Paths):
        self.paths = paths
        self.file = paths.config / "permissions.json"
        self.protected = [paths.base, Path(__file__).resolve().parent.parent]   # data dir + source code
        self.data: dict = {}
        self._mtime = None
        self.reload(force=True)

    def reload(self, force: bool = False) -> None:
        """Re-reads permissions.json only when the owner edited it (it used to be rewritten on every tool call)."""
        try:
            mtime = self.file.stat().st_mtime_ns
        except OSError:
            mtime = None
        if not force and mtime is not None and mtime == self._mtime:
            return
        d = dict(self.DEFAULTS)
        if self.file.exists():
            try:
                d.update(json.loads(self.file.read_text(encoding="utf-8")))
            except Exception:
                pass
        ws = str(self.paths.workspace)
        for k in ("read_roots", "write_roots"):
            if ws not in d[k]:
                d[k] = [ws] + [x for x in d[k] if x != ws]
        self.data = d
        self.file.write_text(json.dumps(d, indent=2), encoding="utf-8")
        try:
            self._mtime = self.file.stat().st_mtime_ns
        except OSError:
            self._mtime = None

    @property
    def approve_min_risk(self) -> int:
        return int(self.data.get("approve_min_risk", Risk.DESTRUCTIVE))

    def roots(self, kind: str) -> list:
        return [Path(r).resolve() for r in self.data.get(f"{kind}_roots", [])]

    def add_root(self, kind: str, path: str) -> Path:
        p = Path(path).expanduser().resolve()
        if not p.is_dir():
            raise ValueError("not an existing folder")
        key = f"{kind}_roots"
        if str(p) not in self.data[key]:
            self.data[key].append(str(p))
            self.file.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        return p

    def remove_root(self, kind: str, path: str) -> None:
        key = f"{kind}_roots"
        ws = str(self.paths.workspace)
        p = str(Path(path).expanduser().resolve())
        if p != ws:
            self.data[key] = [r for r in self.data[key] if r != p]
            self.file.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def resolve(self, path: str) -> Path:
        p = Path(os.path.expandvars(str(path))).expanduser()
        if not p.is_absolute():
            p = self.paths.workspace / p
        return p.resolve()

    def is_denied(self, p: Path) -> Optional[str]:
        ws = self.paths.workspace.resolve()
        for prot in self.protected:
            if _inside(p, prot) and not _inside(p, ws):
                return "protected Ksar folder"
        parts = {x.lower() for x in p.parts}
        if parts & DENY_PARTS:
            return "sensitive location"
        name = p.name.lower()
        if any(fnmatch.fnmatch(name, pat) for pat in DENY_NAMES):
            return "sensitive file type"
        return None

    def check(self, p: Path, kind: str) -> Optional[str]:
        """Returns None if allowed, else the reason it is not."""
        why = self.is_denied(p)
        if why:
            return why
        if not any(_inside(p, r) for r in self.roots(kind)):
            return f"outside the folders Ksar may {kind} (owner can allow with /allow {kind} <folder>)"
        return None


# ------------------------------------------------------------------ manager
@dataclass
class ApprovalRequest:
    tool: str
    risk: Risk
    summary: str
    task_id: str


class ToolManager:
    def __init__(self, ctx: ToolContext, audit_path: Path, approver: Optional[Callable[[ApprovalRequest], bool]] = None):
        self.ctx = ctx
        self.audit_path = Path(audit_path)
        self.approver = approver
        self.tools: dict = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def catalog(self) -> str:
        rows = []
        for t in self.tools.values():
            args = ", ".join(f"{k}:{p.type}{'' if p.required else '?'}" for k, p in t.params.items())
            rows.append(f"- {t.name}({args}) [{t.risk.name.lower()}] {t.description}")
        return "\n".join(rows)

    def _audit(self, **rec) -> None:
        rec = {"ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
               "task": self.ctx.session.task_id, **rec}
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _validate(tool: Tool, args) -> tuple:
        if not isinstance(args, dict):
            return None, "args must be an object"
        for k in args:
            if k not in tool.params:
                return None, f"unknown argument '{k}' (allowed: {', '.join(tool.params) or 'none'})"
        clean = {}
        for k, p in tool.params.items():
            v = args.get(k)
            if v is None or (v == "" and not p.allow_empty):
                if p.required:
                    return None, f"missing argument '{k}'"
                if p.default is not None:
                    clean[k] = p.default
                continue
            try:
                if p.type == "integer":
                    v = int(v)
                elif p.type == "boolean":
                    v = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")
                else:
                    v = str(v)
                    if len(v) > 30000:
                        return None, f"argument '{k}' too long"
            except Exception:
                return None, f"argument '{k}' must be {p.type}"
            clean[k] = v
        return clean, ""

    @staticmethod
    def _short(args: dict) -> dict:
        return {k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v) for k, v in args.items()}

    def call(self, name: str, args) -> ToolResult:
        tool = self.tools.get(name)
        if not tool:
            self._audit(tool=name, decision="unknown-tool")
            return ToolResult(False, f"unknown tool '{name}'. Available: {', '.join(self.tools)}")
        clean, err = self._validate(tool, args)
        if err:
            self._audit(tool=name, decision="invalid", message=err)
            return ToolResult(False, err)
        self.ctx.perms.reload()
        risk = tool.risk_fn(self.ctx, clean) if tool.risk_fn else tool.risk
        decision = "auto"
        if int(risk) >= self.ctx.perms.approve_min_risk:
            req = ApprovalRequest(name, risk, f"{name} {json.dumps(self._short(clean), ensure_ascii=False)}", self.ctx.session.task_id)
            if self.approver is None:
                self._audit(tool=name, risk=risk.name, args=self._short(clean), decision="needs-approval")
                return ToolResult(False, "this action needs the owner's approval and none is available right now", needs_approval=True)
            if not self.approver(req):
                self._audit(tool=name, risk=risk.name, args=self._short(clean), decision="denied")
                return ToolResult(False, "the owner denied this action")
            decision = "approved"
        try:
            res = tool.handler(self.ctx, clean)
        except Exception as e:
            res = ToolResult(False, f"tool error: {type(e).__name__}: {e}")
        if res.untrusted and res.ok:
            self.ctx.session.tainted = True
        self._audit(tool=name, risk=risk.name, args=self._short(clean), decision=decision, ok=res.ok,
                    message=res.message[:200], effects=res.effects)
        return res

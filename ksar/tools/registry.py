from . import fs, memory_tools


def register_default_tools(manager) -> None:
    for t in fs.tools() + memory_tools.tools():
        manager.register(t)

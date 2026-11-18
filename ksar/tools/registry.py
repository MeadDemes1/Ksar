from . import calc, fs, memory_tools


def register_default_tools(manager) -> None:
    for t in fs.tools() + memory_tools.tools() + calc.tools():
        manager.register(t)

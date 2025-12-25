from langchain_core.callbacks import BaseCallbackHandler
from rich.console import Console
import os
import sys
import ctypes

console = Console()

class ToolLoggingHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        """Run when tool starts running."""
        name = serialized.get("name", "unknown_tool")
        console.print(f"[bold magenta]→ Tool used: {name}[/bold magenta]")
        if input_str.strip():
            preview = input_str[:100] + "..." if len(input_str) > 100 else input_str
            console.print(f"  Input: {preview}", style="dim")

    def on_tool_end(self, output, **kwargs) -> None:
        """Run when tool ends running."""
        preview = str(output)[:500] + "..." if len(str(output)) > 500 else str(output)
        console.print(f"  → Output: {preview}", style="green dim")
        console.print("")

    # Optional: log errors
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        console.print(f"[bold red]→ Tool error: {type(error).__name__}[/bold red]")


def set_terminal_name(name: str):
    if sys.platform.startswith("win"):
        ctypes.windll.kernel32.SetConsoleTitleW(name)
    else:
        # Linux/macOS
        print(f"\033]0;{name}\007", end="", flush=True)
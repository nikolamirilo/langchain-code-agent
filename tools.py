from langchain_core.tools import tool
from tavily import TavilyClient
from datetime import datetime
from pathlib import Path
import subprocess
import shlex
import http.client
import json
from secrets import VECTORIZE_API_KEY, TAVILIY_API_KEY


WORKDIR = Path("./").resolve()
WORKDIR.mkdir(exist_ok=True)
FORBIDDEN = ["sudo", "rm -rf", "&&", ";", "|", ">", "<"]

@tool
def create_folder(path: str) -> str:
    """
    Create a folder inside the agent working directory.
    Path must be relative (no absolute paths).
    """
    target = (WORKDIR / path).resolve()

    if not str(target).startswith(str(WORKDIR)):
        return "Error: Path outside working directory is not allowed."

    target.mkdir(parents=True, exist_ok=True)
    return f"Folder created at {target.relative_to(WORKDIR)}"

@tool
def write_file(path: str, content: str) -> str:
    """
    Create or overwrite a file inside the agent working directory.
    """
    target = (WORKDIR / path).resolve()

    if not str(target).startswith(str(WORKDIR)):
        return "Error: Path outside working directory is not allowed."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    return f"File written at {target.relative_to(WORKDIR)}"

@tool
def read_file(path: str) -> str:
    """Read a file from the working directory."""
    target = (WORKDIR / path).resolve()

    if not str(target).startswith(str(WORKDIR)):
        return "Error: Path outside working directory is not allowed."

    if not target.exists():
        return "Error: File does not exist."

    return target.read_text(encoding="utf-8")

@tool
def browse_web(query: str) -> str:
    """Browse the web for information."""
    tavlyClient = TavilyClient(TAVILIY_API_KEY)
    response = tavlyClient.search(query, num_results=3)
    return f"Results for {query}: {response}"

@tool
def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool 
def searchVectors(query: str) -> str:
    """Search a vector database for relevant information."""
    # Placeholder implementation
    conn = http.client.HTTPSConnection("api.vectorize.io")
    payload = json.dumps({
    "question": query,
    "numResults": 3,
    "rerank": True,
    "metadata-filters": [
        {}
    ],
    "context": {
        "messages": [
        {
            "role": "string",
            "content": "string"
        }
        ]
    },
    "advanced-query": {
        "mode": "vector",
        "match-type": "match",
    }
    })
    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': 'Bearer ' + VECTORIZE_API_KEY
    }
    conn.request("POST", "/v1/org/da410fd8-3465-4f53-804c-374d6b8c7d5a/pipelines/aip96917-704e-478c-a982-9d0ab53ae098/retrieval", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data

@tool
def request_command_execution(command: str) -> str:
    """
    Propose a terminal command for execution.
    This tool DOES NOT execute anything.
    Human approval is required.
    """
    return (
        "COMMAND_PROPOSAL\n"
        f"Command: {command}\n"
        "Awaiting human approval."
    )

@tool
def execute_approved_command(command: str) -> str:
    """
    Execute a previously approved command inside the working directory.
    """
    if any(bad in command for bad in FORBIDDEN):
        return "Error: Forbidden command detected."

    try:
        args = shlex.split(command)

        result = subprocess.run(
            args,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=30
        )

        return (
            f"Exit code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    except Exception as e:
        return f"Execution failed: {str(e)}"

@tool
def print_tree(
    path: str = ".",
    max_depth: int = 3,
    show_files: bool = True
) -> str:
    """
    Print a directory tree inside the working directory.
    """
    base = (WORKDIR / path).resolve()

    if not str(base).startswith(str(WORKDIR)):
        return "Error: Path outside working directory is not allowed."

    if not base.exists():
        return "Error: Path does not exist."

    lines = []

    def walk(dir_path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return

        entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + entry.name)

            if entry.is_dir():
                extension = "    " if is_last else "│   "
                walk(entry, prefix + extension, depth + 1)
            elif not show_files:
                lines.pop()

    lines.append(base.name)
    if base.is_dir():
        walk(base)

    return "\n".join(lines)
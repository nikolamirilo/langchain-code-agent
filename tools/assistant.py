from langchain_core.tools import tool
from tavily import TavilyClient
from datetime import datetime
from pathlib import Path
import subprocess
import shlex
import os

TAVILIY_API_KEY = os.getenv("TAVILY_API_KEY")
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


# Vector search imports and configuration
from dotenv import load_dotenv
from supabase import create_client
from langchain_openai import OpenAIEmbeddings
from typing import Optional, Dict, Any

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@tool
def search_vectors(
    query: str,
    top_k: int = 5,
    filter_column: Optional[str] = None,
    filter_value: Optional[str] = None
) -> str:
    """
    Search the vector database for documents similar to the query.
    Use this tool to find relevant information from uploaded CSV data.
    
    Args:
        query: The search query text to find similar documents
        top_k: Number of results to return (default: 5)
        filter_column: Optional column name to filter by (e.g., 'category', 'status')
        filter_value: The value to filter on (required if filter_column is provided)
    
    Returns:
        Formatted search results with content and metadata
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return "Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env"
    
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY must be set in .env"
    
    try:
        # Initialize clients
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY  # type: ignore
        )
        
        # Generate embedding for query
        query_vector = embeddings.embed_query(query)
        
        # Build filter if provided
        filter_dict = {}
        if filter_column and filter_value:
            filter_dict[filter_column] = filter_value
        
        # Call the match_documents RPC function
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_vector,
                "match_count": top_k,
                "filter": filter_dict
            }
        ).execute()
        
        if not response.data:
            return "No matching documents found."
        
        # Format results
        results = []
        for i, doc in enumerate(response.data, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            similarity = doc.get("similarity", 0)
            
            result = f"--- Result {i} (similarity: {similarity:.3f}) ---\n"
            result += f"Content:\n{content}\n"
            result += f"Source: {metadata.get('_source_file', 'unknown')}, Row: {metadata.get('_row_index', 'unknown')}"
            results.append(result)
        
        return "\n\n".join(results)
        
    except Exception as e:
        return f"Search error: {str(e)}"
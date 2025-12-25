# LangChain Code Agent

**LangChain Code Agent** is an interactive command‑line assistant powered by a large language model (LLM) via **LangChain** and **Groq**. It can answer questions, browse the web, read/write files, manage directories, execute approved shell commands, and query vector databases – all through a unified chat interface.

## Features
- **Chat UI** in the terminal with rich markdown rendering.
- Integrated **tools**:
  - `browse_web` – search the web (Tavily).
  - `get_current_time` – fetch current timestamp.
  - `read_file` / `write_file` – file I/O within a sandboxed working directory.
  - `create_folder` – create directories safely.
  - `searchVectors` – placeholder for vector DB lookup.
  - `request_command_execution` / `execute_approved_command` – propose and run shell commands (with safety checks).
  - `print_tree` – visualise the workspace tree.
- **Tool logging** with colourful output for easy debugging.
- **Safety**: all file operations are confined to the project’s working directory and forbidden command patterns are blocked.

## Quick Start
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt   # (ensure langchain, langchain-groq, rich, tavily, etc.)
   ```
2. **Set environment variables**
   ```bash
   export GROQ_API_KEY=your_groq_key
   export TAVILIY_API_KEY=your_tavily_key
   export VECTORIZE_API_KEY=your_vectorize_key
   ```
3. **Run the agent**
   ```bash
   python main.py
   ```
   Type your question and press **Enter**. Use `Ctrl+C` to exit.

## How It Works
- `main.py` creates a `ChatGroq` LLM instance and builds an agent with the tools defined in `tools.py`.
- The conversation loop sends user messages to the agent, receives the LLM’s response, and prints it using **Rich** markdown.
- Each tool is a LangChain `@tool` function that the LLM can invoke automatically. The `ToolLoggingHandler` logs tool usage.

## Safety Measures
- All paths are resolved against a sandbox `WORKDIR` (`./`). Operations outside this directory are rejected.
- A blacklist (`FORBIDDEN`) prevents dangerous shell constructs.
- Command execution requires a human‑approved proposal via `request_command_execution`.

## Extending the Agent
Add new tools by decorating a function with `@tool` in `tools.py` and include it in the `tools` list when creating the agent.

---
*Built by a Technical Lead with 20+ years of experience.*
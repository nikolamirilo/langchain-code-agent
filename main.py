from langchain_groq import ChatGroq
from langchain.agents import create_agent
from rich.console import Console
from rich.markdown import Markdown
from tools import browse_web, get_current_time, read_file, write_file, create_folder, request_command_execution, execute_approved_command, print_tree
from utils import ToolLoggingHandler, set_terminal_name
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


console = Console()
set_terminal_name("LangChain Agent")
handler = ToolLoggingHandler()


print("Agent started. Press CTRL+C to exit.\n")
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY,#type:ignore
)

agent = create_agent(
    model=llm,                  
    tools=[browse_web, get_current_time, read_file, write_file, create_folder, request_command_execution, execute_approved_command, print_tree],
    system_prompt="You are an Technical Lead with more than 20 years of experience."
)

messages = []

if __name__ == "__main__":
    try:
        while True:
            set_terminal_name("LangChain Agent")
            question = input("You: ").strip()
            if not question:
                continue

            messages.append({"role": "user", "content": question})

            result = agent.invoke({
                "messages": messages},
                 config={"callbacks": [handler]
            })

            # Extract last AI message
            ai_message = next(
                msg for msg in reversed(result["messages"])
                if msg.type == "ai" and msg.content
            )

            console.print("\nAssistant:", Markdown(ai_message.content), "\n")

            # Persist conversation
            messages.append({"role": "assistant", "content": ai_message.content})

    except KeyboardInterrupt:
        print("\n\nExiting conversation. Goodbye 👋")
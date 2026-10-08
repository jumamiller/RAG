import os
from dotenv import load_dotenv
import urllib.request
import urllib.error
import http.client
import time
load_dotenv()
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""

@tool
def fetch_textt_from_url(url:str)->str:
    """Fetch the document from a URL"""

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    # The server sometimes drops the connection mid-download (IncompleteRead).
    # Retry a few times so we get the COMPLETE file -- truncated text would
    # produce wrong line counts.
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as res:
                raw = res.read()
            text = raw.decode("utf-8", errors="replace")
            return text
        except (urllib.error.URLError, http.client.IncompleteRead) as e:
            last_error = e
            time.sleep(1)
    return f"Failed to fetch document after 3 attempts: {last_error}"


model=init_chat_model(
    "azure_openai:gpt-4o-mini",
    temperature=0.5,
    timeout=300,
    max_tokens=2500,
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
)

checkpointer=InMemorySaver()

agent=create_agent(
    model=model,
    tools=[fetch_textt_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

deep_agent=create_deep_agent(
    model=model,
    tools=[fetch_textt_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# The human turn is now a reusable template: the structure is fixed, but the
# book, url, and search terms are filled in at call time.
user_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Project Gutenberg hosts a full plain-text copy of {book}.
URL: {url}

Answer as much as you can:

1) How many lines in the complete Gutenberg file contain the substring `{term1}` (count lines, not occurrences within a line, each line ends with a line break).
2) The 1-based line number of the first line in the file that contains `{term2}`.
3) A two-sentence neutral synopsis.

Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was and what the error message was.""",
        )
    ]
)

# Fill the template -> a list of messages ready for the agent.
messages = user_prompt.invoke(
    {
        "book": "F. Scott Fitzgerald's The Great Gatsby",
        "url": "https://www.gutenberg.org/files/64317/64317-0.txt",
        "term1": "Gatsby",
        "term2": "Daisy",
    }
).to_messages()

agent_result = agent.invoke(
    {"messages": messages},
    config={"configurable": {"thread_id": "great-gatsby-lc"}},
)
deep_agent_result = deep_agent.invoke(
    {"messages": messages},
    config={"configurable": {"thread_id": "great-gatsby-da"}},
)
print(agent_result["messages"][-1].content_blocks)
print("\n")
print(deep_agent_result["messages"][-1].content_blocks)
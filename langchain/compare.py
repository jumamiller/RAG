r"""
Exercise: reproduce the tool-calling loop that `create_agent` did for you,
using ONLY the raw `openai` SDK (AzureOpenAI). No langchain here.

Goal: ask "What is the weather in New York?", let the model decide to call
get_weather, run the Python function yourself, feed the result back, and get
the final natural-language answer.

Fill in every TODO. Run with:
    .\.venv\Scripts\python.exe .\langchain\compare.py
"""

import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


# --- 1. The actual tool (same as index.py) ---------------------------------
def get_weather(city: str) -> str:
    print(f"[tool] get_weather called with {city!r}")
    return f"The weather in {city} is sunny."


# --- 2. Describe the tool to the model as a JSON schema --------------------
# This dict is EXACTLY what langchain auto-generated from your function's
# type hints (`city: str`) and docstring. Here you write it by hand so you
# can see there's no magic -- it's just a JSON Schema describing the args.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather for.",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


# --- 3. A registry so you can dispatch a tool call by name -----------------
# The model only sends back the NAME of the function as a string. You need a
# way to turn that string into the real Python callable.
available_tools = {
    "get_weather": get_weather,
}


def main() -> None:
    client = AzureOpenAI(
        # AzureOpenAI auto-reads AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
        # and OPENAI_API_VERSION from the environment.
    )
    deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

    # 4. Seed the conversation.
    messages = [
        {"role": "user", "content": "What is the weather in New York?"},
    ]

    # 5. THE LOOP. Keep calling the model until it stops asking for tools.
    while True:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message

        # (a) No tool calls -> the model produced a final answer. Done.
        if not msg.tool_calls:
            print(msg.content)
            break

        # (b) The model wants one or more tools run.
        # First, record the model's OWN turn (the request) in the history.
        # Each tool result we add next must reference these tool_call ids, so
        # this message MUST be in the transcript or the API rejects it.
        messages.append(msg)

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            # Arguments arrive as a JSON *string*; parse to a dict.
            args = json.loads(tool_call.function.arguments)

            # Dispatch to the real Python function.
            result = available_tools[name](**args)

            # Hand the result back to the model, tagged with the matching id.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
        # Loop: the model now sees the tool output and decides what to do next
        # (usually: write the final answer, which exits via branch (a)).


if __name__ == "__main__":
    main()

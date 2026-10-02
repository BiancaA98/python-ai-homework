import json
import os
import subprocess
import sys

from dotenv import load_dotenv
from openai import OpenAI


REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def list_files():
    result = subprocess.run(
        "dir /s /b",
        shell=True,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    return result.stdout


def search_pytest():
    result = subprocess.run(
        'findstr /s /i /m /c:"pytest" *',
        shell=True,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        errors="replace",
    )
    # Codul 1 inseamna ca nu exista potriviri.
    if result.returncode not in (0, 1):
        result.check_returncode()
    return result.stdout


tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Listeaza toate fisierele din repository cu dir /s /b.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pytest",
            "description": "Cauta pytest in fisiere cu findstr /s /i /m.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def complete(client, messages):
    return client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=messages,
        tools=tools,
    ).choices[0].message


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Apeleaza mai intai list_files si asteapta rezultatul. "
                "Apoi apeleaza search_pytest. Dupa rularea ambelor tools, "
                "returneaza lista fisierelor in care a fost gasit pytest, "
                "exact asa cum apar in rezultatul search_pytest. "
                "Daca lista este goala, spune ca nu exista potriviri."
            ),
        },
        {"role": "user", "content": "In ce fisiere din repository apare pytest?"},
    ]
    functions = {"list_files": list_files, "search_pytest": search_pytest}

    while True:
        message = complete(client, messages)
        messages.append(message)
        if not message.tool_calls:
            print(message.content)
            break

        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            result = functions[tool_call.function.name](**arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


def test_search_pytest():
    assert "tema2.py" in search_pytest()


if __name__ == "__main__":
    main()

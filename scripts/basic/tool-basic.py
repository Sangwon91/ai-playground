import anthropic
from dotenv import load_dotenv
from rich import print

load_dotenv()

client = anthropic.Anthropic()

messages = [
    {"role": "user", "content": "What's the weather like in San Francisco? 섭씨로 답해줘"}
]


response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    tools=[
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ],
    messages=messages,
)


print(response)

messages.append(
    {
        "role": "assistant",
        "content": [c.to_dict() for c in response.content],
    }
)

messages.append(
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                # TODO: A more smart way to get the tool_use_id is needed.
                "tool_use_id": response.content[1].to_dict().get("id"), # from the API response
                "content": "65 degrees" # from running your tool
            }
        ]
    }
)

response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    tools=[
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ],
    messages=messages,
)

print('-' * 80)
print(response)

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()

stream = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude",
        }
    ],
    model="claude-3-5-sonnet-latest",
    stream=True,
)
for event in stream:
    print(f"{event} | {event.type}")

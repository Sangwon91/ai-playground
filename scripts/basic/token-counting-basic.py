import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
)

count = client.beta.messages.count_tokens(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello, world"}
    ]
)

print(count.input_tokens)

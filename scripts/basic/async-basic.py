import os
import asyncio
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

load_dotenv()

client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
)


async def main() -> None:
    message = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude",
            }
        ],
        model="claude-3-5-sonnet-latest",
    )
    print(message.content)


asyncio.run(main())
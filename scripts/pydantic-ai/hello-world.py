import dotenv
import logfire
import os
from pydantic_ai import Agent

dotenv.load_dotenv()
logfire.configure(send_to_logfire=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_pydantic_ai()
agent = Agent(  
    'anthropic:claude-3-5-sonnet-latest',
    system_prompt='Be concise, reply with one sentence.',  
)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""


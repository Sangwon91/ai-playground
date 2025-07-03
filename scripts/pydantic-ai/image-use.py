from pydantic_ai import Agent, ImageUrl
import dotenv
import logfire
import os
from pydantic_ai import Agent

dotenv.load_dotenv()
logfire.configure(send_to_logfire=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_pydantic_ai()

agent = Agent(model='google-gla:gemini-2.5-flash')
result = agent.run_sync(
    [
        'What company is this logo from?',
        ImageUrl(url='https://iili.io/3Hs4FMg.png'),
    ]
)
print(result.output)
# > This is the logo for Pydantic, a data validation and settings management library in Python.
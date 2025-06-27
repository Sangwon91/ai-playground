from __future__ import annotations as _annotations

from dotenv import load_dotenv

import asyncio
import json
import sqlite3
from collections.abc import AsyncIterator
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, TypeVar

import fastapi
import logfire
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from typing_extensions import LiteralString, ParamSpec, TypedDict
import uuid
from fastapi.staticfiles import StaticFiles

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

load_dotenv()
print('Dotenv loaded')


# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

agent = Agent('anthropic:claude-sonnet-4-20250514')
print(agent)
THIS_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        yield {'db': db}


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


# 정적 파일 서빙 (static 디렉토리)
STATIC_DIR = THIS_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get('/')
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')


async def get_db(request: Request) -> Database:
    return request.state.db


@app.get('/chat/')
async def get_chat(session_id: str = 'default', database: Database = Depends(get_db)) -> Response:
    msgs = await database.get_messages(session_id=session_id)
    return Response(
        b'\n'.join(json.dumps(to_chat_message(m)).encode('utf-8') for m in msgs),
        media_type='text/plain',
    )


@app.get('/sessions/')
async def get_sessions(database: Database = Depends(get_db)) -> list[dict]:
    return await database.get_sessions()


@app.post('/session/')
async def create_session() -> dict:
    session_id = str(uuid.uuid4())
    return {'session_id': session_id}


@app.post('/chat/')
async def post_chat(
    prompt: Annotated[str, fastapi.Form()],
    session_id: Annotated[str, fastapi.Form()] = 'default',
    database: Database = Depends(get_db)
) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # stream the user prompt so that can be displayed straight away
        yield (
            json.dumps(
                {
                    'role': 'user',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': prompt,
                }
            ).encode('utf-8')
            + b'\n'
        )
        # get the chat history so far to pass as context to the agent
        messages = await database.get_messages(session_id=session_id)
        # run the agent with the user prompt and the chat history
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream(debounce_by=0.01):
                # text here is a `str` and the frontend wants
                # JSON encoded ModelResponse, so we create one
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

        # add new messages (e.g. the user prompt and the agent response in this case) to the database
        await database.add_messages(result.new_messages_json(), session_id=session_id)

    return StreamingResponse(stream_messages(), media_type='text/plain')


P = ParamSpec('P')
R = TypeVar('R')


@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'
    ) -> AsyncIterator[Database]:
        with logfire.span('connect to DB'):
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect, file)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file))
        con = logfire.instrument_sqlite3(con)
        cur = con.cursor()
        # Add session_id column if not exists
        cur.execute(
            'CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, message_list TEXT);'
        )
        # Migrate old table if needed
        try:
            cur.execute('ALTER TABLE messages ADD COLUMN session_id TEXT')
        except Exception:
            pass  # already exists
        # Set session_id to 'default' for old rows
        cur.execute('UPDATE messages SET session_id = ? WHERE session_id IS NULL', ('default',))
        con.commit()
        return con

    async def add_messages(self, messages: bytes, session_id: str = 'default'):
        await self._asyncify(
            self._execute,
            'INSERT INTO messages (session_id, message_list) VALUES (?, ?);',
            session_id,
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self, session_id: str = 'default') -> list[ModelMessage]:
        c = await self._asyncify(
            self._execute, 'SELECT message_list FROM messages WHERE session_id = ? ORDER BY id', session_id
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    async def get_sessions(self) -> list[dict]:
        c = await self._asyncify(
            self._execute,
            'SELECT session_id, MAX(id), MAX(message_list) FROM messages GROUP BY session_id ORDER BY MAX(id) DESC'
        )
        rows = await self._asyncify(c.fetchall)
        # Return session_id, last_message, last_id
        sessions = []
        for row in rows:
            session_id, last_id, last_message_json = row
            try:
                last_message = ModelMessagesTypeAdapter.validate_json(last_message_json)[-1]
                if isinstance(last_message, ModelRequest):
                    content = last_message.parts[0].content
                    timestamp = last_message.parts[0].timestamp.isoformat()
                elif isinstance(last_message, ModelResponse):
                    content = last_message.parts[0].content
                    timestamp = last_message.timestamp.isoformat()
                else:
                    content = ''
                    timestamp = ''
            except Exception:
                content = ''
                timestamp = ''
            sessions.append({'session_id': session_id, 'last_id': last_id, 'last_message': content, 'last_time': timestamp})
        return sessions

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(  # type: ignore
            self._executor,
            partial(func, **kwargs),
            *args,  # type: ignore
        )


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'chat_app:app',
        # app,
        reload=True,
        reload_dirs=[str(THIS_DIR)],
        host='0.0.0.0',
        port=8000,
    )
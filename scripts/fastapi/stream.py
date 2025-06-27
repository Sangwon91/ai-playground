from fastapi import FastAPI 
from fastapi.responses import StreamingResponse
from asyncio import sleep

app = FastAPI()

@app.get("/")
async def get():
    async def count():
        for i in range(10):
            yield f"Hello {i}\n"
            await sleep(1)

    return StreamingResponse(count(), media_type="text/plain")

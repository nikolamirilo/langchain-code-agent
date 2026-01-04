#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes
from main import agent

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    agent,
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)

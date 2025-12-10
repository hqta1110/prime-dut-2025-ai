from fastapi import FastAPI, Response
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import uvicorn
from typing import Any, Optional
load_dotenv()

BEARER_TOKEN = {
    "vnptai-hackathon-small": os.getenv("SMALL_BEARER_TOKEN"),
    "vnptai-hackathon-large": os.getenv("LARGE_BEARER_TOKEN"),
}

TOKEN_ID = {
    "vnptai-hackathon-small": os.getenv("SMALL_TOKEN_ID"),
    "vnptai-hackathon-large": os.getenv("LARGE_TOKEN_ID"),
}

TOKEN_KEY = {
    "vnptai-hackathon-small": os.getenv("SMALL_TOKEN_KEY"),
    "vnptai-hackathon-large": os.getenv("LARGE_TOKEN_KEY"),
}

BASE_URL = os.getenv("BASE_URL")

app = FastAPI()

class ChatRequest(BaseModel, extra="allow"):
    messages: Any
    model: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = None
    stop: Optional[Any] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    response_format: Optional[Any] = None
    seed: Optional[int] = None
    tools: Optional[Any] = None
    tool_choice: Optional[Any] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    extra_fields: dict | None = None

@app.post("/chat/completions")
async def chat(request: ChatRequest):
    model = request.model
    messages = request.messages
    temperature = request.temperature
    top_p = request.top_p
    top_k = request.top_k
    n = request.n
    stop = request.stop
    max_completion_tokens = request.max_completion_tokens
    presence_penalty = request.presence_penalty
    frequency_penalty = request.frequency_penalty
    response_format = request.response_format
    seed = request.seed
    tools = request.tools
    tool_choice = request.tool_choice
    logprobs = request.logprobs
    top_logprobs = request.top_logprobs

    kwargs = request.extra_fields if request.extra_fields else {}

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN[model]}",
        "Token-id": TOKEN_ID[model],
        "Token-key": TOKEN_KEY[model],
        "Content-Type": "application/json"
    }

    json_data = {
        "model": model.replace("-", "_"),
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "n": n,
        "stop": stop,
        "max_completion_tokens": max_completion_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "response_format": response_format,
        "seed": seed,
        "tools": tools,
        "tool_choice": tool_choice,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    res = requests.post(
        f"{BASE_URL}/v1/chat/completions/{model}",
        headers=headers,
        json=json_data
    )

    return Response(
        content=res.content,
        status_code=res.status_code,
        media_type=res.headers.get("Content-Type", "application/json")
    )

if __name__ == "__main__":
    uvicorn.run("model_api:app", host="0.0.0.0", port=2205)

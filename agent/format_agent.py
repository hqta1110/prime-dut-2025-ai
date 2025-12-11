from agno.agent import Agent
from agno.models.vllm import vLLM
from dotenv import load_dotenv
load_dotenv()
import os
from pydantic import BaseModel, Field

LLM_PORT = os.getenv("LLM_PORT")

class Anwser(BaseModel):
    key: str = Field(..., description="Đáp án cho câu trả lời 1 ký tự duy nhất (A, B, C, D, ...)")
    reason: str = Field(..., description="Giải thích lý do")

def init_format_agent():
    format_agent = Agent(
        name="Format Agent",
        # model=vLLM(
        #     id="vnptai-hackathon-large",
        #     base_url=f"http://localhost:{LLM_PORT}",
        # ),
        model=vLLM(
            id="Qwen3-32B",
            base_url=f"http://167.179.48.115:8000/v1",
            enable_thinking = False,
        ),
        role="""Định dạng văn bản đầu vào theo định dạng được cung cấp""",
        use_json_mode=True,
        response_model=Anwser,
        parser_model=vLLM(
            id="Qwen3-32B",
            base_url=f"http://167.179.48.115:8000/v1",
            enable_thinking = False,
        )
    )

    return format_agent
    
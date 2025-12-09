from agno.agent import Agent
from agno.models.vllm import VLLM
from dotenv import load_dotenv
load_dotenv()
import os

LLM_PORT = os.getenv("LLM_PORT")

def init_multi_domain_agent():
    multi_domain_agent = Agent(
        model=VLLM(
            id="vnptai-hackathon-large",
            base_url=f"http://localhost:{LLM_PORT}",
            api_key="1234"
        ),
        instructions="Trả lời câu hỏi được giao một cách ngắn gọn, chính xác và bằng tiếng Việt.",
        name="Multi-domain agent"
    )
    return multi_domain_agent
    
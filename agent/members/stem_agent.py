from agno.agent import Agent
from agno.models.vllm import VLLM
from dotenv import load_dotenv
load_dotenv()
import os

LLM_PORT = os.getenv("LLM_PORT")

def init_stem_agent():
    stem_agent = Agent(
        model=VLLM(
            id="vnptai-hackathon-large",
            base_url=f"http://localhost:{LLM_PORT}",
            api_key=""
        ),
        
    )
    return stem_agent
    
from agno.agent import Agent
from agno.models.vllm import vLLM
from dotenv import load_dotenv
load_dotenv()
import os
from tools import ReasoningTools

LLM_PORT = os.getenv("LLM_PORT")

def init_rag_agent():
    rag_agent = Agent(
        name="RAG Agent",
        # tools=[ReasoningTools()],
        model=vLLM(
            id="vnptai-hackathon-large",
            base_url=f"http://localhost:{LLM_PORT}",
            api_key="aaaa"
        ),
        
    )
    return rag_agent
    
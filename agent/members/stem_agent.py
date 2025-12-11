from agno.agent import Agent
from agno.models.vllm import vLLM
from dotenv import load_dotenv
load_dotenv()
import os
from tools import ReasoningTools

LLM_PORT = os.getenv("LLM_PORT")

def init_stem_agent():
    stem_agent = Agent(
        name="STEM Agent",
        # tools=[ReasoningTools()],
        # model=vLLM(
        #     id="vnptai-hackathon-large",
        #     base_url=f"http://localhost:{LLM_PORT}",
        # ),
        model=vLLM(
            id="Qwen3-32B",
            base_url=f"http://167.179.48.115:8000/v1",
            enable_thinking = False,
        ),
        tools=[],
        role="""Tiếp nhận và trả lời câu hỏi trắc nghiệm được cung cấp""",
        description="""
Bạn là một chuyên gia về lĩnh vực khoa học, tự nhiên, kĩ thuật.
Nhiệm vụ của bạn là phân tích và trả lời câu hỏi được cung cấp.
""",    
        instructions=f"""  
# Nội dung câu trả lời:
- Đáp án: A, B, C, ...
- Giải thích lý do.
"""
    )
    return stem_agent
    
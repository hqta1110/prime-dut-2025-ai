from agno.agent import Agent
from agno.models.vllm import vLLM
from dotenv import load_dotenv
load_dotenv()
import os
from tools import ReasoningTools

LLM_PORT = os.getenv("LLM_PORT")

def init_vietnamese_agent():
    stem_agent = Agent(
        name="Vietnamese Agent",
        # tools=[ReasoningTools()],
        model=vLLM(
            id="vnptai-hackathon-large",
            base_url=f"http://localhost:{LLM_PORT}",
        ),
        tools=[],
        # model=vLLM(
        #     id="Qwen3-32B",
        #     base_url=f"http://167.179.48.115:8000/v1",
        # ),
        role="""Tiếp nhận và trả lời câu hỏi trắc nghiệm được cung cấp""",
        description="""
Bạn là một chuyên gia về đất nước Việt Nam.
Nhiệm vụ của bạn là phân tích và trả lời câu hỏi được cung cấp.
""",    
        instructions=f"""
# Giới hạn chủ đề:
Bạn có nhiệm vụ trả lời đa đạng các lĩnh vực, bao gồm nhưng không giới hạn:
- lịch sử 
- địa lý
- văn hóa
        
# Nội dung câu trả lời:
- Đáp án: A, B, C, ...
- Giải thích lý do.
"""
    )
    return stem_agent
    
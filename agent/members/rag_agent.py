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
        ),
        tools=[],
        # model=vLLM(
        #     id="Qwen3-32B",
        #     base_url=f"http://167.179.48.115:8000/v1",
        # ),
        role="""Tiếp nhận và trả lời câu hỏi trắc nghiệm được cung cấp""",
        description="""
Bạn là một chuyên gia về trả lời câu hỏi trắc nghiệm dựa trên nội dung được cung cấp trong câu hỏi.
Nhiệm vụ của bạn là phân tích và trả lời câu hoi được cung cấp.
""",    
        instructions=f"""
# Các bước thực hiện
1. Đọc kỹ và phân tích nội dung bối cảnh được cung cấp trong câu hỏi 
2. Trả lời câu hỏi
        
# Nội dung câu trả lời:
- Đáp án: A, B, C, ...
- Giải thích lý do.
"""
    )
    return rag_agent
    
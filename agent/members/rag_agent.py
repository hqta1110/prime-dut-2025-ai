from agno.agent import Agent
from agno.models.vllm import vLLM
from dotenv import load_dotenv
load_dotenv()
import os
from tools import ReasoningTools

model = {
    "vnpt": vLLM(
        id="vnptai-hackathon-large",
        base_url=f"http://localhost:{os.getenv("LLM_PORT")}",
        temperature=0.0
    ),
    "qwen": vLLM(
        id="Qwen3-32B",
        base_url=f"http://167.179.48.115:8000/v1",
        temperature=0.0
    )
}[os.getenv("MODEL")]

def init_rag_agent():
    rag_agent = Agent(
        name="RAG Agent",
        tools=[ReasoningTools(add_instructions=True, add_few_shot=True)],
        model=model,
        reasoning_model=model,
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
    
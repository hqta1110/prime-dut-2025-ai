from agno.agent import Agent
from agno.models.vllm import vLLM
from dotenv import load_dotenv
load_dotenv()
import os
from tools import ReasoningTools, RetrievalTools

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

def init_multi_domain_agent():
    multi_domain_agent = Agent(
        name="Multi-Domain Agent",
        tools=[
            ReasoningTools(add_instructions=True, add_few_shot=True), 
            RetrievalTools(add_instructions=True, add_few_shot=True, enable_filter=False)
        ],
        model=model,
        role="""Tiếp nhận và trả lời câu hỏi trắc nghiệm được cung cấp""",
        description="""
Bạn là một chuyên gia về mọi chủ đề.
Nhiệm vụ của bạn là phân tích và trả lời câu hỏi được cung cấp.
""",    
        instructions=f"""
# Giới hạn chủ đề:
Bạn có nhiệm vụ trả lời đa đạng các lĩnh vực, bao gồm nhưng không giới hạn:
- kinh tế
- chính trị
        
# Nội dung câu trả lời:
- Đáp án: A, B, C, ...
- Giải thích lý do.
"""
    )
    return multi_domain_agent
    
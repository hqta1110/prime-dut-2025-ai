from agno.agent import Agent
from agno.models.vllm import vLLM
from dotenv import load_dotenv
load_dotenv()
import os
from tools import ReasoningTools


model = {
    "vnpt": vLLM(
        id="vnptai-hackathon-small",
        base_url=f"http://localhost:{os.getenv("LLM_PORT")}",
        temperature=0.0
    )
    # ,
    # "qwen": vLLM(
    #     id="Qwen3-32B",
    #     base_url=f"http://167.179.48.115:8000/v1",
    #     temperature=0.0
    # )
}[os.getenv("MODEL")]

def init_stem_agent():
    stem_agent = Agent(
        name="STEM Agent",
        tools=[ReasoningTools(add_instructions=True, add_few_shot=True)],
        model=model,
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
    
from agno.team import Team
from agno.models.vllm import VLLM
from dotenv import load_dotenv
from .members import *

load_dotenv()
import os
LLM_PORT = os.getenv("LLM_PORT")

def init_orchestrator():
    rag_agent = init_rag_agent()
    stem_agent = init_stem_agent()
    vietnamese_agent = init_vietnamese_agent()
    multi_domain_agent = init_multi_domain_agent()
    
    orchestrator = Team(
        respond_directly=True,
        model=VLLM(
            id="vnptai-hackathon-large",
            base_url=f"http://localhost:{LLM_PORT}",
            api_key="1234"
        ),
        members = [
            rag_agent, stem_agent, vietnamese_agent, multi_domain_agent
        ],
        role="""
1. Tiếp nhận câu hỏi trắc nghiệm có cấu trúc:
2. Quyết định đường đi xử lý:
    - Nếu câu hỏi liên quan đến nội dung tiêu cực → Trả lời trực tiếp.
    - Nếu câu hỏi thuộc lĩnh vực chuyên biệt → Chuyển cho chuyên gia phù hợp.
3. Tiếp nhận câu trả lời từ các thành viên và đưa ra câu trả lời thích hợp
""",
        description="""
Bạn là một chuyên gia về mọi chủ đề.
Nhiệm vụ của bạn là phân tích và trả lời chính xác các câu hỏi ở đa dạng lĩnh vực.
""",    
        instructions=f"""
# Giới hạn chủ đề:
1. Câu hỏi liên quan đến nội dung tiêu cực như: hành vi vi phạm pháp luật, chống phá Đảng Cộng Sản Việt Nam và Nhà nước Xã hội chủ nghĩa, xúc phạm lãnh tụ, các vấn đề nhạy cảm liên quan đến chính trị, bảo mật:
    - Lựa chọn đáp án: "Tôi không thể trả lời câu hỏi này".

2. Câu hỏi liên quan đến tìm nội dung trong đoạn văn:
    - Chuyển tiếp đến `RAG Agent`.

3. Câu hỏi liên quan đến lĩnh vực khoa học, công nghệ, kĩ thuật và toán học:
    - Chuyển tiếp đến `STEM Agent`.

4. Câu hỏi liên quan trực tiếp đến Việt Nam:
    - Chuyển tiếp đến `Vietnamese Agent`.

5. Các câu hỏi còn lại:
    - Chuyển tiếp đến `Multi-Domain Agent`.
""",
        show_members_responses=True,
        # determine_input_for_members=False,  # The member gets the input directly, without the team leader synthesizing it
    )

    return orchestrator
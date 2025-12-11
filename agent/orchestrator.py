from agno.team import Team
from agno.models.vllm import vLLM
from dotenv import load_dotenv
from .members import *
from pydantic import BaseModel, Field

load_dotenv()
import os
LLM_PORT = os.getenv("LLM_PORT")
    
def init_orchestrator():
    rag_agent = init_rag_agent()
    stem_agent = init_stem_agent()
    vietnamese_agent = init_vietnamese_agent()
    multi_domain_agent = init_multi_domain_agent()
    
    orchestrator = Team(
        mode="route",
        model=vLLM(
            id="vnptai-hackathon-large",
            base_url=f"http://localhost:{LLM_PORT}",
        ),
        # model=vLLM(
        #     id="Qwen3-32B",
        #     base_url=f"http://167.179.48.115:8000/v1",
        # ),
        show_tool_calls=True,
        members = [
            rag_agent, stem_agent, vietnamese_agent, multi_domain_agent
        ],
        tools = [],
        role="""
1. Tiếp nhận câu hỏi trắc nghiệm có cấu trúc:
2. Quyết định đường đi xử lý:
    - Nếu câu hỏi liên quan đến nội dung tiêu cực → Trả lời trực tiếp.
    - Nếu câu hỏi thuộc lĩnh vực chuyên biệt → Chuyển cho chuyên gia phù hợp.
3. Tiếp nhận câu trả lời từ các thành viên và đưa ra câu trả lời thích hợp
""",
        description="Nhiệm vụ của bạn là phân tích, chuyển tiếp câu hỏi đến các chuyên gia, và tổng hợp lại kết quả.",    
        instructions="""
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
 
# Nội dung câu trả lời:
Trả lời theo định dạng:
```
{
    "key": "<Đáp án: A/B/C/D/...>",
    "reason": "<Lý do>",
}
```

# Chú ý:
- Không tự trả lời câu hỏi nếu câu hỏi không nằm ở nhóm 1.
""",
        show_members_responses=True,
        enable_agentic_context=True,
		read_team_history=True,
		stream_member_events=True,
        stream_intermediate_steps=True,
        # use_json_mode=True,
        # response_model=Anwser,
    )

    return orchestrator
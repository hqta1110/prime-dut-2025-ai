from agno.agent import Agent
from agno.models.vllm import vLLM
from dotenv import load_dotenv
load_dotenv()
import os
from pydantic import BaseModel, Field
from enum import Enum

model = {
    "vnpt": vLLM(
        id="vnptai-hackathon-small",
        base_url=f"http://localhost:{os.getenv("LLM_PORT")}",
        temperature=0.0
    ),
    "qwen": vLLM(
        id="Qwen3-32B",
        base_url=f"http://167.179.48.115:8000/v1",
        temperature=0.0
    )
# }[os.getenv("MODEL")]
}["qwen"]

class Answer(BaseModel):
    key: str = Field(
        ...,
        pattern="^[A-Z]$",
        description="Đáp án (một ký tự từ A đến Z)"
    )

def init_format_agent():
    format_agent = Agent(
        name="Format Agent",
        model=model,
        role="""Định dạng văn bản đầu vào theo định dạng được cung cấp""",
        description="""
Bạn được cung cấp nội dung trả lời và lời giải thích cho câu hỏi trắc nghiệm.
Bạn có nhiệm vụ trích xuất thông tin và định dạng lại theo định dạng được cung cấp
""",
        instructions="""
# Nội dung cần trích xuất:
1. Khóa cho đáp án trắc nghiệm ("A", "B", "C", "D", ...).
2. Lời giải thích cho đáp án.

# Lưu ý:
Nội dung cần trích xuất nằm ở cuối nội dung được cung cấp.

# Ví dụ:
```
{
    "key": "B",
    "reason": "trong bối cảnh Việt Nam bị áp bức bởi các thế lực thực dân và đế quốc..."
}
```
""",
        use_json_mode=True,
        response_model=Answer,
        parser_model=model,
    )

    return format_agent
    
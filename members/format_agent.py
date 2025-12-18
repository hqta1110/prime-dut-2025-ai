from agents import Agent
from constants import inhouse_format, model_setting 

format = Agent(
    name="format_agent",
    model=inhouse_format,
    instructions=(
        "Bạn là trợ lý ảo, nhiệm vụ của bạn là trích xuất kí tự của đáp án câu trả lời được chọn từ trong câu trả lời gốc\n"
        "Format trả lời của bạn phải tuân theo cấu trúc sau:\n"
        "1. Phân tích câu trả lời nhận được\n"
        "3. Trích xuất ra đáp án và trả lời theo định dạng: <answer>A/B/C/D/E/...</answer>\n"
    ),
    model_settings=model_setting,
)

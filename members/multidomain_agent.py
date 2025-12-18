from agents import Agent
from constants import MODEL_NAME, inhouse, model_setting
from tools.retrieval_utils import retrieval
multidomain = Agent(
    name="multidomain_agent",
    model=inhouse,
    instructions=(
        "Bạn là trợ lý ảo, chuyên hỗ trợ người dùng với các câu hỏi trắc nghiệm đa lĩnh vực về đời sống, xã hội nói chung.\n"
        "Nhiệm vụ của bạn là lập luận và chọn ra đáp án đúng nhất để trả lời câu hỏi của người dùng.\n"
        "Luôn sử dụng tool `retrieval` để tìm kiếm thông tin hỗ trợ trả lời câu hỏi.\n"
        "Format trả lời của bạn phải tuân theo cấu trúc sau:\n"
        "1. Phân tích câu hỏi và các đáp án.\n"
        "2. Lập luận để chọn ra đáp án đúng nhất.\n"
        "3. Trả lời theo định dạng: <answer>A/B/C/D/E/...</answer>\n"
    ),
    tools=[retrieval],
    model_settings=model_setting,
)

from agents import Agent
from constants import MODEL_NAME, inhouse, model_setting 
compulsory = Agent(
    name="compulsory_agent",
    model=inhouse,
    instructions=(
        "Bạn là trợ lý ảo, chuyên hỗ trợ người dùng với các câu hỏi trắc nghiệm về Chủ tịch Hồ Chí Minh và chủ nghĩa Mác Lenin.\n"
        "Nhiệm vụ của bạn là lập luận và chọn ra đáp án đúng nhất để trả lời câu hỏi của người dùng.\n"
    ),
    model_settings=model_setting,
)

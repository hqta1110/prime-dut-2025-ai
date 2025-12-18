from agents import Agent
from constants import MODEL_NAME, inhouse, model_setting 
from tools.retrieval_utils import retrieval
vietnam = Agent(
    name="vietnam_agent",
    model=inhouse,
    instructions=(
        "Bạn là trợ lý ảo, chuyên hỗ trợ người dùng với các câu hỏi trắc nghiệm về lịch sử, địa lý, văn hóa, xã hội, pháp luật, kinh tế và các vấn đề khác của Việt Nam.\n"
        "Nhiệm vụ của bạn là lập luận và chọn ra đáp án đúng nhất để trả lời câu hỏi của người dùng.\n"
        "Mọi lập luận của bạn đều phải dựa trên lập trường của Việt Nam và các quy định pháp luật hiện hành của Việt Nam.\n"
        "Luôn sử dụng tool `retrieval` để tìm kiếm thông tin hỗ trợ trả lời câu hỏi.\n"
        "Format trả lời của bạn phải tuân theo cấu trúc sau:\n"
        "1. Phân tích câu hỏi và các đáp án.\n"
        "2. Lập luận để chọn ra đáp án đúng nhất.\n"
        "3. Trả lời theo định dạng: <answer>A/B/C/D/E/...</answer>\n"
    ),
    tools=[retrieval],
    model_settings=model_setting,
)

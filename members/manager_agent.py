from external.inhouse_model import InhouseModel
from constants import MODEL_NAME, inhouse, model_setting
import asyncio
from agents.models.interface import Model, ModelTracing
from agents.model_settings import ModelSettings

from agents import Agent, handoff, Runner

from members.compulsory_agent import compulsory

system_prompt = (
    "## Persona ##\n"
    "Bạn là trợ lý ảo, chuyên hỗ trợ người dùng với các câu hỏi trắc nghiệm thuộc nhiều lĩnh vực.\n"
    "Nhiệm vụ của bạn là phân loại các câu hỏi, từ đó đưa ra quyết định dựa trên các quy định sau:\n"
    "1. **Câu hỏi nhạy cảm, mang nội dung tiêu cực**: với các câu hỏi về các hành vi vi phạm pháp luật, chống phá Đảng Cộng Sản Việt Nam và Nhà nước Xã hội chủ nghĩa, xúc phạm lãnh tụ, các vấn đề nhạy cảm liên quan đến chính trị, bảo mật, an ninh trật tự xã hội, bạn hãy lựa chọn đáp án `Tôi không thể trả lời` hoặc tương tự.\n"
    "2. **Câu hỏi liên quan đến tìm nội dung trong đoạn văn**: với các câu hỏi yêu cầu tìm kiếm thông tin trong một đoạn văn bản cụ thể, bạn hãy chuyển tiếp câu hỏi đến `rag_agent` mà không tóm tắt hay mô tả lại câu hỏi gốc.\n"
    "3. **Câu hỏi thuộc lĩnh vực khoa học, công nghệ, kĩ thuật và toán học**: với các câu hỏi liên quan đến các lĩnh vực STEM, bạn hãy chuyển tiếp câu hỏi đến `stem_agent`.\n"
    "4. **Câu hỏi liên quan trực tiếp đến Việt Nam**: với các câu hỏi có nội dung liên quan đến lịch sử, địa lý, chính trị, văn hóa, xã hội, luật pháp, kinh tế Việt Nam, bạn hãy chuyển tiếp câu hỏi đến `vietnamese_agent`.\n"
    "5. **Các câu hỏi về chủ tịch Hồ Chí minh và chủ nghĩa Mác Lenin**: bạn hãy chuyển tiếp câu hỏi đến `compulsory_agent`.\n"
    "6. **Các câu hỏi chính trị, văn hóa, xã hội chung chung**: bạn hãy chuyển tiếp câu hỏi đến `multi_domain_agent`.\n"
    "\n"
)


async def main():
    manager = Agent(
        name="manager_agent",
        model=inhouse,
        instructions=system_prompt,
        model_settings=model_setting,
        handoffs=[compulsory]
    )
    result = await Runner.run(manager, "Nguồn gốc nào sau đây đã ảnh hưởng sâu sắc đến tư tưởng Chủ Tịch Hồ Chí Minh, góp phần hình thành con người cách mạng của Người?")
    from pprint import pprint
    pprint(result)

if __name__ == "__main__":
    asyncio.run(main())
import re
import string

# ==== IMPORT GIỐNG CODE GỐC ====
from external.inhouse_model import InhouseModel
from constants import MODEL_NAME, inhouse, model_setting
from agents import Agent, Runner

from members.compulsory_agent import compulsory
from members.stem_agent import stem
from members.rag_agent import rag
from members.multidomain_agent import multidomain
from members.vietnam_agent import vietnam


# --- SYSTEM PROMPT ---
system_prompt = (
    "## Persona ##\n"
    "Bạn là trợ lý ảo, chuyên hỗ trợ người dùng với các câu hỏi trắc nghiệm thuộc nhiều lĩnh vực.\n"
    "Nhiệm vụ của bạn là phân loại các câu hỏi, từ đó đưa ra quyết định dựa trên các quy định sau:\n"
    "1. **Câu hỏi nhạy cảm, mang nội dung tiêu cực**: với các câu hỏi về các hành vi vi phạm pháp luật, chống phá Đảng Cộng Sản Việt Nam và Nhà nước Xã hội chủ nghĩa, xúc phạm lãnh tụ, các vấn đề nhạy cảm liên quan đến chính trị, bảo mật, an ninh trật tự xã hội, bạn hãy lựa chọn đáp án nào gần với ý 'từ chối cung cấp thông tin' nhất.\n"
    "2. **Câu hỏi liên quan đến tìm nội dung trong đoạn văn**: với các câu hỏi yêu cầu tìm kiếm thông tin trong một đoạn văn bản cụ thể, bạn hãy chuyển tiếp câu hỏi đến `rag_agent` mà không tóm tắt hay mô tả lại câu hỏi gốc.\n"
    "3. **Câu hỏi thuộc lĩnh vực khoa học, công nghệ, kĩ thuật và toán học**: với các câu hỏi liên quan đến các lĩnh vực STEM, bạn hãy chuyển tiếp câu hỏi đến `stem_agent`.\n"
    "4. **Câu hỏi liên quan trực tiếp đến Việt Nam**: với các câu hỏi có nội dung liên quan đến lịch sử, địa lý, chính trị, văn hóa, xã hội, luật pháp, kinh tế Việt Nam, bạn hãy chuyển tiếp câu hỏi đến `vietnam_agent`.\n"
    "5. **Các câu hỏi về chủ tịch Hồ Chí minh và chủ nghĩa Mác Lenin**: bạn hãy chuyển tiếp câu hỏi đến `compulsory_agent`.\n"
    "6. **Các câu hỏi chính trị, văn hóa, xã hội chung chung**: bạn hãy chuyển tiếp câu hỏi đến `multi_domain_agent`.\n"
    "**Lưu ý**: Nếu trực tiếp trả lời, hãy đảm bảo format trả lời của bạn phải tuân theo cấu trúc sau:\n"
    "1. Phân tích câu hỏi và các đáp án.\n"
    "2. Lập luận để chọn ra đáp án đúng nhất.\n"
    # "3. Trả lời theo định dạng: <answer>A/B/C/D/E/...</answer>\n"
)

# --- HELPER ---
def format_question(question: str, choices: list[str]) -> str:
    labels = list(string.ascii_uppercase)
    formatted = [
        f"{labels[i]}. {choice}" for i, choice in enumerate(choices)
    ]
    return question + "\n" + "\n".join(formatted)


def extract_answer_tag(text: str) -> str | None:
    if not text:
        return None
    match = re.search(
        r"<answer>\s*([A-Z])\s*</answer>",
        text,
        re.IGNORECASE | re.DOTALL
    )
    return match.group(1).upper() if match else None


# --- SINGLE INFER FUNCTION ---
def infer_one_question(question: str, choices: list[str]) -> dict:
    """
    Infer duy nhất 1 câu hỏi, trả về:
    {
        "final_output": raw LLM output,
        "answer": extracted A/B/C/...
    }
    """

    manager_agent = Agent(
        name="manager_agent",
        model=inhouse,
        instructions=system_prompt,
        model_settings=model_setting,
        handoffs=[compulsory, stem, rag, multidomain, vietnam],
    )

    prompt_input = format_question(question, choices)

    result = Runner.run_sync(manager_agent, prompt_input)
    from rich.pretty import pprint
    pprint(result)
    final_text = (
        result.final_output
        if hasattr(result, "final_output")
        else str(result)
    )

    extracted_answer = extract_answer_tag(final_text)

    return {
        "final_output": final_text,
        "answer": extracted_answer
    }


# --- DEMO ---
if __name__ == "__main__":
    question = "Đoạn thông tin:\nTitle: Vũ khí hủy diệt hàng loạt\nContent: Vũ khí hủy diệt hàng loạt (tiếng Anh: weapon of mass destruction, gọi tắt là WMD) là loại vũ khí có khả năng gây cho đối phương tổn thất rất lớn về sinh lực, phương tiện kỹ thuật, cơ sở kinh tế, quốc phòng, môi trường sinh thái, có tác động mạnh đến tâm lý-tinh thần. Nhìn chung đó là thuật ngữ để chỉ các vũ khí hạt nhân, vũ khí sinh học, vũ khí hóa học và phóng xạ. Thuật ngữ này nảy sinh từ năm 1937 khi báo chí đề cập đến vụ ném bom thảm sát tại Guernica, Tây Ban Nha. Sau Vụ ném bom nguyên tử xuống Hiroshima và Nagasaki tại Nhật Bản, cũng như những diễn tiến suốt thời kỳ Chiến tranh Lạnh, nó ngày càng được dùng phổ biến để chỉ những thứ vũ khí phi quy ước. Đồng nghĩa với thuật ngữ WMD, người ta còn sử dụng các thuật ngữ như \"Vũ khí nguyên tử, sinh học và hóa học\" (ABC), \"Vũ khí hạt nhân, sinh học và hóa học\" (NBC) và \"Vũ khí hóa học, sinh học, phóng xạ và hạt nhân\" (CBRN) mặc dù trong số này vũ khí hạt nhân vẫn được coi là có tiềm năng lớn nhất trong hủy diệt hàng loạt. Lối nói này được sử dụng rộng rãi trong mối liên quan tới chiến tranh Iraq năm 2003 do Mỹ đứng đầu.\nCó thể nói vũ khí hạt nhân cũng như vũ khí phóng xạ là hệ quả trực tiếp của cuộc Cách mạng khoa học-kỹ thuật, cái đã giúp loài người chinh phục được nguồn năng lượng lớn chưa từng có. Vì thế chúng được xếp vào thế hệ vũ khí thứ 5, tức là còn hiện đại hơn những vũ khí tự động (liên thanh) ra đời cuối thế kỷ XIX.\nNhìn chung các vũ khí hủy diệt hàng loạt đều mang đặc tính hủy diệt lớn không lựa chọn. Do tác động hủy diệt không lựa chọn nên chính mối lo sợ vũ khí WMD đã có tác động định hình các chính sách và các hoạt động chính trị, thúc đẩy các phong trào xã hội. Thái độ ủng hộ hoạt động phát triển, hoặc ngược lại, kiểm soát vũ khí WMD là khác biệt nhau trên bình diện quốc gia cũng như quốc tế. Song nhìn chung người ta chưa hiểu rõ bản chất của những mối đe dọa này, một phần bởi vì thuật ngữ bị chính giới và giới truyền thông trên thế giới sử dụng một cách không chính xác.\nPhòng chống vũ khí hủy diệt hàng loạt là 1 nội dung được chú ý trong lĩnh vực quân sự hiện nay khi mà trình độ khoa học-công nghệ về vũ khí đã đạt đến trình độ cao.\n\nI. Lịch sử sử dụng thuật ngữ \"Vũ khí hủy diệt hàng loạt\"\n1. Nguồn gốc\n\nTrên tờ TIME (thời báo) số ra ngày 28/12/1937 lần đầu tiên thuật ngữ \"vũ khí hủy diệt hàng loạt\" được đưa ra trong các bài báo nói về vụ ném bom tại Guernica của không quân Đức:\n\nĐiều này nói về vụ ném bom san phẳng Guernica mà trong đó tới 70% thị trấn bị thiêu hủy. Lúc đó vũ khí hạt nhân chưa hề tồn tại, song vũ khí sinh học đã được nghiên cứu, còn vũ khí hóa học đã được sử dụng rỗng rãi. Năm 1946, ngay sau các vụ Hiroshima và Nagasaki, Liên Hợp Quốc đã đưa ra nghị quyết đầu tiên về vấn đề này. Đó là nghị quyết thành lập Ủy ban Năng lượng nguyên tử (tiền thân của Cơ quan Năng lượng nguyên tử Quốc tế IAEA) và dùng các từ ngữ: \"...vũ khí nguyên tử và mọi vũ khí khác có thể gây hủy diệt hàng loạt\".\nTừ đó, thuật ngữ \"vũ khí WMD\" được sử dụng rộng rãi trong cộng đồng kiểm soát vũ khí. Các thuật ngữ bộ 3 vũ khí nguyên tử, sinh học và hóa học (ABC) và sau đó là bộ 3 vũ khí hạt nhân, sinh học và hóa học (NBC) dần dần được đưa ra. Công ước về vũ khí sinh học và độc tố năm 1972 đã dứt khoát đưa các vũ khí sinh học và vũ khí hóa học vào hàng vũ khí WMD: \"Tin chắc tầm quan trọng và tính cấp bách của việc loại trừ khỏi kho vũ khí của các quốc gia, thông qua những biện pháp hữu hiệu, những vũ khí hủy diệt hàng loạt nguy hiểm như những vũ khí sử dụng các tác nhân hóa học và vi trùng\".\n\n2. Thời Chiến tranh lạnh\n\nThuật ngữ WMD đã bị sử dụng lệch lạc từ đầu Chiến tranh Lạnh (chủ yếu để chỉ vũ khí hạt nhân) cho đến tận năm 1990. Sau đó, trong Chiến tranh Vùng Vịnh 1991, nó đã được chính giới và giới truyền thông đại chúng cải tử hoàn sinh và sử dụng phổ biến, mặc dù có hơi hướng khá lỗi thời. Nó được sử dụng suốt những năm 90 khi đề cập tới yêu cầu tiếp tục trừng phạt và ngăn chặn Iraq bằng quân sự. Thuật ngữ vũ khí WMD, gộp vũ khí thuộc những phạm trù rất khác nhau thành 1 nắm (vũ khí hóa học và vũ khí sinh học khác hẳn vũ khí hạt nhân) về cơ bản là 1 cách dùng từ mang tính chính trị chứ không phải là quân sự, và có thể nhất trí rằng việc sử dụng thuật ngữ WMD trong giai đoạn 1990-2003 rõ ràng là nhằm vào các mục đích chính trị. Cách dùng từ này đạt đến đỉnh cao với cuộc khủng hoảng giải trừ vũ khí Iraq năm 2002 về sự tồn tại của vũ khí hủy diệt hàng loạt tại Iraq, cái đã trở thành lý lẽ chính của Mỹ trong cuộc chiến tranh Iraq 2003. Do sử dụng quá nhiều, hội phương ngữ Hoa Kỳ đã bầu chọn WMD là từ của năm vào năm 2002. Và trong năm 2003, đại học bang Lake Superior đã bổ sung thuật ngữ WMD vào danh mục các thuật ngữ bị tẩy chay do \"sử dụng sai, lạm dụng và nói chung là vô dụng\".\n\nII. Các loại vũ khí hủy diệt hàng loạt\n1. Vũ khí hạt nhân (tiếng Anh: nuclear weapon)\n\nVũ khí hạt nhân là loại vũ khí mà năng lượng của nó do các phản ứng phân hạch hoặc/và nhiệt hạch gây ra. 1 vũ khí hạt nhân nhỏ nhất cũng có sức công phá lớn hơn bất kỳ vũ khí thông thường nào. Loại vũ khí này có sức công phá tương đương với 10 triệu tấn thuốc nổ có thể phá hủy hoàn toàn 1 thành phố. Trong lịch sử chiến tranh có 2 quả bom hạt nhân được dùng trong Chiến tranh thế giới thứ hai: quả bom thứ nhất được ném xuống Hiroshima (Nhật Bản) vào ngày 6/8/1945 có tên là Little Boy và được làm từ urani; quả sau có tên là Fat Man và được ném xuống Nagasaki, cũng ở Nhật Bản 3 ngày sau đó, được làm từ plutoni gây ra hậu quả hủy diệt vô cùng lớn.\n\n2. Vũ khí hóa học (tiếng Anh: chemical weapon)\n\nVũ khí hóa học là loại vũ khí sử dụng hóa chất (thường là chất độc quân sự) gây tổn thương, nguy hại trực tiếp cho người, động vật và cây cỏ. Vũ khí hóa học là một trong những loại vũ khí hủy diệt lớn gây chết người hàng loạt. Vũ khí hóa học dựa trên đặc điểm độc tính cao và gây tác dụng nhanh của chất độc quân sự để gây tổn thất lớn cho đối phương.\n\n3. Vũ khí sinh học (tiếng Anh: biological weapon)\n\nVũ khí sinh học là loại vũ khí hủy diệt lớn dựa vào đặc tính gây bệnh hay truyền bệnh của các vi sinh vật như vi trùng, vi khuẩn; hoặc các độc tố do một số vi trùng tiết ra để giết hại hàng loạt.\n\n4. Vũ khí phóng xạ (tiếng Anh: radioative weapon)\n\n5. Vũ khí xung điện từ (tiếng Anh: electromagnetic pulse weapon)\n\nVũ khí xung điện từ là loại vũ khí hủy diệt, phóng xung điện từ ngắn đôi khi còn được gọi là nhiễu điện từ quá độ do con người chủ động tạo ra. Tùy thuộc vào nguồn phát mà nó có thể như là 1 bức xạ điện hoặc từ trường. Ảnh hưởng của vũ khí xung điện từ lên các thiết bị điện tử sẽ gây rối loạn và tổn hại, nếu ở mức năng lượng cao (như sét đánh) thậm chí có thể làm hư hại các công trình kiến trúc.\n\n6. Vũ khí trọng lực (tiếng Anh: Gravity weapon)\n\nVũ khí trọng lực là loại vũ khí hủy diệt có thể phát ra các sóng hấp dẫn, nếu được đặt ở trong quỹ đạo gần Trái Đất.\nCâu hỏi: Thuật ngữ 'vũ khí hủy diệt hàng loạt' lần đầu tiên được sử dụng trong bối cảnh nào?"
    choices = [
      "Trong vụ ném bom Hiroshima và Nagasaki",
      "Trong báo chí năm 1937 đề cập đến vụ ném bom Guernica",
      "Trong Công ước Vũ khí Sinh học năm 1972",
      "Trong Chiến tranh Vùng Vịnh năm 1991"
    ]

    result = infer_one_question(question, choices)

    print("=== RAW OUTPUT ===")
    print(result["final_output"])
    print("\n=== EXTRACTED ANSWER ===")
    print(result["answer"])

from typing import Optional, Literal
from textwrap import dedent
from agents import Agent, FunctionTool, RunContextWrapper, function_tool

@function_tool
def reasoning(
    step_type: Literal["think", "analyze"],
    title: str,
    content: str,
    next_action: Literal["continue", "validate", "final_answer"] = "continue",
    confidence: float = 0.8,
) -> str:
    """
    Ghi lại một bước suy luận (Reasoning Step) vào bộ nhớ hội thoại. 
    BẮT BUỘC dùng tool này để suy nghĩ trước khi hành động hoặc kết luận.

    CÁCH DÙNG:
    1. step_type="think": Dùng ĐẦU TIÊN để lập kế hoạch hoặc nháp suy nghĩ.
       - content: Mô tả chi tiết suy nghĩ, lập luận nội tâm.
    
    2. step_type="analyze": Dùng SAU KHI gọi tool khác để đánh giá kết quả.
       - content: Phân tích kết quả vừa nhận được (đúng/sai/đủ/thiếu).
       - next_action: 
            + 'continue': Cần suy nghĩ hoặc tìm kiếm thêm.
            + 'final_answer': Đã đủ thông tin để trả lời người dùng.

    Args:
        step_type: Loại bước ('think' hoặc 'analyze').
        title: Tiêu đề ngắn gọn cho bước này.
        content: Nội dung chi tiết của suy nghĩ hoặc phân tích.
        next_action: Hành động tiếp theo ('continue', 'validate', 'final_answer').
        confidence: Độ tự tin (0.0 - 1.0).

    Returns:
        Một chuỗi văn bản đã định dạng để Agent tự đọc lại trong lịch sử.
    """
    
    # Chuẩn hóa icon và header để LLM dễ nhận diện trong lịch sử
    if step_type == "think":
        header = "🧠 THOUGHT (Suy nghĩ)"
    else:
        header = "🔍 ANALYSIS (Phân tích)"

    # Format kết quả trả về dưới dạng Markdown rõ ràng
    # LLM sẽ nhìn thấy cái này trong phần "Tool Output" của lịch sử chat
    output = dedent(f"""
    === {header} ===
    📌 Tiêu đề: {title}
    📝 Nội dung: {content}
    🎯 Hướng tiếp theo: {next_action.upper()}
    ⚖️ Độ tin cậy: {confidence}
    =========================
    """)
    
    return output.strip()
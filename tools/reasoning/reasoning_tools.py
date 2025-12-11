from textwrap import dedent
from typing import Any, Dict, List, Optional

from .step import NextAction, ReasoningStep
from agno.tools import Toolkit
from agno.utils.log import log_debug, log_error

class ReasoningTools(Toolkit):
    def __init__(
        self,
        enable_think: bool = True,
        enable_analyze: bool = True,
        all: bool = False,
        instructions: Optional[str] = None,
        add_instructions: bool = False,
        add_few_shot: bool = False,
        few_shot_examples: Optional[str] = None,
        **kwargs,
    ):
        """A toolkit that provides step-by-step reasoning tools: Think and Analyze."""

        if instructions is None:
            self.instructions = "<reasoning_instructions>\n" + self.DEFAULT_INSTRUCTIONS
            if add_few_shot:
                if few_shot_examples is not None:
                    self.instructions += "\n" + few_shot_examples
                else:
                    self.instructions += "\n" + self.FEW_SHOT_EXAMPLES
            self.instructions += "\n</reasoning_instructions>\n"
        else:
            self.instructions = instructions

        tools: List[Any] = []
        # Prefer new flags; fallback to legacy ones
        if all or enable_think:
            tools.append(self.think)
        if all or enable_analyze:
            tools.append(self.analyze)

        super().__init__(
            name="reasoning_tools",
            instructions=self.instructions,
            add_instructions=add_instructions,
            tools=tools,
            **kwargs,
        )

    def think(
        self,
        session_state: Dict[str, Any],
        title: str,
        thought: str,
        action: Optional[str] = None,
        confidence: float = 0.8,
    ) -> str:
        """Use this tool as a scratchpad to reason about the question and work through it step-by-step.
        This tool will help you break down complex problems into logical steps and track the reasoning process.
        You can call it as many times as needed. These internal thoughts are never revealed to the user.

        Args:
            title: A concise title for this step
            thought: Your detailed thought for this step
            action: What you'll do based on this thought
            confidence: How confident you are about this thought (0.0 to 1.0)

        Returns:
            A list of previous thoughts and the new thought
        """
        try:
            log_debug(f"Thought about {title}")

            # Create a reasoning step
            reasoning_step = ReasoningStep(
                title=title,
                reasoning=thought,
                action=action,
                next_action=NextAction.CONTINUE,
                confidence=confidence,
            )

            current_run_id = session_state.get("current_run_id", None)

            # Add this step to the Agent's session state
            if session_state is None:
                session_state = {}
            if "reasoning_steps" not in session_state:
                session_state["reasoning_steps"] = {}
            if current_run_id not in session_state["reasoning_steps"]:
                session_state["reasoning_steps"][current_run_id] = []
            session_state["reasoning_steps"][current_run_id].append(reasoning_step.model_dump_json())

            # Return all previous reasoning_steps and the new reasoning_step
            if "reasoning_steps" in session_state and current_run_id in session_state["reasoning_steps"]:
                formatted_reasoning_steps = ""
                for i, step in enumerate(session_state["reasoning_steps"][current_run_id], 1):
                    step_parsed = ReasoningStep.model_validate_json(step)
                    step_str = dedent(f"""\
Bước {i}:
Tiêu đề: {step_parsed.title}
Diễn giải: {step_parsed.reasoning}
Hành động: {step_parsed.action}
Độ tin cậy: {step_parsed.confidence}
""")
                    formatted_reasoning_steps += step_str + "\n"
                return formatted_reasoning_steps.strip()
            return reasoning_step.model_dump_json()
        except Exception as e:
            log_error(f"Error recording thought: {e}")
            return f"Lỗi ghi lại suy nghĩ: {e}"

    def analyze(
        self,
        session_state: Dict[str, Any],
        title: str,
        result: str,
        analysis: str,
        next_action: str = "continue",
        confidence: float = 0.8,
    ) -> str:
        """Use this tool to analyze results from a reasoning step and determine next actions.

        Args:
            title: A concise title for this analysis step
            result: The outcome of the previous action
            analysis: Your analysis of the results
            next_action: What to do next ("continue", "validate", or "final_answer")
            confidence: How confident you are in this analysis (0.0 to 1.0)

        Returns:
            A list of previous thoughts and the new analysis
        """
        try:
            log_debug(f"Analyzed {title}")

            # Map string next_action to enum
            next_action_enum = NextAction.CONTINUE
            if next_action.lower() == "validate":
                next_action_enum = NextAction.VALIDATE
            elif next_action.lower() in ["final", "final_answer", "finalize"]:
                next_action_enum = NextAction.FINAL_ANSWER

            # Create a reasoning step for the analysis
            reasoning_step = ReasoningStep(
                title=title,
                result=result,
                reasoning=analysis,
                next_action=next_action_enum,
                confidence=confidence,
            )

            current_run_id = session_state.get("current_run_id", None)
            # Add this step to the Agent's session state
            if session_state is None:
                session_state = {}
            if "reasoning_steps" not in session_state:
                session_state["reasoning_steps"] = {}
            if current_run_id not in session_state["reasoning_steps"]:
                session_state["reasoning_steps"][current_run_id] = []
            session_state["reasoning_steps"][current_run_id].append(reasoning_step.model_dump_json())

            # Return all previous reasoning_steps and the new reasoning_step
            if "reasoning_steps" in session_state and current_run_id in session_state["reasoning_steps"]:
                formatted_reasoning_steps = ""
                for i, step in enumerate(session_state["reasoning_steps"][current_run_id], 1):
                    step_parsed = ReasoningStep.model_validate_json(step)
                    step_str = dedent(f"""\
Bước {i}:
Tiêu đề: {step_parsed.title}
Diễn giải: {step_parsed.reasoning}
Hành động: {step_parsed.action}
Độ tin cậy: {step_parsed.confidence}
""")
                    formatted_reasoning_steps += step_str + "\n"
                return formatted_reasoning_steps.strip()
            return reasoning_step.model_dump_json()
        except Exception as e:
            log_error(f"Error recording analysis: {e}")
            return f"Lỗi ghi lại phân tích: {e}"

    # --------------------------------------------------------------------------------
    # Default instructions and few-shot examples
    # --------------------------------------------------------------------------------

    DEFAULT_INSTRUCTIONS = dedent(
        """\
        Bạn có quyền sử dụng các công cụ `think` và `analyze` để xử lý vấn đề  theo từng bước và cấu trúc hóa quá trình suy nghĩ của mình. Bạn phải LUÔN LUÔN dùng `think` trước khi gọi công cụ hoặc tạo phản hồi.

        1. **Think** (bảng nháp):
            - Mục đích: Dùng công cụ `think` như một bảng nháp để phân tích các vấn đề phức tạp, phác thảo các bước và quyết định hành động ngay lập tức trong luồn suy nghĩ của bạn. Dùng nó cho cấu trúc độc thoại nội tâm.            
            - Cách dùng: Gọi `think` trước khi gọi công cụ khác hoặc tạo phản hồi. Giải thích lý do và chỉ định hành động dự định (ví dụ: "gọi tool", "thực hiện tính toán", "hỏi câu làm rõ").

        2. **Analyze** (đánh giá):
            - Mục đích: Đánh giá kết quả của một bước think hoặc một chuỗi tool calls. Xác định xem kết quả có đúng, đủ hoặc cần điều tra thêm hay không.
            - Cách dùng: Gọi `analyze` sau khi thực hiện các tool calls. Xác định `next_action` dựa trên phân tích:
                - `continue`: cần suy luận thêm.
                - `validate`: cần xác nhận/đối chiếu bên ngoài nếu có thể.
                - `final_answer`: đã đủ để kết luận
            - Giải thích lập luận, nêu rõ lý do đúng/đủ.
                
        ## Các hướng dẫn quan trọng
        - **Luôn Think trước:** Bạn PHẢI sử dụng công cụ `think` trước khi gọi tool hoặc tạo phản hồi.
        - **Lặp lại để giải quyết:** Sử dụng `think` và `analyze` lặp vòng để xây dựng lộ trình suy luận rõ ràng. Quy trình chuẩn là: Think → (Tool Calls nếu cần) → Analyze (nếu cần) → … → final_answer. Lặp lại chu trình cho đến khi đạt kết luận thỏa đáng.
        - **Có thể gọi nhiều tool song song:** Sau một bước `think`, bạn có thể gọi nhiều tool cùng lúc.
        - **Giữ suy nghĩ ở cục bộ:** Các bước suy nghĩ (thoughts và analyses) chỉ dành cho nội bộ, **không chia sẻ trực tiếp với người dùng**.
        - **Kết luận rõ ràng:** Khi phân tích chỉ ra `next_action` là `final_answer`, hãy đưa ra câu trả lời ngắn gọn và chính xác."""
    )

    FEW_SHOT_EXAMPLES = dedent(
        """
        Dưới đây là các ví dụ minh họa cách sử dụng công cụ `think` và `analyze`

        ### Ví dụ

        **Ví dụ 1: Truy xuất thông tin đơn giản**

        *Yêu cầu của người dùng:* Trên Trái Đất có bao nhiêu châu lục?

        *Quy trình nội bộ của Agent:*

        ```tool_call
        think(
          title="Hiểu yêu cầu",
          thought="Người dùng muốn biết số lượng châu lục tiêu chuẩn trên Trái Đất. Đây là một kiến thức phổ biến.",
          action="Ghi nhớ hoặc xác minh số lượng châu lục.",
          confidence=0.95
        )
        ```
        *--(Agent ghi nhớ thông tin nội bộ)--*
        ```tool_call
        analyze(
          title="Đánh giá sự thật",
          result="Các mô hình địa lý tiêu chuẩn liệt kê 7 châu lục: Châu Phi, Châu Nam Cực, Châu Á, Châu Úc, Châu Âu, Bắc Mỹ, Nam Mỹ.",
          analysis="Thông tin được nhớ lại trả lời trực tiếp và chính xác câu hỏi của người dùng.",
          next_action="final_answer",
          confidence=1.0
        )
        ```

        *Câu trả lời cuối cùng của Agent cho người dùng:*
        Trên Trái Đất có 7 châu lục: Châu Phi, Châu Nam Cực, Châu Á, Châu Úc, Châu Âu, Bắc Mỹ và Nam Mỹ.

        **Ví dụ 2: Thu thập thông tin nhiều bước**

        *Yêu cầu của người dùng:* Thủ đô của Pháp là gì và dân số hiện tại của nó là bao nhiêu?

        *Quy trình nội bộ của Agent:*

        ```tool_call
        think(
          title="Lập kế hoạch tìm kiếm thông tin",
          thought="TNgười dùng cần hai thông tin: thủ đô của Pháp và dân số hiện tại. Tôi nên dùng công cụ bên ngoài (như search) để lấy dữ liệu mới nhất và chính xác nhất.",
          action="Trước tiên, tìm kiếm thủ đô của Pháp.",
          confidence=0.95
        )
        ```

        *Thực hiện nhiều lời gọi công cụ song song*
        *--(Tool call 1: search(query="thủ đô của nước Pháp"))--*
        *--(Tool call 2: search(query="Dân số hiện tại của nước Pháp"))--*
        *--(Tool Result 1: "Paris")--*
        *--(Tool Result 2: "Khoảng 2,1 triệu người (nội đô, ước tính đầu năm 2024)")--*

        ```tool_call
        analyze(
          title="Phân tích kết quả tìm kiếm thủ đô",
          result="Kết quả tìm kiếm cho thấy Paris là thủ đô của Pháp.",
          analysis="Điều này cung cấp phần thông tin đầu tiên người dùng yêu cầu. Giờ tôi cần tìm dân số của Paris.",
          next_action="continue",
          confidence=1.0
        )
        ```
        ```tool_call
        analyze(
          title="Phân tích kết quả tìm kiếm dân số",
          result="Tìm kiếm đã cung cấp số liệu ước tính dân số của Paris.",
          analysis="Tôi đã có cả thủ đô và dân số ước tính. Tôi có thể đưa ra câu trả lời cuối cùng.",
          next_action="final_answer",
          confidence=0.9
        )
        ```

        *Câu trả lời cuối cùng của Agent cho người dùng:*
        Thủ đô của Pháp là Paris. Dân số ước tính (khu vực nội đô) khoảng 2,1 triệu người tính đến đầu năm 2024."""
    )
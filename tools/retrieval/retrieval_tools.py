from textwrap import dedent
from typing import Any, Dict, List, Optional, Literal

from agno.tools import Toolkit
from agno.utils.log import log_debug, log_error
from .embedding_utils import *

class RetrievalTools(Toolkit):
    def __init__(
        self,
        enable_hybrid=True,
        enable_filter=True,
        distance_metric="cosine",
        k=10,
        instructions: Optional[str] = None,
        add_instructions: bool = False,
        few_shot_examples: Optional[str] = None,
        add_few_shot: bool = False,
        **kwargs,
    ):
        self.enable_hybrid = enable_hybrid
        self.enable_filter = enable_filter
        self.distance_metric = distance_metric
        self.k = k

        if instructions is None:
            self.instructions = "<retrieval_instructions>\n" + self.DEFAULT_INSTRUCTIONS
            if add_few_shot:
                if few_shot_examples is not None:
                    self.instructions += "\n" + few_shot_examples
                else:
                    self.instructions += "\n" + self.FEW_SHOT_EXAMPLES
                self.instructions += "\n</retrieval_instructions>\n"
            else:
                self.instructions = instructions

        super().__init__(
            name="retrieval_tools",
            instructions=self.instructions,
            add_instructions=add_instructions,
            tools=[self.retrieval],
            **kwargs,
        )

    def retrieval(self, query: str, fields: List[Literal[
        "circular", "constitution", "culture", "decree", 
        "geography" "history", "law", "philosophy", "regulation", 
        "others"]]):
        try:
            log_debug(f"Retrieval for {query}")

            knowledge = load_knowledge()

            if self.enable_filter and fields:
                field_key = "_".join(sorted(fields))
                data = [
                    x for x in knowledge
                    if any(d in fields for d in x["fields"])
                ]
            else:
                field_key = "all"
                data = knowledge

            query_emb = get_embedding(query)

            # TODO hybrid
            if self.enable_hybrid:
                pass

            index = get_ivf_index(field_key, data, metric=self.distance_metric)

            result = vector_search(
                index=index,
                data=data,
                query_emb=query_emb,
                k=self.k,
                metric=self.distance_metric
            )

            return [r['item']['text'] for r in result]

        except Exception as e:
            log_error(f"Retrieval error: {e}")
            return f"Lỗi retrieval: {e}"

    # --------------------------------------------------------------------------------
    # Default instructions and few-shot examples
    # --------------------------------------------------------------------------------

    DEFAULT_INSTRUCTIONS = dedent(
        """\
        Bạn có quyền sử dụng công cụ `retrieval` để tìm kiếm thông tin. 
        Cách dùng: `retrieval` để tìm thông tin dựa trên nội dung cần truy vấn `query` và các lĩnh vực `fields` liên quan đến nội dung truy vấn."""
    )

    FEW_SHOT_EXAMPLES = dedent(
        """
        Dưới đây là các ví dụ minh họa cách sử dụng công cụ `retrieval`.

        ### Ví dụ

        **Ví dụ 1: Truy xuất thông tin đơn giản**

        *Yêu cầu của người dùng:* Việt Nam có khí hậu gì?

        *Quy trình nội bộ của Agent:*

        ```tool_call
        retrieval(
          query="Khí hậu ở Việt Nam",
          fields=["geography"]
        )
        ```

        *Câu trả lời cuối cùng của Agent cho người dùng:*
        Việt Nam có khí hậu nhiệt đới ẩm gió mùa.
        
        **Ví dụ 2: Truy xuất thông tin đa lĩnh vực**

        *Yêu cầu của người dùng:* Phân tích tác động của chính sách Đổi Mới năm 1986 đối với tốc độ tăng trưởng GDP Việt Nam thập niên 1990, có so sánh với các nền kinh tế chuyển đổi ở Đông Âu.

        *Quy trình nội bộ của Agent:*

        ```tool_call
        retrieval(
          query="Các cải cách kinh tế - chính trị cốt lõi của chính sách Đổi Mới năm 1986",
          fields=["history", "politics", "economy"]
        )
        ```

        ```tool_call
        retrieval(
          query="Tốc độ tăng trưởng GDP Việt Nam trong thập niên 1990",
          fields=["economy"]
        )
        ```

        ```tool_call
        retrieval(
          query="Quá trình chuyển đổi kinh tế và tốc độ tăng trưởng GDP của Việt Nam so với các nước Đông Âu",
          fields=["history", "economy"]
        )
        ```

        *Câu trả lời cuối cùng của Agent cho người dùng:*
        Đổi Mới 1986 mở cửa kinh tế, cho phép tư nhân, tự do hóa thương mại và cải cách quản lý, giúp Việt Nam thoát khủng hoảng và tăng trưởng GDP ổn định 7–9% trong thập niên 1990.
        Trong khi đó, nhiều nước Đông Âu áp dụng “liệu pháp sốc”, dẫn đến suy giảm kinh tế mạnh đầu giai đoạn chuyển đổi."""
    )
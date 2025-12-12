from agent import *
from pydantic import BaseModel
from typing import List
import time

agent = init_orchestrator()
format_agent = init_format_agent()

class MultipleChoiceQuestion(BaseModel):
    question: str
    choices: List[str]

start = time.time()
answer = agent.run(
    message=MultipleChoiceQuestion(
        question="Chủ Tịch Hồ Chí Minh đã dùng hình tượng nào dưới đây để chỉ chủ nghĩa tư bản?",
        choices = [
            "Con bạch tuộc",
            "Con đỉa hai vòi",
            "Con chim đại bàng.",
            "Con chim cánh cụt"
        ]
    )
).content
formatted_answer = format_agent.run(answer).to_dict()['content']
print(formatted_answer)
print(time.time() - start)
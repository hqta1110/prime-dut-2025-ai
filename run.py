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
        question="Điện trở tương đương khi hai điện trở, R1 và R2, được mắc song song là gì?",
        choices = [
            "R1 + R2",
            "R1 - R2",
            "(R1 * R2) / (R1 + R2)",
            "(R1 + R2) / (R1 * R2)",
            "R1 * R2",
            "R1 / R2",
            "R2 / R1",
            "1 / (R1 + R2)",
            "1 / (R1 * R2)",
            "(R1 + R2) / 2"
        ]
    )
).content

formatted_answer = format_agent.run(answer).to_dict()['content']
print(formatted_answer)
print(time.time() - start)
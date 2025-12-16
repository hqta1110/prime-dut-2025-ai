from agent import *
from pydantic import BaseModel
from typing import List
import json
import random

agent = init_orchestrator()
format_agent = init_format_agent()
number_of_answers = 1

answers = []

class MultipleChoiceQuestion(BaseModel):
    question: str
    choices: List[str]

with open("data/question_samples/val.json", mode="r", encoding="utf-8") as f:
    questions = json.load(f)
    if number_of_answers >= 0:
        questions = random.sample(questions, number_of_answers)

for question in questions:
    answer = agent.run(
        message=MultipleChoiceQuestion(
            question = question['question'],
            choices = question['choices']
        )
    ).content
    formatted_answer = format_agent.run(answer).to_dict()['content']

    answer = {
        "qid": question['qid'],
        
    }
    print(formatted_answer)


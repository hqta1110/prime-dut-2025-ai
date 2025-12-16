from agent import *
from pydantic import BaseModel
from typing import List
import json
import random
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio

agent = init_orchestrator()
format_agent = init_format_agent()

number_of_answers = -1
concurrent = 5
question_path = "data/question_samples/test.json"

semaphore = asyncio.Semaphore(concurrent)

class MultipleChoiceQuestion(BaseModel):
    question: str
    choices: List[str]

async def process_one_question(question: dict):
    async with semaphore:
        result = await agent.arun(
            message=MultipleChoiceQuestion(
                question=question["question"],
                choices=question["choices"]
            )
        )
        answer_text = result.content

        formatted_result = await format_agent.arun(answer_text)
        formatted_answer = formatted_result.to_dict()["content"]

        answer = {
            "qid": question["qid"],
            "answer": formatted_answer["key"],
        }

        if question.get("answer"):
            answer["isCorrect"] = formatted_answer["key"] == question["answer"]

        return answer

async def main():
    with open(question_path, mode="r", encoding="utf-8") as f:
        questions = json.load(f)

    if number_of_answers >= 0:
        questions = random.sample(questions, number_of_answers)

    tasks = [
        process_one_question(question)
        for question in questions
    ]

    answers = await tqdm_asyncio.gather(
        *tasks,
        desc=f"Processing questions (concurrent={concurrent})"
    )

    answer_df = pd.DataFrame(answers)
    answer_df.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())

from agent import *
from pydantic import BaseModel
from typing import List
import json
import random
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
import csv
import os
from agent.context import SESSION_STATE
agent = init_orchestrator()
format_agent = init_format_agent()

number_of_answers = -1
concurrent = 5
question_path = "data/question_samples/test.json"
output_file = "submission_test.csv"

semaphore = asyncio.Semaphore(concurrent)
file_write_lock = asyncio.Lock()

class MultipleChoiceQuestion(BaseModel):
    question: str
    choices: List[str]

def get_processed_qids(file_path):
    if not os.path.exists(file_path):
        return set()
    
    processed = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and 'qid' in header:
                qid_idx = header.index('qid')
                for row in reader:
                    if row:
                        processed.add(row[qid_idx])
            else:
                 pass
    except Exception as e:
        print(f"Cannot read previous file: {e}")
    return processed

async def append_result_to_csv(file_path, data_dict):
    file_exists = os.path.exists(file_path)
    
    async with file_write_lock:
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_dict)

async def process_one_question(question: dict):
    qid = question["qid"]

    async with semaphore:
        while True:
            try:
                session_state = {
                    "session_id": qid,
                    "current_run_id": qid,
                    "reasoning_steps": {}
                }
                token = SESSION_STATE.set(session_state)
                try:
                    result = await agent.arun(
                        message=MultipleChoiceQuestion(
                            question=question["question"],
                            choices=question["choices"]
                        )
                    )
                finally:
                    SESSION_STATE.reset(token)
                answer_text = result.content
                print("------ RAW ANSWER ------")
                print(f"Raw answer for QID {qid}: {answer_text}")
                print("------------------------")


                formatted_result = await format_agent.arun(answer_text)
                # formatted_answer = formatted_result.to_dict()["content"]
                formatted_dict = formatted_result.to_dict()
                content = formatted_dict.get("content", {})

                if not isinstance(content, dict) or "key" not in content:
                    raise ValueError(f"Invalid formatted answer: {content}")

                raw_key = str(content["key"]).strip().lower()
                answer_key = None

                # ===== REFUSAL HANDLING =====
                if raw_key in ["", "n/a", "na", "None", "none", "null", "unknown"]:
                    for idx, choice in enumerate(question["choices"]):
                        if "không thể" in choice.lower() or "không cung cấp" in choice.lower():
                            answer_key = chr(ord("A") + idx)
                            break

                # Case 1: already a letter
                if len(raw_key) == 1 and raw_key.isalpha():
                    answer_key = raw_key.upper()

                # Case 2: semantic match (QUAN TRỌNG NHẤT)
                else:
                    for idx, choice in enumerate(question["choices"]):
                        if not choice:
                            continue
                        choice_l = choice.strip().lower()
                        if raw_key == choice_l:
                            answer_key = chr(ord("A") + idx)
                            break
                        if raw_key in choice_l:
                            answer_key = chr(ord("A") + idx)
                            break

                # Case 3: numeric index (CHỈ KHI TRÊN THẤT BẠI)
                if answer_key is None and raw_key.isdigit():
                    idx = int(raw_key)
                    if 0 <= idx < len(question["choices"]):
                        answer_key = chr(ord("A") + idx)


                print("------ DEBUG ANSWER ------")
                print("QID:", question["qid"])
                print("Raw answer_text:", answer_text)
                print("Formatted content:", content)
                print("Final answer_key:", answer_key)
                print("--------------------------")

                # Final guard
                if answer_key is None:
                    raise ValueError(f"Unrecognized answer key: {raw_key}")

                answer = {
                    "qid": question["qid"],
                    "answer": answer_key,
                }


                # if question.get("answer"):
                #     answer["isCorrect"] = formatted_answer["key"] == question["answer"]

                await append_result_to_csv(output_file, answer)
            
                return answer
            except Exception as e:
                error_msg = str(e).lower()
                wait_time = 10   # ⭐ default

                if "rate limit" in error_msg or "401" in error_msg:
                    wait_time = 60
                    print(f"Rate limit at {qid}. Waiting {wait_time}s...")
                else:
                    print(f"Error at {qid}: {e}. Retrying in {wait_time}s...")

                await asyncio.sleep(wait_time)
                continue

async def main():
    with open(question_path, mode="r", encoding="utf-8") as f:
        questions = json.load(f)

    if number_of_answers >= 0:
        questions = random.sample(questions, number_of_answers)

    processed_qids = get_processed_qids(output_file)
    print(f"Number of processed qids: {len(processed_qids)}")

    pending_questions = [q for q in questions if q["qid"] not in processed_qids]
    print(f"Number of pending questions: {len(pending_questions)}")
     
    if not pending_questions:
        print("All questions have been processed. Exiting...")
        return
    
    tasks = [
        process_one_question(question)
        for question in pending_questions
    ]

    await tqdm_asyncio.gather(
        *tasks,
        desc=f"Processing questions (concurrent={concurrent})"
    )

    # answer_df = pd.DataFrame(answers)
    # answer_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import json
import os
import csv
from tqdm import tqdm

from agent import *

INPUT_FILE = 'data/val.json'
OUTPUT_JSON_FILE = 'submission_val.json'
OUTPUT_CSV_FILE = 'submission_val.csv'

MAX_CONCURRENCY = 5

print("Initialize Agent...")
agent = init_orchestrator()
format_agent = init_format_agent()
print("Agent initialization completed")


def load_processed_data(file_path):
    if not os.path.exists(file_path):
        return [], set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            processed_ids = {item.get('qid') for item in data}
            return data, processed_ids
    except json.JSONDecodeError:
        return [], set()


def append_result_to_csv(output_file, qid, answer):
    file_exists = os.path.exists(output_file)

    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['qid', 'answer'])

        writer.writerow([qid, answer])
        f.flush()


async def run_sync(func, *args):
    """Chạy hàm sync trong thread pool để sử dụng await."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


async def process_item(
    item,
    agent,
    format_agent,
    results,
    processed_ids,
    csv_lock,
    json_lock,
):
    qid = item.get('qid')
    if qid in processed_ids:
        return

    question = item.get('question')
    choices = item.get('choices', [])

    prompt_choices = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    user_input = (
        f"{question}\n\nCác lựa chọn:\n{prompt_choices}\n\n"
        "Hãy chọn 1 đáp án đúng nhất."
    )

    final_answer = "A"

    try:
        response = await run_sync(agent.run, user_input)
        raw_answer = response.content

        formatted = await run_sync(format_agent.run, raw_answer)

        content = formatted.to_dict()['content']

        # convert content depending on format
        if isinstance(content, dict) and "key" in content:
            final_answer = content["key"]
        else:
            final_answer = str(content)

    except Exception as e:
        print(f"\nError at {qid}: {e}")
        final_answer = "Error"

    # ---- Ghi JSON ----
    async with json_lock:
        results.append({"qid": qid, "answer": final_answer})
        processed_ids.add(qid)

        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    # ---- Ghi CSV ----
    async with csv_lock:
        append_result_to_csv(OUTPUT_CSV_FILE, qid, final_answer)


async def main_async():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Cannot find {INPUT_FILE}")
        return

    results, processed_ids = load_processed_data(OUTPUT_JSON_FILE)

    print(f"Total question: {len(input_data)}")
    print(f"Completed previously: {len(processed_ids)}")

    csv_lock = asyncio.Lock()
    json_lock = asyncio.Lock()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = []

    for item in input_data:
        if item["qid"] in processed_ids:
            continue

        async def wrapped(it=item):
            async with sem:
                await process_item(
                    it, agent, format_agent, results, processed_ids,
                    csv_lock, json_lock
                )

        tasks.append(wrapped())

    # tqdm chạy khi từng task hoàn thành:
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        await fut

    print(f"\nCompleted! Output saved to {OUTPUT_JSON_FILE}")


if __name__ == "__main__":
    asyncio.run(main_async())

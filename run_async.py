import asyncio
import json
import os
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Set
from tqdm import tqdm

from agent import *

INPUT_FILE = 'data/val.json'
OUTPUT_JSON_FILE = 'submission_val.json'
OUTPUT_CSV_FILE = 'submission_val.csv'

MAX_CONCURRENCY = 5
MAX_RETRIES = 3


@dataclass
class AgentBundle:
    orchestrator: Any
    formatter: Any


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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


async def safe_agent_run(bundle: AgentBundle, user_input: str):
    """
    Gọi orchestrator.run với retry + re-init nếu bị lỗi race kiểu NoneType.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await run_sync(bundle.orchestrator.run, user_input)
        except Exception as e:
            last_exc = e
            msg = str(e)
            # tuỳ bạn refine điều kiện này
            is_race = "add_member_run" in msg or "NoneType" in msg
            if not is_race or attempt == MAX_RETRIES:
                # không phải lỗi race, hoặc retry hết số lần
                raise
            print(f"[WARN] agent error ({e}), re-init and retry {attempt}/{MAX_RETRIES}...")
            # re-init bundle tại chỗ
            bundle.orchestrator = init_orchestrator()
            await asyncio.sleep(0.5)
    # nếu tới đây vẫn fail thì ném lỗi cuối cùng ra
    raise last_exc


async def safe_format_run(bundle: AgentBundle, raw_answer: str):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await run_sync(bundle.formatter.run, raw_answer)
        except Exception as e:
            last_exc = e
            msg = str(e)
            is_race = "add_member_run" in msg or "NoneType" in msg
            if not is_race or attempt == MAX_RETRIES:
                raise
            print(f"[WARN] format_agent error ({e}), re-init and retry {attempt}/{MAX_RETRIES}...")
            bundle.formatter = init_format_agent()
            await asyncio.sleep(0.5)
    raise last_exc


async def process_item(
    item: Dict[str, Any],
    agent_pool: "asyncio.Queue[AgentBundle]",
    results: List[Dict[str, Any]],
    processed_ids: Set[str],
    csv_lock: asyncio.Lock,
    json_lock: asyncio.Lock,
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

    # lấy 1 bundle agent từ pool
    bundle = await agent_pool.get()
    try:
        final_answer = "A"
        try:
            response = await safe_agent_run(bundle, user_input)
            raw_answer = response.content
            formatted = await safe_format_run(bundle, raw_answer)
            content = formatted.to_dict()['content']

            if isinstance(content, dict) and "key" in content:
                final_answer = content["key"]
            else:
                final_answer = str(content)
        except Exception as e:
            print(f"\nError at {qid}: {e}")
            final_answer = "Error"

        async with json_lock:
            results.append({"qid": qid, "answer": final_answer})
            processed_ids.add(qid)
            with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        async with csv_lock:
            append_result_to_csv(OUTPUT_CSV_FILE, qid, final_answer)

    finally:
        # luôn trả bundle về pool, kể cả khi lỗi
        await agent_pool.put(bundle)


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

    # --- khởi tạo pool agent ---
    agent_pool: asyncio.Queue[AgentBundle] = asyncio.Queue()
    print("Initialize Agent pool...")
    for _ in range(MAX_CONCURRENCY):
        orchestrator = init_orchestrator()
        formatter = init_format_agent()
        await agent_pool.put(AgentBundle(orchestrator, formatter))
    print("Agent pool ready")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = []

    for item in input_data:
        if item["qid"] in processed_ids:
            continue

        async def wrapped(it=item):
            async with sem:
                await process_item(
                    it,
                    agent_pool,
                    results,
                    processed_ids,
                    csv_lock,
                    json_lock,
                )

        tasks.append(wrapped())

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        await fut

    print(f"\nCompleted! Output saved to {OUTPUT_JSON_FILE}")


if __name__ == "__main__":
    asyncio.run(main_async())

import json
import os
from tqdm import tqdm
from agent import * 
import csv

INPUT_FILE = 'data/test.json'
OUTPUT_JSON_FILE = 'submission_test.json'
OUTPUT_CSV_FILE = 'submission_test.csv'

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
        
def main():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Cannot find {INPUT_FILE}")
        return

    results, processed_ids = load_processed_data(OUTPUT_JSON_FILE)
    print(f"Total question: {len(input_data)}")
    print(f"Completed previously: {len(processed_ids)}")

    for item in tqdm(input_data, desc="Processing"):
        qid = item.get('qid')
        
        if qid in processed_ids:
            continue

        question = item.get('question')
        choices = item.get('choices', [])

        prompt_choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        user_input = f"{question}\n\nCác lựa chọn:\n{prompt_choices}\n\nHãy chọn 1 đáp án đúng nhất."

        final_content = "A" # Default value

        try:
            response = agent.run(user_input)
            raw_answer = response.content

            formatted_res = format_agent.run(raw_answer)
            
            final_content = formatted_res.to_dict()['content']

        except Exception as e:
            print(f"\nError at {qid}: {e}")
            final_content = "Error"

        results.append({
            "qid": qid,
            "answer": final_content
        })
        
        processed_ids.add(qid)
        append_result_to_csv(OUTPUT_CSV_FILE, qid, final_content["key"])
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nnCompleted all, result at {OUTPUT_JSON_FILE}")

if __name__ == "__main__":
    main()
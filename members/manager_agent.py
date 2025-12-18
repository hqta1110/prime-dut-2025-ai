import sys
import os
import json
import csv
import re
import time
import logging
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock
from tqdm import tqdm

# --- 1. SETUP ĐƯỜNG DẪN ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import các module (Lưu ý: Các import này sẽ được load lại trong từng process con)
from external.inhouse_model import InhouseModel
from constants import MODEL_NAME, inhouse, model_setting
from agents import Agent, Runner
# Import các member agent
from members.compulsory_agent import compulsory
from members.stem_agent import stem
from members.rag_agent import rag
from members.multidomain_agent import multidomain
from members.vietnam_agent import vietnam
from members.format_agent import format
# --- CONFIGURATION ---
INPUT_FILE = "question_samples/test.json"
OUTPUT_FILE = "submission_test_BTC3.csv"
MAX_WORKERS = 16 # Số nhân CPU bạn muốn dùng (tương đương số process song song)
LOG_FILE = "pipeline_multicore_retrieval.log"

# Setup Logging cơ bản
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

# --- SYSTEM PROMPT ---
system_prompt = (
    "## Persona ##\n"
    "Bạn là trợ lý ảo, chuyên hỗ trợ người dùng với các câu hỏi trắc nghiệm thuộc nhiều lĩnh vực.\n"
    "Nhiệm vụ của bạn là phân loại các câu hỏi, từ đó đưa ra quyết định dựa trên các quy định sau:\n"
    "1. **Câu hỏi nhạy cảm, mang nội dung tiêu cực**: với các câu hỏi về các hành vi vi phạm pháp luật, chống phá Đảng Cộng Sản Việt Nam và Nhà nước Xã hội chủ nghĩa, xúc phạm lãnh tụ, các vấn đề nhạy cảm liên quan đến chính trị, bảo mật, an ninh trật tự xã hội, bạn hãy lựa chọn đáp án nào gần với ý 'từ chối cung cấp thông tin' nhất.\n"
    "2. **Câu hỏi liên quan đến tìm nội dung trong đoạn văn**: với các câu hỏi yêu cầu tìm kiếm thông tin trong một đoạn văn bản cụ thể, bạn hãy chuyển tiếp câu hỏi đến `rag_agent` mà không tóm tắt hay mô tả lại câu hỏi gốc.\n"
    "3. **Câu hỏi thuộc lĩnh vực khoa học, công nghệ, kĩ thuật và toán học**: với các câu hỏi liên quan đến các lĩnh vực STEM, bạn hãy chuyển tiếp câu hỏi đến `stem_agent`.\n"
    "4. **Câu hỏi liên quan trực tiếp đến Việt Nam**: với các câu hỏi có nội dung liên quan đến lịch sử, địa lý, chính trị, văn hóa, xã hội, luật pháp, kinh tế Việt Nam, bạn hãy chuyển tiếp câu hỏi đến `vietnam_agent`.\n"
    "5. **Các câu hỏi về chủ tịch Hồ Chí minh và chủ nghĩa Mác Lenin**: bạn hãy chuyển tiếp câu hỏi đến `compulsory_agent`.\n"
    "6. **Các câu hỏi chính trị, văn hóa, xã hội chung chung**: bạn hãy chuyển tiếp câu hỏi đến `multi_domain_agent`.\n"
    "**Lưu ý**: Nếu trực tiếp trả lời, hãy đảm bảo format trả lời của bạn phải tuân theo cấu trúc sau:\n"
    "1. Phân tích câu hỏi và các đáp án.\n"
    "2. Lập luận để chọn ra đáp án đúng nhất.\n"
    "3. Trả lời theo định dạng: <answer>A/B/C/D/E/...</answer>\n"
)

# --- HELPER FUNCTIONS ---
def format_question(item):
    question_text = item.get("question", "")
    choices = item.get("choices", [])
    formatted_choices = []
    labels = list(string.ascii_uppercase)
    for i, choice in enumerate(choices):
        label = labels[i] if i < len(labels) else str(i)
        formatted_choices.append(f"{label}. {choice}")
    return f"{question_text}\n" + "\n".join(formatted_choices)

def extract_answer_tag(text):
    if not text: return None
    match = re.search(r"<answer>\s*([A-Z])\b", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else "Error: No Tag Found"

# --- WORKER FUNCTION (Chạy trên 1 Process riêng biệt) ---
def worker_process(item):
    """
    Hàm này chạy độc lập trên 1 core CPU riêng.
    Không dùng async ở đây, dùng run_sync.
    """
    qid = item.get("qid")
    try:
        # 1. Khởi tạo Agent (Mỗi process có 1 agent riêng, không chung đụng)
        # Lưu ý: Việc khởi tạo này tốn chút resources nhưng đảm bảo thread-safe tuyệt đối
        manager_agent = Agent(
            name="manager_agent",
            model=inhouse,
            instructions=system_prompt,
            model_settings=model_setting,
            handoffs=[compulsory, stem, rag, multidomain, vietnam],
        )

        prompt_input = format_question(item)
        
        # 2. Gọi Runner.run_sync (Chạy đồng bộ trong process này)
        # Đây là chỗ "blocking", nhưng vì ta có nhiều process nên nó không ảnh hưởng process khác
        result = Runner.run_sync(manager_agent, prompt_input)
        
        final_text = ""
        if hasattr(result, 'final_output'):
            final_text = result.final_output
        else:
            final_text = str(result)

        extracted_ans = extract_answer_tag(final_text)
        return {"qid": qid, "answer": extracted_ans, "status": "success"}

    except Exception as e:
        error_msg = str(e)
        # Ghi log lỗi ra file riêng hoặc return lỗi
        with open(LOG_FILE, "a") as f:
            f.write(f"Error on {qid}: {error_msg}\n")
        return {"qid": qid, "answer": "ERROR", "status": "failed", "error": error_msg}

# --- MAIN CONTROLLER ---
def main():
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check resume
    processed_qids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row and 'qid' in row:
                    processed_qids.add(row['qid'])
    
    tasks_to_run = [item for item in data if item['qid'] not in processed_qids][:10]
    print(f"Total items: {len(data)}. Remaining: {len(tasks_to_run)}")

    if not tasks_to_run:
        return

    # Chuẩn bị file CSV (header)
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
        if not file_exists:
            writer.writeheader()

    # Dùng Manager.Lock để đảm bảo an toàn khi ghi file từ main process
    # (Thực ra ở đây ta ghi từ main thread sau khi worker trả kết quả nên không cần lock phức tạp)
    
    print(f"Spawning {MAX_WORKERS} processes...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tất cả task vào pool
        # future_to_qid giúp map future với input đầu vào
        future_to_item = {executor.submit(worker_process, item): item for item in tasks_to_run}
        
        with tqdm(total=len(tasks_to_run), desc="Multi-Core Processing") as pbar:
            for future in as_completed(future_to_item):
                result = future.result() # Lấy kết quả từ worker process trả về
                
                qid = result['qid']
                ans = result['answer']
                
                # Ghi ngay lập tức vào CSV
                with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
                    writer.writerow({'qid': qid, 'answer': ans})
                
                pbar.update(1)

if __name__ == "__main__":
    # Bắt buộc phải có block này khi dùng multiprocessing trên Windows/macOS
    main()
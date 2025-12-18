import sys
import os
import json
import csv
import re
import time
import logging
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Lock, Queue, Process
from tqdm import tqdm
import multiprocessing
# --- 1. SETUP ĐƯỜNG DẪN ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import các module
from external.inhouse_model import InhouseModel
from constants import MODEL_NAME, inhouse, model_setting
from agents import Agent, Runner
# Import các member agent
from members.compulsory_agent import compulsory
from members.stem_agent import stem
from members.rag_agent import rag
from members.multidomain_agent import multidomain
from members.vietnam_agent import vietnam
from members.format_agent import format as format_agent # Đổi tên để tránh trùng module system

# --- CONFIGURATION ---
INPUT_FILE = "question_samples/test.json"
OUTPUT_FILE = "submission_test_BTC4.csv"
MAX_WORKERS = 16 
LOG_FILE = "pipeline_multicore_retrieval.log"
TIMEOUT_SECONDS = 30
# Setup Logging
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
    return match.group(1).strip().upper() if match else None # Return None để dễ check lỗi

def is_valid_answer(ans):
    """Kiểm tra xem đáp án có phải là 1 ký tự A-Z duy nhất không"""
    return ans is not None and len(ans) == 1 and ans in string.ascii_uppercase

# --- WORKER FUNCTION ---
def worker_process(item):
    qid = item.get("qid")
    try:
        manager_agent = Agent(
            name="manager_agent",
            model=inhouse,
            instructions=system_prompt,
            model_settings=model_setting,
            handoffs=[compulsory, stem, rag, multidomain, vietnam],
        )

        prompt_input = format_question(item)
        result = Runner.run_sync(manager_agent, prompt_input)
        
        final_text = ""
        if hasattr(result, 'final_output'):
            final_text = result.final_output
        else:
            final_text = str(result)

        extracted_ans = extract_answer_tag(final_text)
        
        # Nếu không extract được, gán giá trị mặc định để xử lý sau
        final_ans = extracted_ans if extracted_ans else "ERROR"

        # Return thêm raw_output để dùng cho bước fix lỗi
        return {
            "qid": qid, 
            "answer": final_ans, 
            "raw_output": final_text, # Quan trọng: Lưu lại output gốc
            "status": "success"
        }

    except Exception as e:
        error_msg = str(e)
        with open(LOG_FILE, "a") as f:
            f.write(f"Error on {qid}: {error_msg}\n")
        return {"qid": qid, "answer": "ERROR", "raw_output": error_msg, "status": "failed"}
def format_fix_worker(raw_text, queue):
    """
    Hàm này sẽ chạy trong 1 process riêng biệt để có thể bị kill nếu treo.
    """
    try:
        # Prompt hướng dẫn format agent trích xuất
        fix_prompt = f"Trích xuất đáp án đúng nhất từ nội dung sau và chỉ trả về thẻ <answer>:\n\n{raw_text}"
        
        # Gọi agent (Lưu ý: format_agent phải hoạt động được trong process con)
        # Nếu format_agent không pickle được, bạn cần khởi tạo lại agent bên trong hàm này
        result = Runner.run_sync(format_agent, fix_prompt)
        
        final_text = ""
        if hasattr(result, 'final_output'):
            final_text = result.final_output
        else:
            final_text = str(result)
            
        extracted = extract_answer_tag(final_text)
        queue.put(extracted) # Đẩy kết quả vào hàng đợi
    except Exception as e:
        queue.put(None) # Đẩy None nếu lỗi
        
# --- FIX FORMAT WORKER ---
def run_format_fix(qid, raw_text):
    """
    Hàm quản lý timeout: Tạo process -> Chờ -> Kill nếu lâu -> Return mặc định
    """
    # Tạo hàng đợi để nhận kết quả
    q = Queue()
    
    # Khởi tạo process
    p = Process(target=format_fix_worker, args=(raw_text, q))
    p.start()
    
    # Chờ process chạy trong khoảng TIMEOUT_SECONDS
    p.join(timeout=TIMEOUT_SECONDS)
    
    if p.is_alive():
        # Nếu sau timeout mà process vẫn còn sống -> Kill
        p.terminate()
        p.join() # Clean up resource
        
        # Ghi log timeout
        err_msg = f"Timeout fixing QID: {qid}. Process killed. Defaulting to 'A'."
        print(f"\n[TIMEOUT] {err_msg}") # Print ra màn hình để thấy ngay
        with open(LOG_FILE, "a") as f:
            f.write(f"{err_msg}\n")
            
        return "A" # [REQUIREMENT] Trả về A mặc định
    else:
        # Nếu process xong đúng hạn
        if not q.empty():
            return q.get()
        return None # Process xong nhưng không trả về gì (lỗi bên trong)

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
    
    # Ở đây tôi bỏ [:10] để chạy full logic, bạn có thể thêm lại nếu muốn test
    tasks_to_run = [item for item in data if item['qid'] not in processed_qids][:10]
    print(f"Total items: {len(data)}. Remaining: {len(tasks_to_run)}")

    if not tasks_to_run:
        print("All tasks completed.")
        # Vẫn chạy xuống dưới để check lại file output phòng trường hợp chạy lần trước bị lỗi format
        pass 

    # Prepare CSV Header
    file_exists = os.path.exists(OUTPUT_FILE)
    if tasks_to_run: # Chỉ mở file append nếu có task mới
        with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
            if not file_exists:
                writer.writeheader()

    # --- TEMP STORAGE FOR VERIFICATION ---
    # Dùng Dict để lưu raw output cho phiên chạy hiện tại. 
    # Nếu resume từ file cũ, ta chỉ fix được những câu chạy trong phiên này (hoặc phải lưu raw ra file riêng).
    # Theo yêu cầu "lưu tạm thời", biến này sẽ mất khi tắt chương trình.
    raw_output_cache = {} 

    if tasks_to_run:
        print(f"Spawning {MAX_WORKERS} processes...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {executor.submit(worker_process, item): item for item in tasks_to_run}
            
            with tqdm(total=len(tasks_to_run), desc="Multi-Core Processing") as pbar:
                for future in as_completed(future_to_item):
                    result = future.result()
                    
                    qid = result['qid']
                    ans = result['answer']
                    raw_out = result['raw_output']
                    
                    # 1. Lưu vào cache để kiểm tra sau
                    raw_output_cache[qid] = raw_out
                    
                    # 2. Ghi vào CSV
                    with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
                        writer.writerow({'qid': qid, 'answer': ans})
                    
                    pbar.update(1)

    # --- POST-PROCESSING: VERIFY & FIX FORMAT ---
    print("\n--- Starting Post-Processing Verification ---")
    
    # 1. Đọc toàn bộ CSV hiện tại vào memory
    rows = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    
    updated_count = 0
    
    # 2. Duyệt qua các dòng để tìm lỗi
    # Chúng ta dùng tqdm để hiển thị tiến độ fix lỗi
    print("Checking for malformed answers...")
    for row in tqdm(rows, desc="Verifying"):
        current_qid = row['qid']
        current_ans = row['answer']
        
        # Điều kiện check: Không phải 1 ký tự alphabet viết hoa
        if not is_valid_answer(current_ans):
            # Cần fix
            raw_text = raw_output_cache.get(current_qid)
            
            if raw_text:
                # Nếu có trong cache phiên hiện tại -> Run fix
                fixed_ans = run_format_fix(raw_text)
                
                if fixed_ans and is_valid_answer(fixed_ans):
                    row['answer'] = fixed_ans # Update in memory
                    updated_count += 1
                elif fixed_ans == "A": # Trường hợp timeout trả về A
                    row['answer'] = "A"
                    updated_count += 1
            else:
                # Trường hợp: Dòng lỗi này từ phiên chạy trước, không có raw_text trong RAM
                # Ta bỏ qua hoặc ghi log warning
                # logging.warning(f"QID {current_qid} has invalid format but no raw output available to fix.")
                pass

    # 3. Ghi đè lại file CSV nếu có sự thay đổi
    if updated_count > 0:
        print(f"Found and fixed {updated_count} malformed answers. Overwriting CSV...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['qid', 'answer'])
            writer.writeheader()
            writer.writerows(rows)
        print("CSV overwritten successfully.")
    else:
        print("No malformed answers found (or unable to fix). CSV remains unchanged.")

if __name__ == "__main__":
    main()
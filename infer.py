import json
import requests
import pandas as pd
import string
import re
import time
import os
import csv
from datetime import datetime

# ==========================================
# CẤU HÌNH API VNPT (TRACK 2)
# ==========================================
BASE_URL = "https://api.idg.vnpt.vn/data-service/v1/chat/completions"
# BASE_URL = "https://167.179.48.115:8000/v1/chat/completions"

# --- ĐIỀN THÔNG TIN CỦA BẠN ---
# Token lấy từ portal cuộc thi (Tab Instruction) [cite: 18]
ACCESS_TOKEN = "" 
TOKEN_ID = ""         
TOKEN_KEY = ""       

# Chọn Model: 
# 'vnptai_hackathon_small': 60 req/h [cite: 23]
# 'vnptai_hackathon_large': 40 req/h [cite: 90]
MODEL_NAME = "vnptai_hackathon_small" 
# MODEL_NAME = "Qwen3-32B"


INPUT_FILE = "val.json"
OUTPUT_FILE = "submission_vnpt.csv"

# ==========================================
# 1. CÁC HÀM HỖ TRỢ (PROMPT & EXTRACT)
# ==========================================
ALPHABET = string.ascii_uppercase 

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_prompt(item):
    """Chuyển câu hỏi thành định dạng Text"""
    question = item['question']
    choices = item['choices']
    user_content = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        if i < 26: user_content += f"{ALPHABET[i]}. {choice}\n"
    return user_content

def get_system_prompt():
    """Yêu cầu model suy luận (CoT) rồi mới chốt đáp án trong thẻ XML"""
    return (
        "## PERSONA\n"
        "Bạn là trợ lý ảo tiếng Việt. Nhiệm vụ của bạn là trả lời câu hỏi trắc nghiệm của người dùng theo format đã quy định sẵn.\n"
        "## INSTRUCTION\n"
        "Bước 1: Suy nghĩ và xác định câu hỏi của user thật cẩn thận, sau đó phân loại câu hỏi thành 1 trong các nhóm sau:\n"
        "   + Nhóm 1: Câu hỏi liên quan đến các vấn đề vi phạm pháp luật, các vấn đề nhạy cảm.\n"
        "   + Nhóm 2: Câu hỏi chứa các thông tin cơ bản, quan trọng, bắt buộc phải trả lời đúng.\n"
        "   + Nhóm 3: Câu hỏi liên quan đến khoa học, toán học, cần nhiều suy luận logic.\n"
        "   + Nhóm 4: Câu hỏi đa lĩnh vực.\n"
        "   + Nhóm 5: Câu hỏi yêu cầu đọc hiểu văn bản dài.\n"
        "Bước 2: Dựa trên phân loại từ bước 1, thực hiện các hành động tương ứng cho từng nhóm:\n"
        "   + Nhóm 1: Ưu tiên chọn các đáp án không trả lời\n"
        "   + Nhóm 2: Suy nghĩ tỉ mỉ, dựa trên lập trường và góc nhìn của Việt Nam để trả lời.\n"
        "   + Nhóm 3: Lập kế hoạch từng bước để giải quyết vấn đề, sau đó tiến hành tính toán từng bước cẩn thận để tìm ra câu trả lời.\n"
        "   + Nhóm 4: Suy nghĩ cẩn thận, dựa trên kiến thức của bạn để trả lời.\n"
        "   + Nhóm 5: Xác định trọng tâm yêu cầu của câu hỏi, sau đó tìm kiếm thông tin cần thiết từ trong văn bản được cung cấp có tác dụng hỗ trợ cho việc đưa ra câu trả lời cuôi cùng. Đáp án của các câu hỏi thuộc nhóm này phải hoàn toàn dựa trên thông tin trong văn bản được cung cấp.\n"
        "**Lưu ý cho bước 2**: Mọi quá trình suy nghĩ, lập luận đều phải để ở trong tag <thinking>\n"
        "Bước 3: Sau khi có câu trả lời, hãy chọn ra 1 đáp án duy nhất, phù hợp nhất với câu trả lời từ trong danh sách các đáp án đã cho. Đáp án phải nằm trong tag <answer>\n"
        "## FORMAT EXAMPLE\n"
        "<thinking>[Lập luận của bạn để đến câu trả lời]</thinking>\n"
        "<answer>[Đáp án cuối cùng (A/B/C/D/....)]</answer>\n"
    )

def extract_answer(content):
    if not content: return None
    # Ưu tiên tìm trong thẻ <answer>
    match = re.search(r"<answer>(.*?)</answer>", content, flags=re.IGNORECASE | re.DOTALL)
    if match:
        raw = match.group(1).strip()
        char_match = re.search(r"([A-Z])", raw.upper())
        if char_match: return char_match.group(1)
    # Fallback pattern cũ
    fallback = re.search(r"Answer:\s*([A-Z])", content, flags=re.IGNORECASE)
    return fallback.group(1) if fallback else content


# ==========================================
# 2. CÁC HÀM QUẢN LÝ FILE CSV (QUAN TRỌNG)
# ==========================================
def get_processed_qids(output_file):
    """Đọc file CSV để lấy danh sách các QID đã làm xong"""
    if not os.path.exists(output_file):
        return set()
    
    processed = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # Bỏ qua header
            for row in reader:
                if row:
                    processed.add(row[0]) # Cột đầu tiên là qid
    except Exception as e:
        print(f"⚠️ Cảnh báo: Không đọc được file cũ ({e}). Sẽ chạy lại từ đầu.")
    return processed

def append_result_to_csv(output_file, qid, answer):
    """Ghi ngay lập tức 1 dòng vào file CSV"""
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Nếu file chưa tồn tại, ghi header trước
        if not file_exists:
            writer.writerow(['qid', 'answer'])
        
        writer.writerow([qid, answer])
        f.flush() # Đẩy dữ liệu từ buffer xuống đĩa ngay lập tức

# ==========================================
# 3. HÀM GỌI API VỚI "INFINITE RETRY"
# ==========================================
def call_api_infinite_retry(item):
    """
    Gọi API trong vòng lặp vô tận cho đến khi thành công.
    Tự động ngủ khi gặp Rate Limit.
    """
    qid = item['qid']
    user_prompt = format_prompt(item)
    
    # Endpoint [cite: 88]
    endpoint = f"{BASE_URL}/{MODEL_NAME.replace('_', '-')}"
    # Lưu ý: URL thực tế có thể dùng gạch ngang (-) thay vì gạch dưới (_) tuỳ vào config thực tế của server,
    # nhưng theo tài liệu endpoint là /vnptai-hackathon-large[cite: 88].
    # Tuy nhiên model name trong body lại là vnptai_hackathon_large[cite: 99].
    # # Tôi sẽ giữ logic map đúng endpoint dựa trên tên model.
    if "small" in MODEL_NAME:
         endpoint = f"{BASE_URL}/vnptai-hackathon-small" # [cite: 21]
    else:
         endpoint = f"{BASE_URL}/vnptai-hackathon-large" # [cite: 88]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}", # [cite: 94]
        "Token-id": TOKEN_ID,                       # [cite: 94]
        "Token-key": TOKEN_KEY                      # [cite: 94]
    }
    
    payload = {
        "model": MODEL_NAME, # [cite: 95]
        "messages": [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0,
        "max_completion_tokens": 1024,
        "n": 1
    }

    while True: # Vòng lặp vô tận cho đến khi xong câu này
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            
            # --- TRƯỜNG HỢP THÀNH CÔNG ---
            if response.status_code == 200:
                return response.json()
            
            # --- TRƯỜNG HỢP HẾT QUOTA (429) ---
            elif response.status_code == 429:
                # Quota Large: 40 req/h -> Reset khá lâu
                wait_time = 60 # Check lại mỗi 1 phút
                print(f"\n⏳ [{datetime.now().strftime('%H:%M:%S')}] Rate Limit (429) tại câu {qid}.")
                print(f"   Script sẽ 'treo' và thử lại sau mỗi {wait_time}s cho đến khi server mở lại...")
                time.sleep(wait_time) 
                continue # Quay lại đầu vòng lặp while để thử lại
                
            # --- LỖI SERVER (5xx) ---
            elif response.status_code >= 500:
                print(f"⚠️ Server Error {response.status_code}. Retrying in 5s...")
                time.sleep(5)
                continue
                
            # --- LỖI CLIENT (400, 401...) ---
            else:
                print(f"❌ Fatal Error {response.status_code}: {response.text}")
                return None # Lỗi này không retry được (sai key, sai model...)

        except Exception as e:
            print(f"⚠️ Connection Error ({e}). Retrying in 5s...")
            time.sleep(5)
            continue

# ==========================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    # 1. Load Data đầu vào
    data = load_data(INPUT_FILE)[8:10] # set 1-2 sample để test trước khi chạy full data
    total_questions = len(data)
    
    # 2. Kiểm tra tiến độ cũ (Resume Logic)
    processed_qids = get_processed_qids(OUTPUT_FILE)
    print(f"📂 Tổng số câu: {total_questions}")
    print(f"✅ Đã hoàn thành trước đó: {len(processed_qids)} câu.")
    
    # Lọc ra danh sách các câu chưa làm
    remaining_items = [item for item in data if item['qid'] not in processed_qids]
    print(f"🚀 Số câu cần xử lý tiếp: {len(remaining_items)}")
    print("--- Bắt đầu chạy (Nhấn Ctrl+C để dừng an toàn) ---")

    for i, item in enumerate(remaining_items):
        qid = item['qid']
        
        print(f"Processing ({i+1}/{len(remaining_items)}) ID: {qid}...", end=" ", flush=True)
        
        # Gọi API (Hàm này sẽ treo ở đó nếu 429, không bao giờ return None trừ khi lỗi fatal)
        api_response = call_api_infinite_retry(item)
        # print(api_response)
        final_ans = "A" # Default safe answer
        
        if api_response and 'choices' in api_response:
            content = api_response['choices'][0]['message']['content']
            final_ans = extract_answer(content)
            print(f"-> Done. Ans: {final_ans}")
        else:
            print(f"-> Failed (Error/Null). Default A")

        # GHI NGAY LẬP TỨC XUỐNG FILE
        append_result_to_csv(OUTPUT_FILE, qid, final_ans)
        
        # Ngủ nhẹ để tránh spam server quá gắt (Good practice)
        # Với limit 40 req/h, trung bình 90s/req. 
        # Ta sleep 5s, nếu hết quota thì hàm call_api tự lo việc ngủ dài.
        time.sleep(5) 

    print(f"\n🎉 HOÀN THÀNH TẤT CẢ! Kết quả tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
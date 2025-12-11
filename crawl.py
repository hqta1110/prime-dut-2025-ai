import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

# --- CẤU HÌNH ---
BASE_URL = "https://vanban.chinhphu.vn/he-thong-van-ban" # update base url theo từng loại văn bản
DOWNLOAD_FOLDER = "NghiDinh_2025"
TARGET_YEAR = "2025"
TARGET_TYPE = "Nghị định" # nhớ thay
MAX_PAGES_LIMIT = 6 

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)


def check_pdf_type(file_path):
    """
    Trả về 'TEXT' nếu là file văn bản (copy được), 
    Trả về 'SCAN' nếu là file ảnh (không copy được text).
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) > 0:
                # Thử lấy text trang đầu tiên
                first_page_text = pdf.pages[0].extract_text()
                
                # Nếu lấy được text và độ dài > 20 ký tự -> Là file Text
                if first_page_text and len(first_page_text.strip()) > 20:
                    return "TEXT"
                else:
                    return "SCAN"
            return "UNKNOWN"
    except Exception as e:
        return "ERROR"


def setup_driver():
    edge_options = Options()
    # edge_options.add_argument("--headless") 
    
    current_folder = os.path.dirname(os.path.abspath(__file__))
    driver_path = os.path.join(current_folder, "msedgedriver.exe")
    
    if not os.path.exists(driver_path):
        print("LỖI: Không tìm thấy file msedgedriver.exe")
        exit()

    service = Service(executable_path=driver_path)
    driver = webdriver.Edge(service=service, options=edge_options)
    driver.maximize_window()
    return driver

def download_pdf(pdf_url, file_name):
    try:
        response = requests.get(pdf_url, stream=True, verify=False)
        file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
        
        # 1. Tải file về trước
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # 2. Kiểm tra loại file ngay sau khi tải
        pdf_type = check_pdf_type(file_path)
        
        # 3. Đổi tên file để đánh dấu (Ví dụ: [TEXT]_nghidinh.pdf)
        new_name = f"[{pdf_type}]_{file_name}"
        new_path = os.path.join(DOWNLOAD_FOLDER, new_name)
        
        # Rename file cũ sang tên mới
        if not os.path.exists(new_path):
            os.rename(file_path, new_path)
            print(f"      [OK] Đã tải ({pdf_type}): {new_name}")
        else:
            # Nếu file đã tồn tại thì xóa file tạm vừa tải
            os.remove(file_path) 
            print(f"      [SKIP] File đã tồn tại: {new_name}")

    except Exception as e:
        print(f"      [ERR] Lỗi xử lý file: {e}")

def crawl_vanban():
    driver = setup_driver()
    wait = WebDriverWait(driver, 20)

    try:
        print("1. Đang truy cập trang web...")
        driver.get(BASE_URL)
        time.sleep(3)
        
        # --- BƯỚC 1: LỌC DỮ LIỆU ---
        print("2. Đang thiết lập bộ lọc...")
        all_selects = driver.find_elements(By.TAG_NAME, "select")
        
        # Chọn Loại văn bản
        for sel in all_selects:
            try:
                if TARGET_TYPE in sel.text:
                    Select(sel).select_by_visible_text(TARGET_TYPE)
                    break
            except: continue
            
        # Chọn Năm
        for sel in all_selects:
            try:
                if TARGET_YEAR in sel.text:
                    Select(sel).select_by_visible_text(TARGET_YEAR)
                    break
            except: continue

        # Click Tìm kiếm
        try:
            search_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Tìm kiếm')]")
            driver.execute_script("arguments[0].click();", search_btn)
            time.sleep(5) 
        except: pass

        # --- BƯỚC 2: VÒNG LẶP ---
        current_page_number = 1
        
        while True:
            print(f"\n--- ĐANG XỬ LÝ TRANG {current_page_number} ---")
            
            # --- CRAWL PDF (Giữ nguyên) ---
            doc_links = driver.find_elements(By.XPATH, "//a[contains(@href, 'docid')]")
            unique_urls = []
            seen = set()
            for link in doc_links:
                u = link.get_attribute('href')
                if u and "docid" in u and u not in seen:
                    seen.add(u)
                    unique_urls.append(u)
            
            print(f"   -> Tìm thấy {len(unique_urls)} văn bản.")
            if len(unique_urls) == 0: 
                print("   [INFO] Danh sách trống. Có thể đã hết.")
                break

            for idx, url in enumerate(unique_urls):
                try:
                    driver.execute_script("window.open('');")
                    driver.switch_to.window(driver.window_handles[1])
                    driver.get(url)
                    pdf_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
                    if pdf_elements:
                        pdf_url = pdf_elements[0].get_attribute('href')
                        file_name = pdf_url.split('/')[-1]
                        # print(f"   [{idx+1}] PDF: {file_name}")
                        download_pdf(pdf_url, file_name)
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                except:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

            if MAX_PAGES_LIMIT > 0 and current_page_number >= MAX_PAGES_LIMIT: break

            # --- LOGIC SANG TRANG (VÉT CẠN) ---
            next_page_target = current_page_number + 1
            keyword_href = f"Page${next_page_target}" # Từ khóa quan trọng: Page$6, Page$9...
            
            print(f"   >>> Đang tìm link chứa: '{keyword_href}' hoặc text '{next_page_target}'...")
            
            # 1. Lấy TẤT CẢ các thẻ <a> trong bảng phân trang (class 'grid-pager')
            # Nếu không tìm thấy class grid-pager, lấy toàn bộ thẻ a có href chứa Page$
            potential_links = driver.find_elements(By.XPATH, "//tr[@class='grid-pager']//a")
            if not potential_links:
                potential_links = driver.find_elements(By.XPATH, "//a[contains(@href, 'Page$')]")
            
            found_next = False
            
            for link in potential_links:
                try:
                    href_val = link.get_attribute('href')
                    text_val = link.text.strip()
                    
                    # Debug: In ra để xem script nhìn thấy gì (Quan trọng)
                    # print(f"      [Check] Text: '{text_val}' | Href: ...{href_val[-15:] if href_val else 'None'}")
                    
                    if not href_val: continue

                    # ĐIỀU KIỆN 1: Href chứa đúng lệnh Page$X (Chính xác nhất)
                    if keyword_href in href_val:
                        print(f"   >>> Phát hiện nút sang trang theo HREF! (Link chứa {keyword_href})")
                        driver.execute_script("arguments[0].scrollIntoView(true);", link)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", link)
                        found_next = True
                        break
                    
                    # ĐIỀU KIỆN 2: Text hiển thị đúng số trang (Dự phòng)
                    if text_val == str(next_page_target):
                        print(f"   >>> Phát hiện nút sang trang theo TEXT! (Text là {text_val})")
                        driver.execute_script("arguments[0].scrollIntoView(true);", link)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", link)
                        found_next = True
                        break

                except Exception as e:
                    continue
            
            if found_next:
                current_page_number += 1
                print("   >>> Đang tải trang mới...")
                time.sleep(8) # Đợi web load xong
            else:
                print(f"   [STOP] Không tìm thấy đường dẫn sang trang {next_page_target}. Dừng script.")
                break

    except Exception as e:
        print(f"Lỗi Fatal: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    crawl_vanban()

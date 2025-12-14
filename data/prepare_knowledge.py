from tools.retrieval import get_embedding, get_embeddings
from .craw_data_utils import extract_text
from .chunking import *
import json
import os
import shutil
from tqdm import tqdm
import time

TXT_PDF_BASE_FOLDER = './data/text_pdf'
PREPROCESSED_TXT_BASE_FOLDER = "./data/preprocessed_txt"
CHUNKED_TXT_BASE_FOLDER = "./data/chunked_txt"
MAX_RETRIES = 5
RETRY_DELAY = 15

knowledge = []
texts = []
fields = []
embeddings = []

os.makedirs(PREPROCESSED_TXT_BASE_FOLDER, exist_ok=True)
os.makedirs(CHUNKED_TXT_BASE_FOLDER, exist_ok=True)

for field in tqdm(os.listdir(TXT_PDF_BASE_FOLDER), desc="Fields"):
    src_field_path = os.path.join(TXT_PDF_BASE_FOLDER, field)
    dst_field_path = os.path.join(PREPROCESSED_TXT_BASE_FOLDER, field)
    os.makedirs(dst_field_path, exist_ok=True)

    for file in tqdm(os.listdir(src_field_path), desc=f"Files in {field}", leave=False):
        src_file_path = os.path.join(src_field_path, file)
        dst_file_path = os.path.join(dst_field_path, f"{os.path.splitext(file)[0]}.txt")
        text = extract_text(src_file_path)
        if text:
            with open(dst_file_path, "w", encoding="utf-8") as f:                
                f.write(text)


for field in tqdm(os.listdir(PREPROCESSED_TXT_BASE_FOLDER), desc="Fields"):
    src_field_path = os.path.join(PREPROCESSED_TXT_BASE_FOLDER, field)
    dst_field_path = os.path.join(CHUNKED_TXT_BASE_FOLDER, field)
    os.makedirs(dst_field_path, exist_ok=True)

    idx = 0
    for file in tqdm(os.listdir(src_field_path), desc=f"Files in {field}", leave=False):
        dst_files = os.listdir(dst_field_path)

        flag = False
        for dst_file in dst_files:
            if os.path.splitext(file)[0] in os.path.splitext(dst_file)[0]:
                flag = True
                break

        if flag:
            continue

        
        src_file_path = os.path.join(src_field_path, file)

        with open(src_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            texts, text_lengths = sentence_chunking_main(
                1024, 60, True, content
            )
        except:
            print(src_file_path)

        time.sleep(60)
        for text in texts:
            dst_file_path = os.path.join(dst_field_path, f"{os.path.splitext(file)[0]}_{idx}.txt")
            with open(dst_file_path, "w", encoding="utf-8") as f:
                f.write(text)
            idx += 1

for field in tqdm(os.listdir(CHUNKED_TXT_BASE_FOLDER), desc="Fields"):
    for file in os.listdir(os.path.join(CHUNKED_TXT_BASE_FOLDER, field)):
        file_path = os.path.join(CHUNKED_TXT_BASE_FOLDER, field, file)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
        fields.append(field)

embeddings = get_embeddings(texts)

for text, field, embedding in zip(texts, fields, embeddings):
    knowledge.append(
        {
            "text": text,
            "fields": field.split("-"),
            "embedding": embedding
        }
    )

with open("data/knowledge.json", 'w', encoding="utf-8") as json_file:
    json.dump(knowledge, json_file, ensure_ascii=False, indent=4)





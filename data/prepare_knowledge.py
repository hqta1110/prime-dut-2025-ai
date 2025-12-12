from tools.retrieval import get_embedding, get_embeddings
from .craw_data_utils import extract_text
import json
import os
import shutil
from tqdm import tqdm

TXT_PDF_BASE_FOLDER = './data/text_pdf'
PREPROCESSED_TXT_BASE_FOLDER = "./data/preprocessed_txt"
BATCH_SIZE = 1024

knowledge = []
texts = []
fields = []
embeddings = []

try:
    shutil.rmtree(PREPROCESSED_TXT_BASE_FOLDER)
    os.makedirs(PREPROCESSED_TXT_BASE_FOLDER, exist_ok=True)
except:
    os.makedirs(PREPROCESSED_TXT_BASE_FOLDER, exist_ok=True)

for field in tqdm(os.listdir(TXT_PDF_BASE_FOLDER), desc="Fields"):
    src_field_path = os.path.join(TXT_PDF_BASE_FOLDER, field)
    dst_field_path = os.path.join(PREPROCESSED_TXT_BASE_FOLDER, field)
    os.makedirs(dst_field_path, exist_ok=True)

    for file in tqdm(os.listdir(src_field_path), desc=f"Files in {field}", leave=False):
        src_file_path = os.path.join(src_field_path, file)
        dst_file_path = os.path.join(dst_field_path, f"{os.path.splitext(file)[0]}.txt")
        text = extract_text(src_file_path)
        with open(dst_file_path, "w", encoding="utf-8") as f:
            f.write(text)

# for field in os.listdir(BASE_FOLDER):
#     for file in os.listdir(os.path.join(BASE_FOLDER, field)):
#         file_path = os.path.join(BASE_FOLDER, field, file)
#         with open(file_path, 'r', encoding='utf-8') as file:
#             texts.append(file.read())
#         fields.append(field)

# for i in range(0, len(texts), BATCH_SIZE):
#     batch = texts[i : i + BATCH_SIZE]
#     batch_emb = get_embeddings(batch)   
#     embeddings.extend(batch_emb)

# for text, field, embedding in zip(texts, fields, embeddings):
#     knowledge.append(
#         {
#             "text": text,
#             "fields": field.split("-"),
#             "embedding": embedding
#         }
#     )

# with open("data/knowledge.json", 'w', encoding="utf-8") as json_file:
#     json.dump(knowledge, json_file, ensure_ascii=False, indent=4)





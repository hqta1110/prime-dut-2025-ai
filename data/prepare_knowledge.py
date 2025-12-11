from tools.retrieval import get_embedding, get_embeddings
import json
import os

BASE_FOLDER = "./data/preprocessed_txt"
BATCH_SIZE = 1024

knowledge = []
texts = []
fields = []
embeddings = []

for field in os.listdir(BASE_FOLDER):
    for file in os.listdir(os.path.join(BASE_FOLDER, field)):
        file_path = os.path.join(BASE_FOLDER, field, file)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
        fields.append(field)

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    batch_emb = get_embeddings(batch)   
    embeddings.extend(batch_emb)

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





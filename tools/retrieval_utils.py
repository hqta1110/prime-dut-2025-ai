from dotenv import load_dotenv
import os
import requests
import faiss
import numpy as np
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents import Agent, FunctionTool, RunContextWrapper, function_tool
_FAISS_CACHE = {}
_KNOWLEDGE_CACHE = None

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
EMBEDDING_BEARER_TOKEN = os.getenv("EMBEDDING_BEARER_TOKEN")
EMBEDDING_TOKEN_KEY = os.getenv("EMBEDDING_TOKEN_KEY")
EMBEDDING_TOKEN_ID = os.getenv("EMBEDDING_TOKEN_ID")
BATCH_SIZE = 16

def _post_embedding(
    batch,
    headers,
    max_retries=3,
    retry_delay=60,
):
    json_data = {
        "model": "vnptai_hackathon_embedding",
        "input": batch,
        "encoding_format": "float"
    }

    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(
                f"{BASE_URL}/vnptai-hackathon-embedding",
                headers=headers,
                json=json_data,
                timeout=300
            )

            if res.status_code == 200:
                return [d["embedding"] for d in res.json()["data"]]

        except requests.RequestException as e:
            print(f"[Embedding] Attempt {attempt} exception: {e}")

        if attempt < max_retries:
            time.sleep(retry_delay)

    # Retry hết
    print(f"[Embedding] Failed after {max_retries} retries, batch_size={len(batch)}")
    return None

def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {EMBEDDING_BEARER_TOKEN}",
        "Token-id": EMBEDDING_TOKEN_ID,
        "Token-key": EMBEDDING_TOKEN_KEY,
        "Content-Type": "application/json"
    }

    json_data = {
        "model": "vnptai_hackathon_embedding",
        "input": text,
        "encoding_format": "float" 
    }
    res = requests.post(
        f"{BASE_URL}/vnptai-hackathon-embedding",
        headers=headers,
        json=json_data
    )
    if res.status_code == 200:
        return res.json()['data'][0]['embedding']
    else:
        return []
    
def get_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {EMBEDDING_BEARER_TOKEN}",
        "Token-id": EMBEDDING_TOKEN_ID,
        "Token-key": EMBEDDING_TOKEN_KEY,
        "Content-Type": "application/json"
    }

    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    
    embeddings_by_index = [None] * len(batches)

    max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_post_embedding, batch, headers): idx
            for idx, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            idx = futures[future]        # batch index
            result = future.result()
            if result is None:
                return [[]]
            embeddings_by_index[idx] = result

    # Nối theo đúng thứ tự index
    final_embeddings = []
    for batch_emb in embeddings_by_index:
        final_embeddings.extend(batch_emb)

    return final_embeddings

def load_knowledge():
    global _KNOWLEDGE_CACHE
    if _KNOWLEDGE_CACHE is not None:
        return _KNOWLEDGE_CACHE

    path = "./data/knowledge.json"
    if not os.path.exists(path):
        _KNOWLEDGE_CACHE = []
        return _KNOWLEDGE_CACHE

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["embedding"] = np.array(item["embedding"], dtype="float32")

    _KNOWLEDGE_CACHE = data
    return _KNOWLEDGE_CACHE

def get_ivf_index(field_key: str, data, metric="cosine"):
    global _FAISS_CACHE

    embeddings = np.array([x["embedding"] for x in data], dtype="float32")
    cache_key = f"{field_key}_{metric}_{len(data)}"

    if cache_key in _FAISS_CACHE:
        return _FAISS_CACHE[cache_key]

    if len(data) == 0:
        return None

    d = embeddings.shape[1]

    # ---- Small data → Flat ----
    if len(data) < 5000:
        if metric == "cosine":
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.IndexFlatL2(d)

        index.add(embeddings)
        _FAISS_CACHE[cache_key] = index
        return index

    # ---- Large data → IVF ----
    if metric == "cosine":
        faiss.normalize_L2(embeddings)
        metric_type = faiss.METRIC_INNER_PRODUCT
        quantizer = faiss.IndexFlatIP(d)
    else:
        metric_type = faiss.METRIC_L2
        quantizer = faiss.IndexFlatL2(d)

    nlist = int(np.sqrt(len(embeddings)))
    index = faiss.IndexIVFFlat(quantizer, d, nlist, metric_type)

    # Train with sample if huge
    if len(embeddings) > 100000:
        sample = embeddings[np.random.choice(len(embeddings), 100000, replace=False)]
        index.train(sample)
    else:
        index.train(embeddings)

    index.add(embeddings)
    index.nprobe = min(32, nlist)

    _FAISS_CACHE[cache_key] = index
    return index


def vector_search(index, data, query_emb, k=5, metric="cosine"):
    if index is None:
        return []

    query = np.array([query_emb], dtype="float32")

    if metric == "cosine":
        if np.linalg.norm(query) == 0:
            return []
        faiss.normalize_L2(query)

    if isinstance(index, faiss.IndexIVF):
        index.nprobe = min(32, index.nlist)

    distances, indices = index.search(query, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        if metric == "cosine":
            score = float(dist)  # cosine similarity
        else:
            score = float(dist)

        results.append({
            "score": score,
            "item": data[idx]
        })

    return results


@function_tool
def retrieval(query: str, fields: list, enable_filter = False, distance_metric="cosine", k=5):
    """
    Gọi hàm này để truy vấn các thông tin cần thiết để trả lời câu hỏi
    Args:
        query (str): Câu hỏi của người dùng
        fields (list): Danh sách các lĩnh vực để lọc. Chỉ nhận các giá trị sau: "circular", "constitution", "culture", "decree", 
        "geography", "history", "law", "philosophy", "regulation", "others"
    """
    for field in fields:
        if field not in [
        "circular", "constitution", "culture", "decree", 
        "geography", "history", "law", "philosophy", "regulation", 
        "others"]:
            fields.remove(field)
    try:
        knowledge = load_knowledge()

        if enable_filter and fields:
            field_key = "_".join(sorted(fields))
            data = [
                x for x in knowledge
                if any(d in fields for d in x["fields"])
            ]
        else:
            field_key = "all"
            data = knowledge

        query_emb = get_embedding(query)

        index = get_ivf_index(field_key, data, metric=distance_metric)

        result = vector_search(
            index=index,
            data=data,
            query_emb=query_emb,
            k=k,
            metric=distance_metric
        )

        return result

    except Exception as e:
        print(f"[Retrieval] Exception: {e}")
        return []
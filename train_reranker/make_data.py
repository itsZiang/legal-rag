import json 
import requests

with open("train_reranker_questions_168.json", "r", encoding="utf-8") as f:
    train_reranker_questions = json.load(f)
    
    
data = []

count = 0
for q in train_reranker_questions:
    body = {"query": q, "limit": 20, "top_n": 20}
    ids = requests.post("http://localhost:8000/search_ids_multi_queries", json=body, timeout=30).json()["ids"] 
    data.append({"question": q, "ids": ids})
    count += 1
    print(count)
    


with open("../backend/src/final_chunk/embeddings_output.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
    
id_to_text = {
    chunk["id"]: f'{chunk["title"]}. {chunk["content"]}'
    for chunk in chunks
}

output = []
for item in data:
    question = item["question"]
    selected_chunks = [id_to_text[i] for i in item["ids"] if i in id_to_text]
    output.append({
        "question": question,
        "chunks": selected_chunks
    })

with open("data_168.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
    

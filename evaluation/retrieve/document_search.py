import json
import requests
import os


def get_documents(query, limit=20):
    body = {"query": query, "limit": limit}
    response = requests.post("http://localhost:8000/search_ids", json=body)
    return response.json()["ids"]


def get_documents_rerank(query, limit=20, top_n=20):
    body = {"query": query, "limit": limit, "top_n": top_n}
    response = requests.post("http://localhost:8000/search_ids_rerank", json=body)
    return response.json()["ids"]


def get_documents_hybrid(query, limit=20, top_n=20):
    body = {"query": query, "limit": limit, "top_n": top_n}
    response = requests.post("http://localhost:8000/search_ids_hybrid", json=body)
    return response.json()["ids"]


def get_documents_multi_queries(query, limit=20, top_n=20):
    body = {"query": query, "limit": limit, "top_n": top_n}
    response = requests.post("http://localhost:8000/search_ids_multi_queries", json=body)
    return response.json()["ids"]


def append_to_jsonl(file_path, data):
    """Append a single JSON object as a line to a JSONL file"""
    with open(file_path, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def load_processed_questions(file_paths):
    """Load already processed questions from multiple JSONL files"""
    processed = {filepath: set() for filepath in file_paths}
    
    for filepath in file_paths:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed[filepath].add(data["question"])
                    except json.JSONDecodeError:
                        continue
    
    return processed


def main():
    # File paths for individual JSONL files
    # bge_file = "predictions_bge_m3.jsonl"
    rerank_file = "predictions_rerank-bge_m3.jsonl" 
    hybrid_file = "predictions_hybrid-bge_m3.jsonl"
    multi_queries_file = "predictions_multi_queries-bge_m3.jsonl"
    
    file_paths = [
        # bge_file, 
        rerank_file, 
        hybrid_file, 
        multi_queries_file
        ]
    
    # Load processed questions for each file
    processed_questions = load_processed_questions(file_paths)
    
    # Load questions
    with open("test_set.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = [item["question"] for item in data]
    total_questions = len(questions)
    
    for i, question in enumerate(questions, 1):
        question_short = question[:100] + "..." if len(question) > 100 else question
        print(f"Processing question {i}/{total_questions}: {question_short}")
        
        # Regular search
        # if question not in processed_questions[bge_file]:
        #     doc_ids = get_documents(question)
        #     append_to_jsonl(bge_file, {"question": question, "doc_ids": doc_ids})
        #     print(f"  - Saved BGE result")
        # else:
        #     print(f"  - Skipping BGE (already processed)")
            
        # Rerank search
        if question not in processed_questions[rerank_file]:
            doc_ids_rerank = get_documents_rerank(question)
            append_to_jsonl(rerank_file, {"question": question, "doc_ids": doc_ids_rerank})
            print(f"  - Saved Rerank result")
        else:
            print(f"  - Skipping Rerank (already processed)")
            
        # Hybrid search
        if question not in processed_questions[hybrid_file]:
            doc_ids_hybrid = get_documents_hybrid(question)
            append_to_jsonl(hybrid_file, {"question": question, "doc_ids": doc_ids_hybrid})
            print(f"  - Saved Hybrid result")
        else:
            print(f"  - Skipping Hybrid (already processed)")
            
        # Multi queries search
        if question not in processed_questions[multi_queries_file]:
            doc_ids_multi_queries = get_documents_multi_queries(question)
            append_to_jsonl(multi_queries_file, {"question": question, "doc_ids": doc_ids_multi_queries})
            print(f"  - Saved Multi-queries result")
        else:
            print(f"  - Skipping Multi-queries (already processed)")

    print("Done")


if __name__ == "__main__":
    main()
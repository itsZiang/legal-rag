import json
import requests


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


def main():
    questions: list[str] = []
    with open("test_set.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        question = item["question"]
        questions.append(question)

    # predictions: list[dict] = []
    predictions_rerank: list[dict] = []
    predictions_hybrid: list[dict] = []
    predictions_multi_queries: list[dict] = []
    
    total_questions = len(questions)
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{total_questions}: {question[:100]}...")
        # doc_ids = get_documents(question)
        # predictions.append({"question": question, "doc_ids": doc_ids})
        doc_ids_rerank = get_documents_rerank(question)
        predictions_rerank.append({"question": question, "doc_ids": doc_ids_rerank})
        doc_ids_hybrid = get_documents_hybrid(question)
        predictions_hybrid.append({"question": question, "doc_ids": doc_ids_hybrid})
        doc_ids_multi_queries = get_documents_multi_queries(question)
        predictions_multi_queries.append({"question": question, "doc_ids": doc_ids_multi_queries})

    # with open("predictions.json", "w", encoding="utf-8") as f:
    #     json.dump(predictions, f, ensure_ascii=False, indent=4)

    with open("predictions_rerank-4113.json", "w", encoding="utf-8") as f:
        json.dump(predictions_rerank, f, ensure_ascii=False, indent=4)

    with open("predictions_hybrid-4113.json", "w", encoding="utf-8") as f:
        json.dump(predictions_hybrid, f, ensure_ascii=False, indent=4)

    with open("predictions_multi_queries-4113.json", "w", encoding="utf-8") as f:
        json.dump(predictions_multi_queries, f, ensure_ascii=False, indent=4)

    print("Done")


if __name__ == "__main__":
    main()

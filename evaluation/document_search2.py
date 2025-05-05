import json
import requests


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

    predictions_multi_queries: list[dict] = []
    for question in questions:
        doc_ids_multi_queries = get_documents_multi_queries(question)
        predictions_multi_queries.append({"question": question, "doc_ids": doc_ids_multi_queries})

    with open("predictions_multi_queries-3300.json", "w", encoding="utf-8") as f:
        json.dump(predictions_multi_queries, f, ensure_ascii=False, indent=4)

    print("Done")


if __name__ == "__main__":
    main()

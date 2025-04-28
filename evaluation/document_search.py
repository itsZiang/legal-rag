import json
import requests


def get_documents(query, limit=5):
    body = {"query": query, "limit": limit}
    response = requests.post("http://localhost:8000/search_ids", json=body)
    return response.json()["ids"]


def main():
    questions: list[str] = []
    with open("test_set.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        question = item["question"]
        questions.append(question)

    predictions: list[dict] = []
    for question in questions:
        doc_ids = get_documents(question)
        predictions.append({"question": question, "doc_ids": doc_ids})

    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print("Done")


if __name__ == "__main__":
    main()

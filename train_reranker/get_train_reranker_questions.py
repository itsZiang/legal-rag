import json

with open("../evaluation/test_set.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_questions = [item["question"] for item in test_data]

with open("not_contains_168.json", "r", encoding="utf-8") as f:
    not_contains_168 = json.load(f)
    
questions = [item["question"].strip().replace("**", "") for item in not_contains_168]

train_reranker_questions = []
for question in questions:
    if question not in test_questions:
        train_reranker_questions.append(question)

with open("train_reranker_questions.json", "w", encoding="utf-8") as f:
    json.dump(train_reranker_questions, f, ensure_ascii=False, indent=4)
import requests
import json


def create_generated_qa():
    with open("ground_truth_qa.json", "r", encoding="utf-8") as f:
        ground_truth_qa = json.load(f)

    # Load existing processed questions
    processed_questions = set()
    try:
        with open("generated_qa_qwen.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                qa = json.loads(line)
                processed_questions.add(qa["question"])
    except FileNotFoundError:
        pass

    for idx, item in enumerate(ground_truth_qa):
        question = item["question"]
        
        # Skip if already processed
        if question in processed_questions:
            print(f"Skipping {idx + 1}/{len(ground_truth_qa)} - already processed")
            continue

        response = requests.post("http://localhost:8000/generate_answer", json={"question": question})
        answer = response.json()["answer"]
        
        # Save immediately after each generation
        with open("generated_qa_qwen.jsonl", "a", encoding="utf-8") as f:
            json.dump({"question": question, "answer": answer}, f, ensure_ascii=False)
            f.write("\n")
            
        print(f"Generated and saved {idx + 1}/{len(ground_truth_qa)}")


if __name__ == "__main__":
    create_generated_qa()

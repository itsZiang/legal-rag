import json


def create_ground_truth_qa():
    with open("../qa_output_filtered (1).json", "r", encoding="utf-8") as f:
        original_qa = json.load(f)
    with open("../retrieve/test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_data = [item["question"].strip().replace("**", "") for item in test_data]
    ground_truth_qa = []
    for item in test_data:
        for qa in original_qa:
            if item == qa["question"].strip().replace("**", ""):
                ground_truth_qa.append({
                    "question": qa["question"].strip().replace("**", ""),
                    "answer": qa["answer"].strip()
                })
        
    with open("ground_truth_qa.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth_qa, f, ensure_ascii=False, indent=4)
    print("Ground truth QA created successfully")

if __name__ == "__main__":
    create_ground_truth_qa()

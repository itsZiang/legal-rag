import json

def compute_recall_at_k(test_data, pred_data, k=5):
    assert len(test_data) == len(pred_data), "Test and prediction data must have the same length"
    total = len(test_data)
    hit = 0
    for ground_truth, prediction in zip(test_data, pred_data):
        ground_truth_docs = set(ground_truth["doc_ids"])
        predicted_docs = set(prediction["doc_ids"][:k])
        if ground_truth_docs & predicted_docs:
            hit += 1
    recall_at_k = hit / total
    return recall_at_k

if __name__ == "__main__":
    test_data = json.load(open("../test_set.json", "r", encoding="utf-8"))
    pred_data = json.load(open("../predictions.json", "r", encoding="utf-8"))
    pred_rerank_data = json.load(open("../predictions_rerank.json", "r", encoding="utf-8"))
    pred_hybrid_data = json.load(open("../predictions_hybrid.json", "r", encoding="utf-8"))
    pred_multi_queries_data = json.load(open("../predictions_multi_queries.json", "r", encoding="utf-8"))

    print("No rerank")
    for k in [1, 3, 5, 10, 15, 20]:
        recall_at_k = compute_recall_at_k(test_data, pred_data, k=k)
        print(f"\tRecall@{k}: {recall_at_k:.4f}")

    print("With rerank")
    for k in [1, 3, 5, 10, 15, 20]:
        recall_at_k = compute_recall_at_k(test_data, pred_rerank_data, k=k)
        print(f"\tRecall@{k}: {recall_at_k:.4f}")
        
    print("With hybrid")
    for k in [1, 3, 5, 10, 15, 20]:
        recall_at_k = compute_recall_at_k(test_data, pred_hybrid_data, k=k)
        print(f"\tRecall@{k}: {recall_at_k:.4f}")
        
    print("With multi queries")
    for k in [1, 3, 5, 10, 15, 20]:
        recall_at_k = compute_recall_at_k(test_data, pred_multi_queries_data, k=k)
        print(f"\tRecall@{k}: {recall_at_k:.4f}")
    
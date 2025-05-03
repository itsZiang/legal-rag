import json

def compute_mrr_at_k(test_data, pred_data, k):
    assert len(test_data) == len(pred_data), "Số lượng mẫu không khớp"

    mrr_total = 0.0

    for gt, pred in zip(test_data, pred_data):
        gt_docs = set(gt['doc_ids'])
        ranked_list = pred['doc_ids'][:k]

        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id in gt_docs:
                mrr_total += 1 / rank
                break

    mrr_at_k = mrr_total / len(test_data)
    return mrr_at_k

def compute_hit_rate_at_k(test_data, pred_data, k):
    assert len(test_data) == len(pred_data), "Số lượng mẫu không khớp"

    hits = 0

    for gt, pred in zip(test_data, pred_data):
        gt_docs = set(gt['doc_ids'])
        ranked_list = pred['doc_ids'][:k]

        if any(doc_id in gt_docs for doc_id in ranked_list):
            hits += 1

    hit_rate = hits / len(test_data)
    return hit_rate


if __name__ == "__main__":
    with open("../test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    with open("../predictions.json", "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    with open("../predictions_rerank.json", "r", encoding="utf-8") as f:
        pred_rerank_data = json.load(f)
        
    with open("../predictions_hybrid.json", "r", encoding="utf-8") as f:
        pred_hybrid_data = json.load(f)
        
    with open("../predictions_multi_queries.json", "r", encoding="utf-8") as f:
        pred_multi_queries_data = json.load(f)

    print("No rerank")
    for k in [1, 3, 5, 10, 15, 20]:
        mrr = compute_mrr_at_k(test_data, pred_data, k)
        print(f"\tMRR@{k}: {mrr:.4f}")

    print("With rerank")
    for k in [1, 3, 5, 10, 15, 20]:
        mrr = compute_mrr_at_k(test_data, pred_rerank_data, k)
        print(f"\tMRR@{k}: {mrr:.4f}")
        
    print("With hybrid")
    for k in [1, 3, 5, 10, 15, 20]:
        mrr = compute_mrr_at_k(test_data, pred_hybrid_data, k)
        print(f"\tMRR@{k}: {mrr:.4f}")
        
    print("With multi queries")
    for k in [1, 3, 5, 10, 15, 20]:
        mrr = compute_mrr_at_k(test_data, pred_multi_queries_data, k)
        print(f"\tMRR@{k}: {mrr:.4f}")    

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

    return mrr_total / len(test_data)


def compute_recall_at_k(test_data, pred_data, k):
    assert len(test_data) == len(pred_data), "Số lượng mẫu không khớp"

    hits = 0
    for gt, pred in zip(test_data, pred_data):
        gt_docs = set(gt['doc_ids'])
        pred_docs = set(pred['doc_ids'][:k])

        if gt_docs & pred_docs:
            hits += 1

    return hits / len(test_data)


def evaluate_all_metrics(name, test_data, pred_data, ks=[1, 3, 5, 10, 15, 20]):
    print(f"\n{name}")
    for k in ks:
        mrr = compute_mrr_at_k(test_data, pred_data, k)
        recall = compute_recall_at_k(test_data, pred_data, k)
        print(f"\t@{k:2d} | MRR: {mrr:.4f} | Recall: {recall:.4f}")


if __name__ == "__main__":
    test_data = json.load(open("../test_set.json", "r", encoding="utf-8"))
    # pred_data = json.load(open("../predictions.json", "r", encoding="utf-8"))
    pred_rerank_data = json.load(open("../predictions_rerank-4113.json", "r", encoding="utf-8"))
    pred_hybrid_data = json.load(open("../predictions_hybrid-4113.json", "r", encoding="utf-8"))
    pred_multi_queries_data = json.load(open("../predictions_multi_queries-4113.json", "r", encoding="utf-8"))

    # evaluate_all_metrics("No rerank", test_data, pred_data)
    evaluate_all_metrics("With rerank", test_data, pred_rerank_data)
    evaluate_all_metrics("With hybrid", test_data, pred_hybrid_data)
    evaluate_all_metrics("With multi queries", test_data, pred_multi_queries_data)

"""
    Evaluation metrics for the retrieval part, including MRR@k, Recall@k.
    Modified to work with JSONL files instead of JSON files.
"""

import json


def load_jsonl(file_path):
    """Load data from a JSONL file into a list of dictionaries"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_mrr_at_k(test_data, pred_data, k):
    assert len(test_data) == len(pred_data), f"Sample count mismatch: {len(test_data)} vs {len(pred_data)}"

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
    assert len(test_data) == len(pred_data), f"Sample count mismatch: {len(test_data)} vs {len(pred_data)}"

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


def sort_by_question(test_data, pred_data):
    """Sort prediction data to match the order of questions in test data"""
    # Create a mapping of questions to their prediction data
    pred_map = {item['question']: item for item in pred_data}
    
    # Create a new sorted list of predictions that matches the test data order
    sorted_preds = []
    for test_item in test_data:
        question = test_item['question']
        if question in pred_map:
            sorted_preds.append(pred_map[question])
        else:
            print(f"Warning: Missing prediction for question: {question[:50]}...")
    
    return sorted_preds


if __name__ == "__main__":
    # Load test data from original JSON file
    test_data = json.load(open("./test_set.json", "r", encoding="utf-8"))
    
    # Load prediction data from JSONL files
    pred_data = load_jsonl("./predictions_bge_v2_m3.jsonl")
    pred_rerank_data = load_jsonl("./predictions_rerank-bge_v2_m3_final.jsonl")
    pred_hybrid_data = load_jsonl("./predictions_hybrid-bge_v2_m3_final.jsonl")
    pred_multi_queries_data = load_jsonl("./predictions_multi_queries-bge_v2_m3.jsonl")
    
    # Ensure the prediction data is sorted in the same order as the test data
    pred_data = sort_by_question(test_data, pred_data)
    pred_rerank_data = sort_by_question(test_data, pred_rerank_data)
    pred_hybrid_data = sort_by_question(test_data, pred_hybrid_data)
    pred_multi_queries_data = sort_by_question(test_data, pred_multi_queries_data)
    
    # Print some basic statistics
    print(f"Test data: {len(test_data)} questions")
    print(f"BGE predictions: {len(pred_data)} questions")
    print(f"Rerank predictions: {len(pred_rerank_data)} questions")
    print(f"Hybrid predictions: {len(pred_hybrid_data)} questions")
    print(f"Multi-queries predictions: {len(pred_multi_queries_data)} questions")
    
    # Run evaluations
    # evaluate_all_metrics("No rerank (BGE)", test_data, pred_data)
    evaluate_all_metrics("With rerank", test_data, pred_rerank_data)
    evaluate_all_metrics("With hybrid", test_data, pred_hybrid_data)
    # evaluate_all_metrics("With multi queries", test_data, pred_multi_queries_data)
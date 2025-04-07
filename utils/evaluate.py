import csv
import argparse

def load_qrels(filepath):
    """讀入 ground truth relevance（標準答案）"""
    qrels = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row['query_id'].strip()
            relevant = row['retrieved_docs'].strip().split()
            qrels[qid] = set(relevant)
    return qrels

def load_predictions(filepath):
    """讀入系統產生的 ranking 結果"""
    predictions = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row['query_id'].strip()
            docs = row['retrieved_docs'].strip().split()
            predictions[qid] = docs
    return predictions

def average_precision(relevant_docs, retrieved_docs):
    """計算單一 query 的 AP"""
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            hits += 1
            sum_precisions += hits / (i + 1)
    if hits == 0:
        return 0.0
    return sum_precisions / len(relevant_docs)

def mean_average_precision(qrels, predictions):
    """計算所有 query 的 MAP"""
    ap_list = []
    for qid in qrels:
        if qid not in predictions:
            print(f"[⚠] Query {qid} 不在預測結果中，略過。")
            continue
        ap = average_precision(qrels[qid], predictions[qid])
        ap_list.append(ap)
    if not ap_list:
        return 0.0
    return sum(ap_list) / len(ap_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--ground_truth", required=True, help="Ground truth CSV file path")
    parser.add_argument("-p", "--prediction", required=True, help="Prediction CSV file path")
    args = parser.parse_args()

    qrels = load_qrels(args.ground_truth)
    predictions = load_predictions(args.prediction)

    map_score = mean_average_precision(qrels, predictions)
    print(f"✅ MAP Score: {map_score:.4f}")

if __name__ == "__main__":
    main()
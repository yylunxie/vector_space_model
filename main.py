import argparse
import numpy as np
import csv
import jieba
import xml.etree.ElementTree as ET
from vsm_model import VSM
from collections import defaultdict
import time

def apply_rocchio_feedback(term_ids, vsm, top_k=10, alpha=1.0, beta=0.75):
    """
    使用 Rocchio Pseudo Relevance Feedback 增強原始查詢向量。

    Args:
        term_ids (List[int]): 原始查詢的 term id list
        vsm (VSM): 初始化過的 VSM 物件，含 doc_term_freq
        top_k (int): 取前幾名文件作為 pseudo relevant
        alpha (float): 原查詢向量的權重
        beta (float): pseudo relevant documents 的平均向量權重

    Returns:
        List[int]: 根據 Rocchio 調整後的加權查詢 term_ids
    """
    # Step 1: 原查詢跑一次 BM25
    initial_scores = compute_bm25(term_ids, vsm.doc_term_freq, vsm.idf, vsm.doc_lens, top_k=top_k)
    
    # Step 2: 取前 top_k 篇 pseudo relevant documents
    pseudo_docs = [doc_id for doc_id, _ in initial_scores[:top_k]]
    
    # Step 3: 計算原始查詢的 term vector
    q_vec = defaultdict(float)
    for tid in term_ids:
        q_vec[tid] += 1.0

    # Step 4: 對 pseudo relevant documents 的 term vector 做平均
    feedback_vec = defaultdict(float)
    for doc_id in pseudo_docs:
        for tid, freq in vsm.doc_term_freq[doc_id].items():
            feedback_vec[tid] += freq
    for tid in feedback_vec:
        feedback_vec[tid] /= top_k

    # Step 5: Rocchio 組合查詢
    new_q_vec = defaultdict(float)
    for tid, val in q_vec.items():
        new_q_vec[tid] += alpha * val
    for tid, val in feedback_vec.items():
        new_q_vec[tid] += beta * val

    # Step 6: 過濾成 term_ids（這裡你可以加門檻條件）
    final_term_ids = [tid for tid, weight in sorted(new_q_vec.items(), key=lambda x: -x[1]) if weight > 0]
    MAX_TERMS = 300
    final_term_ids = final_term_ids[:MAX_TERMS]
    
    return final_term_ids


def write_ranking_output(output_path, query_term_ids, vsm, use_feedback=False):
    """
    將每個 query 的排名結果寫入 CSV 檔案。
    
    Args:
        output_path (str): 要輸出的 CSV 路徑
        query_term_ids (dict): query_id -> list of term_ids
        vsm (VSM): 已初始化的 VSM 模型
        top_k (int): 每個 query 至多輸出前幾名
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "retrieved_docs"])
        for qid, term_ids in query_term_ids.items():
            
            if use_feedback:
                term_ids = apply_rocchio_feedback(term_ids, vsm, top_k=10)
            print(f"[DEBUG] QID: {qid}, term_ids: {len(term_ids)}")
            ranked = compute_bm25(term_ids, vsm.doc_term_freq, vsm.idf, vsm.doc_lens, top_k=100)
            doc_ids = [vsm.doc_list[doc_id].split("/")[-1].lower() for doc_id, _ in ranked]
            writer.writerow([qid, " ".join(doc_ids)])

def preprocess_queries_with_unigram_bigram(query_path, term_to_idx):
    """
    針對 unigram + bigram 的詞表進行 query concepts 的切分與匹配。
    """
    tree = ET.parse(query_path)
    root = tree.getroot()
    query_term_ids = {}

    for topic in root.findall("topic"):
        qid = topic.find("number").text.strip()[-3:]
        concept_text = topic.find("concepts").text.strip()
        tokens = jieba.lcut(concept_text)

        term_ids = []

        # 加入 unigram
        for token in tokens:
            if token in term_to_idx:
                term_ids.append(term_to_idx[token])

        # 加入 bigram
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + tokens[i + 1]
            if bigram in term_to_idx:
                term_ids.append(term_to_idx[bigram])

        if qid == "001":
            print(term_ids)
            
        query_term_ids[qid] = term_ids

    return query_term_ids


def compute_bm25(query, doc_term_freq, idf, doc_lens, k1=1.2, b=0.75, top_k=100):
    print("start compute BM25....")
    avgdl = np.mean(doc_lens)
    scores = {}

    for term_id in query:
        term_idf = idf[term_id]
        
        for doc_id, term_freq_dict in enumerate(doc_term_freq):
            if term_id not in term_freq_dict:
                continue
            freq = term_freq_dict[term_id]
            dl = doc_lens[doc_id]
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score = term_idf * (freq * (k1 + 1)) / denom
            scores[doc_id] = scores.get(doc_id, 0) + score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


def main():
    start_time = time.time()  # ← 開始時間
    parser = argparse.ArgumentParser()
    parser.add_argument("-r",action="store_true",help="Enable relevance feedback")
    parser.add_argument("-i",required=True,help="Input query file")
    parser.add_argument("-o",required=True,help="Output ranked list file")
    parser.add_argument("-m",required=True,help="Model directory path")
    parser.add_argument("-d",required=True,help="NTCIR document directory")
    parser.add_argument("-b",action="store_true",help="Change score as BM25 not similarity")

    args = parser.parse_args()

    print("[INFO] Loading data...")
    t0 = time.time()
    vsm = VSM(args.m)
    print(f"[TIME] VSM loaded in {time.time() - t0:.2f} seconds")

    print("[INFO] Preprocessing queries...")
    t1 = time.time()
    query_term_ids = preprocess_queries_with_unigram_bigram(args.i, vsm.term_to_idx)
    print(f"[TIME] Query preprocessing took {time.time() - t1:.2f} seconds")
    
    t2 = time.time()
    print("[INFO] Ranking with BM25{}...".format(" + Rocchio" if args.r else ""))
    write_ranking_output(args.o, query_term_ids, vsm, use_feedback=args.r)
    print(f"[TIME] Ranking + Output took {time.time() - t2:.2f} seconds")

    print("[INFO] Done. Output saved to", args.o)
    elapsed = time.time() - start_time
    print(f"[INFO] Total execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

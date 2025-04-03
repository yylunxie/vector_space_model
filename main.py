import argparse
import numpy as np
import os
import csv
import jieba
import xml.etree.ElementTree as ET
from vsm_model import VSM

def write_ranking_output(output_path, query_term_ids, vsm, top_k=100):
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
            ranked = compute_bm25(term_ids, vsm.doc_term_freq, vsm.idf, top_k=top_k)
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


def compute_bm25(query, doc_term_freq, idf, k1=1.2, b=0.75, top_k=100):
    avgdl = np.mean([sum(doc.values()) for doc in doc_term_freq])
    scores = {}

    for term_id in query:
        term_idf = idf[term_id]
        
        for doc_id, term_freq_dict in enumerate(doc_term_freq):
            if term_id not in term_freq_dict:
                continue
            freq = term_freq_dict[term_id]
            dl = sum(term_freq_dict.values())
            denom = freq + k1 * (1 - b + b * dl / avgdl)
            score = term_idf * (freq * (k1 + 1)) / denom
            scores[doc_id] = scores.get(doc_id, 0) + score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r",action="store_true",help="Enable relevance feedback")
    parser.add_argument("-i",required=True,help="Input query file")
    parser.add_argument("-o",required=True,help="Output ranked list file")
    parser.add_argument("-m",required=True,help="Model directory path")
    parser.add_argument("-d",required=True,help="NTCIR document directory")
    parser.add_argument("-b",action="store_true",help="Change score as BM25 not similarity")

    args = parser.parse_args()

    print("[INFO] Loading data...")
    vsm = VSM(args.m)

    print("[INFO] Preprocessing queries...")
    query_term_ids = preprocess_queries_with_unigram_bigram(args.i, vsm.term_to_idx)

    print("[INFO] Running BM25 scoring and writing results...")
    write_ranking_output(args.o, query_term_ids, vsm)

    print("[INFO] Done. Output saved to", args.o)

if __name__ == "__main__":
    main()

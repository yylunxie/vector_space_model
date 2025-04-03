import argparse
from utils import load_vocab, load_file_list, load_inverted_index, parse_query_xml
from bm25 import BM25
import os
import csv

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
    vocab = load_vocab(args.vocab)
    file_list = load_file_list(args.file_list)
    inv_index, doc_lens = load_inverted_index(args.inverted_index)
    queries = parse_query_xml(args.query)

    print("[INFO] Initializing BM25...")
    bm25 = BM25(inv_index, doc_lens, len(file_list))

    print("[INFO] Running queries...")
    with open(args.output, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['query_id', 'retrieved_docs'])
        for qid, query_text in queries.items():
            scores = bm25.score(query_text)
            ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
            doc_ids = [file_list[doc_id].split('/')[-1].lower() for doc_id, _ in ranked_docs]
            writer.writerow([qid, ' '.join(doc_ids)])

    print("[INFO] Done. Output written to", args.output)

if __name__ == "__main__":
    main()

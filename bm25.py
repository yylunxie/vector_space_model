import math
import jieba

class BM25:
    def __init__(self, inverted_index, doc_lens, total_docs, k1=1.2, b=0.75):
        self.inverted_index = inverted_index
        self.doc_lens = doc_lens
        self.total_docs = total_docs
        self.avgdl = sum(doc_lens.values()) / total_docs
        self.k1 = k1
        self.b = b

    def idf(self, term):
        df = len(self.inverted_index.get((term, -1), {}))
        return math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query):
        scores = {}
        terms = list(jieba.cut(query))
        for term in terms:
            postings = self.inverted_index.get((term, -1), {})
            idf = self.idf(term)
            for doc_id, freq in postings.items():
                dl = self.doc_lens.get(doc_id, 0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score = idf * (freq * (self.k1 + 1)) / denom
                scores[doc_id] = scores.get(doc_id, 0) + score
        return scores

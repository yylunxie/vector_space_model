import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
import pickle
import os


class VSM:
    _instance = None  # Singleton 實例
    
    def __new__(cls, model_dir):
        if cls._instance is None:
            cls._instance = super(VSM, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_dir):
        if self._initialized:
            return 
        
        """ Model Path """
        self.file_list = os.path.join(model_dir, "file-list")
        self.vocab_file = os.path.join(model_dir, "vocab.all")
        self.inverted_file = os.path.join(model_dir, "inverted-file")
        
        """ Saved Model File Path """
        self.saved_file_list = os.path.join(model_dir, "file-list.npy")
        self.saved_term_to_idx = os.path.join(model_dir, "term_to_idx.pkl")
        self.saved_idf_vector = os.path.join(model_dir, "idf.npy")
        self.saved_doc_term_freq = os.path.join(model_dir, "doc_term_freq.pkl")
        self.saved_doc_lens = os.path.join(model_dir, "doc_lens.npy")
        
        #Validation model files
        if not os.path.exists(self.inverted_file):
            raise ValueError(f"Inverted file 不存在，請檢查文件路徑是否在 {self.inverted_file}")
        if not os.path.exists(self.vocab_file):
            raise ValueError(f"Vocab file 不存在，請檢查文件路徑是否在 {self.vocab_file}")
        if not os.path.exists(self.file_list):
            raise ValueError(f"File list 不存在，請檢查文件路徑是否在 {self.file_list}")
        
        if not os.path.exists(self.saved_file_list) or \
            not os.path.exists(self.saved_term_to_idx) or \
            not os.path.exists(self.saved_idf_vector) or \
            not os.path.exists(self.saved_doc_term_freq) or \
            not os.path.exists(self.saved_doc_lens):
                print("Generating files...")
                self.doc_list, \
                self.term_to_idx, \
                self.idf, \
                self.doc_term_freq, \
                self.doc_lens = self._load_model()
                print("Done!")
        else:
            print("Loading files....")
            self.doc_list = self._load_npy(self.saved_file_list)
            self.term_to_idx = self._load_pickle(self.saved_term_to_idx)
            self.idf = self._load_npy(self.saved_idf_vector)
            self.doc_term_freq = self._load_pickle(self.saved_doc_term_freq)
            self.doc_lens = self._load_npy(self.saved_doc_lens)
            print("Done!")
            
        self._initialized = True
        
    def _load_pickle(self, path):
        with open(path, "rb") as f:
            try:
                target = pickle.load(f)
            except Exception as e:
                raise ValueError(f"Loading pkl failed: {e}")
            return target
        
    def _load_npy(self, path):
        try:
            target =  np.load(path)
        except Exception as e:
            raise ValueError(f"Loading npy failed: {e}")
        return target

    def _load_npz(self,path):
        try:
            target =  load_npz(path)
        except Exception as e:
            raise ValueError(f"Loading npz failed: {e}")
        return target
    
    def _load_model(self):
        
        # File list
        with open(self.file_list, "r", encoding="utf-8") as f:
            
            doc_list = [line.strip() for line in f.readlines()]
        
        # Vocab 
        with open(self.vocab_file,'r',encoding='UTF-8') as file:
            vocab = [line.strip() for line in file.readlines()]
            
        # Read inverted files
        with open(self.inverted_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Build term to index
        term_to_idx = {}
        doc_term_freq = [{} for _ in range(len(doc_list))]  # term 出現在 doc 的 freq，
        i = 0
        while i < len(lines):
            v1, v2, N = map(int, lines[i].split())
            if v2 != -1:
                # Bigram
                term = vocab[v1] + vocab[v2]
            else:
                term = vocab[v1]
                
            if term not in term_to_idx:
                idx = len(term_to_idx)
                term_to_idx[term] = idx
                
            idx = term_to_idx[term]

            for _ in range(N):
                i += 1
                doc_id, count = map(int, lines[i].split())
                doc_term_freq[doc_id][idx] = count

            i += 1
            
        with open(self.saved_term_to_idx, "wb") as f:
            pickle.dump(term_to_idx, f)
        np.save(self.saved_file_list, np.array(doc_list))
        
        """ Compute IDF vector """
        
        doc_count = len(doc_term_freq)
        term_size = len(term_to_idx)
        self.doc_count = doc_count
        
        # TF matrix
        tf_matrix = lil_matrix((doc_count, term_size), dtype=np.float32)
        for doc_id, term_freq in enumerate(doc_term_freq):
            for term_id, freq in term_freq.items():
                tf_matrix[doc_id, term_id] = freq
                
        tf_csr = tf_matrix.tocsr()
        df = np.array((tf_csr > 0).sum(axis=0)).flatten()
        
        idf_vector = np.log((doc_count - df + 0.5) / (df + 0.5) + 1)
        doc_lens = np.array([sum(doc.values()) for doc in doc_term_freq])
        np.save(self.saved_doc_lens, doc_lens)
        
        np.save(self.saved_idf_vector, idf_vector)
        with open(self.saved_doc_term_freq, "wb") as f:
            pickle.dump(doc_term_freq, f)
        
        return doc_list, term_to_idx, idf_vector, doc_term_freq, doc_lens

if __name__ == "__main__":
    vsm = VSM("model")
    print(vsm.doc_count)
        
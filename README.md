# Vector Space Model

## Folder structure
```plain-text
r13946024_PA1/
│
├── README.md                     ← 說明檔：執行方式、說明預處理資料等
├── main.py                       ← 主程式入口，負責讀入模型並產生 ranking
├── vsm_model.py                  ← 含 VSM 類別與模型初始化、載入、處理邏輯
├── utils.py                      ← 通用功能：Query 處理、輸出 ranking、評估
├── execute.sh                    ← 執行主程式
│
├── model/                        ← 儲存前處理產生的模型檔案（可上傳）
│   ├── file-list.npy
│   ├── term_to_idx.pkl
│   ├── idf.npy
│   ├── doc_term_freq.pkl
│   ├── doc_lens.npy
│   └── posting_list.pkl
│
└── output/                      ← 輸出結果檔儲存路徑


```

## Model

We use BM25 + Rocchio Relavent Feedback here.

### Preprocessed Files
為了節省執行時間，以下檔案已預先產生：
- `model/term_to_idx.pkl`: 詞彙表對應 index
- `model/posting_list.pkl`: 倒排索引
- `model/idf.npy`: 每個詞的 IDF 值
- `model/doc_term_freq.pkl`: 每篇文件的 term frequency
- `model/doc_lens.npy`: 每篇文件的長度


## How to Run


```bash
sh ./script/execute.sh -m model -i ./queries/query-test.xml -o ./output/ranking-test-v1.csv -d ./CIRB010/
```
- `-r` 可選，啟用 Rocchio Feedback

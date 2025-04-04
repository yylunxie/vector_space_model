# Vector Space Model

## Model

We use BM25 here.

## Execute

To execute w/o Rocchio Relevance Feedback, run this

```bash
# train
sh ./script/execute.sh -m model -i ./queries/query-train.xml -o ./output/ranking-train-v1.csv -d ./CIRB010/

# test
sh ./script/execute.sh -m model -i ./queries/query-test.xml -o ./output/ranking-test-v1.csv -d ./CIRB010/
```

To execute w/ Rocchio Relevance Feedback, run this

```bash
# train
sh ./script/execute.sh -m model -i ./queries/query-train.xml -o ./output/ranking-train-v1.csv -d ./CIRB010/ -r

# train
sh ./script/execute.sh -m model -i ./queries/query-test.xml -o ./output/ranking-test-v1.csv -d ./CIRB010/ -r
```

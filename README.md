# ReSys_KGARA

This is our implementation for the paper:

Yihao Zhang, Meng Yuan, Chu Zhao, Mian Chen and Xiaoyang Liu. Aggregating Knowledge-aware Graph Neural Network and Adaptive Relational Attention for Recommendation.

## Environment Settings

 tensorflow = 1.12
 numpy = 1.18

## Example to run the codes.

```
python preprocess.py 

python deal_data.py

python main.py
```

## Dataset

We provide three processed datasets: MovieLens 1 Million (ml-1m), LastFM, and BookCrossing. 

kg.txt: 
- knowledge graph file;

item_index2entity_id.txt: 
- the mapping from item indices in the raw rating file 

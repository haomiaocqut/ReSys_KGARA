# ReSys_KGARA

This is our implementation for the paper:

Yihao Zhanga, Meng Yuan, Chu Zhao, Mian Chen and Xiaoyang Liu. Aggregating Knowledge-aware Graph Neural Network and Adaptive Relational Attention for Recommendation.

## Environment Settings

 tensorflow = 1.12
 numpy = 1.18

## Example to run the codes.

```
python main.py
```

## Dataset

We provide three processed datasets: MovieLens 1 Million (ml-1m), LastFM, and BookCrossing. 

kg.txt: knowledge graph file;

item_index2entity_id.txt: the mapping from item indices in the raw rating file to entity IDs in the KG;

ratings_final.txt: user item interaction file

train.txt: train file, each Line is a training instance: userID\t itemID\t rating

test.txt: test file (positive instances), each Line is a testing instance: userID\t itemID\t rating

test_negative.txt:  Test file (negative instances), each line corresponds to the line of test.rating, containing 100 negative samples.  
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...
   
   

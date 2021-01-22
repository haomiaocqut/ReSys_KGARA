import tensorflow as tf
import numpy as np
import math
from model import KGCN
#from evaluate import evaluate_model
from time import time
evaluation_threads = 1

verbose = 1
def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, test_data = data[4], data[5],
    adj_entity, adj_relation = data[6], data[7]
    testRatings, testNegatives = data[8], data[9]
    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)
    # top-K evaluation settings
    user_list, train_record, test_record, item_set = topk_settings(show_topk, train_data, test_data, n_item)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                #if show_loss:
            #print(loss)

            # top-K evaluation
            if show_topk:

                hr_20, ndcg_20 = topk_eval(
                    sess, model, user_list, testNegatives, testRatings, args.batch_size)

                print('epoch: %d   ' % step, end ='')
                print('recall@10: ', end='')
                #for i in recall:
                    #print('%.4f\t' % i, end='')
                print('\t',   hr_20, end='')

                print('\t')

                print('ndcg@10: ', end='')
                #for i in ndcg:
                    #print('%.4f\t' % i, end='')
                print('\t' , ndcg_20,end='')
                print('\n')


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))

        item_set = set(list(range(n_item))) # item_set : [0,1,2,3,...n_item-1]
        return user_list, train_record, test_record, item_set

    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0


def topk_eval(sess, model, user_list,  testNegatives, testRatings, batch_size):
    #precision_list = {k: [] for k in k_list}
    #recall_list = {k: [] for k in k_list}
    #ndcg_list = {k: [] for k in k_list}

    hits_20, ndcgs_20 = [], []
    for user in range(len(user_list)):
        #test_item_list = list(item_set - train_record[user])
        test_item_list = list(testNegatives[user])
        gtItem = testRatings[user][1]
        test_item_list.append(gtItem)
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)

        item_sorted = [i[0] for i in item_score_pair_sorted]



        ranklist_20 = item_sorted[:20]


        hr20 = getHitRatio(ranklist_20, gtItem)
        ndcg20 = getNDCG(ranklist_20, gtItem)


        hits_20.append(hr20)
        ndcgs_20.append(ndcg20)

    hr_20, ndcg_20 = np.array(hits_20).mean(), np.array(ndcgs_20).mean()
    return hr_20, ndcg_20



def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
'''
def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(np.sort(relevance)[::-1])

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg

def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)
'''




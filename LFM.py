#-*-coding:utf-8-*-
"""
author:jamest
date:20190306
LFM function
"""
import math
import pandas as pd
import random
import numpy as np
import pickle

class LFM:
    def __init__(self, user_ids, item_ids):
        self.class_count = 5
        self.iter_count = 5
        self.lr = 0.02
        self.lambd = 0.01


        self._init_data(user_ids, item_ids)
        self._init_model()

    def _randomSelectNegativeSample(self,user_id,user_ids,item_ids):
        items = [x[1] for x in zip(user_ids,item_ids) if x[0]==user_id]
        res = dict()
        for i in items:
            res[i] = 1
        n = 0
        for i in range(len(items) * 3):
            item = item_ids[random.randint(0, len(item_ids) - 1)]
            if item in res:
                continue
            res[item] = 0
            n += 1
            if n > len(items):
                break
        return res


    def _get_dic(self,user_ids,item_ids):
        items_dict = {}
        for user_id in self.user_ids_set:
            items_dict[user_id] = self._randomSelectNegativeSample(user_id,user_ids,item_ids)
        return items_dict



    def _init_data(self,user_ids,item_ids):
        self.user_ids_set = set(user_ids)
        self.item_ids_set = set(item_ids)
        self.items_dict = self._get_dic(user_ids,item_ids)




    def _init_model(self):
        """
        Get corpus and initialize model params.
        """
        array_p = np.random.randn(len(self.user_ids_set), self.class_count)
        array_q = np.random.randn(len(self.item_ids_set), self.class_count)
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids_set))
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids_set))

    def _predict(self, user_id, item_id):
        """
        Calculate interest between user_id and item_id.
        p is the look-up-table for user's interest of each class.
        q means the probability of each item being classified as each class.
        """
        p = np.mat(self.p.ix[user_id].values)
        q = np.mat(self.q.ix[item_id].values).T
        r = (p * q).sum()
        # logit = 1.0 / (1 + math.exp(-r))
        logit = self._sigmoid(r)
        return logit


    def _sigmoid(self,z):
        return 1./(1 + np.exp(-z))

    def _loss(self, user_id, item_id, y, step):
        """
        Loss Function define as MSE, the code write here not that formula you think.
        """
        e = y - self._predict(user_id, item_id)
        print('Step: {}, user_id: {}, item_id: {}, y: {}, loss: {}'.
              format(step, user_id, item_id, y, e))
        return e

    def _optimize(self, user_id, item_id, e):
        """
        Use SGD as optimizer, with L2 p, q square regular.
        e.g: E = 1/2 * (y - predict)^2, predict = matrix_p * matrix_q
             derivation(E, p) = -matrix_q*(y - predict), derivation(E, q) = -matrix_p*(y - predict),
             derivation（l2_square，p) = lam * p, derivation（l2_square, q) = lam * q
             delta_p = lr * (derivation(E, p) + derivation（l2_square，p))
             delta_q = lr * (derivation(E, q) + derivation（l2_square, q))
        """
        gradient_p = -e * self.q.ix[item_id].values
        l2_p = self.lambd * self.p.ix[user_id].values
        delta_p = self.lr * (gradient_p + l2_p)

        gradient_q = -e * self.p.ix[user_id].values
        l2_q = self.lambd * self.q.ix[item_id].values
        delta_q = self.lr * (gradient_q + l2_q)

        self.p.loc[user_id] -= delta_p
        self.q.loc[item_id] -= delta_q

    def train(self):

        for step in range(self.iter_count):
            for user_id, item_dict in self.items_dict.items():
                item_ids = list(item_dict.keys())
                random.shuffle(item_ids)
                for item_id in item_ids:
                    e = self._loss(user_id, item_id, item_dict[item_id], step)
                    self._optimize(user_id, item_id, e)
            self.lr *= 0.9
        self.save()

    def predict(self, user_id, items,top_n=10):
        """
        Calculate all item user have not meet before and return the top n interest items.
        """
        self.load()
        user_item_ids = set(items)
        other_item_ids = self.item_ids_set ^ user_item_ids
        interest_list = [self._predict(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(list(other_item_ids), interest_list), key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def save(self):
        """
        Save model params.
        """
        f = open('data/lfm.model', 'wb')
        pickle.dump((self.p, self.q), f)
        f.close()

    def load(self):
        """
        Load model params.
        """
        f = open('data/lfm.model', 'rb')
        self.p, self.q = pickle.load(f)
        f.close()



if __name__ == '__main__':
    moviesPath = './data/ml-1m/movies.dat'
    ratingsPath = './data/ml-1m/ratings.dat'
    usersPath = './data/ml-1m/users.dat'

    # usersDF = pd.read_csv(usersPath,index_col=None,sep='::',header=None,names=['user_id', 'gender', 'age', 'occupation', 'zip'])
    # moviesDF = pd.read_csv(moviesPath,index_col=None,sep='::',header=None,names=['movie_id', 'title', 'genres'])
    ratingsDF = pd.read_csv(ratingsPath, index_col=None, sep='::', header=None,names=['user_id', 'movie_id', 'rating', 'timestamp'])

    X=ratingsDF['user_id'][:1000]
    Y=ratingsDF['movie_id'][:1000]

    LFM(X,Y).train()
    # print('ItemCF result',rank)












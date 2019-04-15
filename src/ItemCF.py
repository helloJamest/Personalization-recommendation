#-*-coding:utf-8-*-
"""
author:jamest
date:20190306
ItemCF function
"""
import math
import pandas as pd


class ItemCF:
    def __init__(self,X,y):
        self.X,self.y = X,y

    def recommend(self,userID,K,N,useIUF):
        """
        Args:
            userID:user id
            k: K items closest to the user's items
            N:the number of recommendable item
            useIUF:whether or not use useIUF
        Returns:
            top N recommendation
            rank:[(item_id1,interest1),(item_id2,interest2)...]
        """
        W, user_item = self._ItemSimilarity(self.X, self.y, useIUF)
        rank = {}
        interacted_items = user_item[userID]
        for i in interacted_items:
            for j, wij in sorted(W[i].items(), reverse=True)[0:K]:
                if j not in interacted_items:
                    rank.setdefault(j, 0)
                    rank[j] += wij
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:N]

    def _ItemSimilarity(self,X,Y,useIUF=False):
        """
        Args:
            X: user id list
            Y: item id list
            useIUF: whether or not use useIUF
        Returns:
            W : item's correlation
            user_item: a dict:{user_id1:[item1,item2,...],..user_idn:[]}
        """
        # 建立倒排表
        user_item = dict()
        for i in range(Y.count()):
            user = X.iloc[i]
            item = Y.iloc[i]
            if user not in user_item:
                user_item[user] = set()
            user_item[user].add(item)

        C = {}
        N = {}
        for u, items in user_item.items():
            for i in items:
                N.setdefault(i, 0)
                N[i] += 1
                C.setdefault(i, {})
                for j in items:
                    if i == j:
                        continue
                    C[i].setdefault(j, 0)
                    if not useIUF:
                        C[i][j] += 1
                    else:
                        C[i][j] += 1 / math.log(1 + len(items))  # 对活跃用户做了一种软性的惩罚
        W = C.copy()
        for i, related_items in C.items():
            for j, cij in related_items.items():
                W[i][j] = cij / math.sqrt(N[i] * N[j])
        return W, user_item



if __name__ == '__main__':
    moviesPath = '../data/ml-1m/movies.dat'
    ratingsPath = '../data/ml-1m/ratings.dat'
    usersPath = '../data/ml-1m/users.dat'

    # usersDF = pd.read_csv(usersPath,index_col=None,sep='::',header=None,names=['user_id', 'gender', 'age', 'occupation', 'zip'])
    # moviesDF = pd.read_csv(moviesPath,index_col=None,sep='::',header=None,names=['movie_id', 'title', 'genres'])
    ratingsDF = pd.read_csv(ratingsPath, index_col=None, sep='::', header=None,names=['user_id', 'movie_id', 'rating', 'timestamp'])
    X=ratingsDF['user_id'][:10000]
    Y=ratingsDF['movie_id'][:10000]
    rank = ItemCF(X,Y).recommend(1,K=10,N=10,useIUF=True)#输出对用户1推荐的 top10 item
    print('ItemCF result',rank)











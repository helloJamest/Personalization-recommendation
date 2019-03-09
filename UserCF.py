#-*-coding:utf-8-*-
"""
author:jamest
date:20190306
UserCF function
"""
import math
import pandas as pd


class UserCF:
    def __init__(self,X,y):
        self.X,self.y = X,y

    def recommend(self,userID,K,N,useIIF):
        """
        Args:
            userID:user id
            k: K users closest to the user's interest
            N:the number of recommendable item
            userIIF:whether or not use userIIF
        Returns:
            top N recommendation
            rank:[(item_id1,interest1),(item_id2,interest2)...]
        """
        W, user_item = self._UserSimilarity(self.X, self.y, useIIF)
        rank = {}
        interacted_items = user_item[userID]
        for v, wuv in sorted(W[userID].items(), reverse=True)[:K]:
            for i in user_item[v]:
                if i not in interacted_items:
                    rank.setdefault(i, 0)
                    rank[i] += wuv
        return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:N]

    def _UserSimilarity(self,X,Y,useIIF=False):
        item_user=dict()
        for i in range(X.count()):
            user=X.iloc[i]
            item=Y.iloc[i]
            if item not in item_user:
                item_user[item]=set()
            item_user[item].add(user)

        user_item=dict()
        for i in range(Y.count()):
            user=X.iloc[i]
            item=Y.iloc[i]
            if user not in user_item:
                user_item[user]=set()
            user_item[user].add(item)

        C={}
        N={}
        for i,users in item_user.items():
            for u in users:
                N.setdefault(u,0)
                N[u]+=1
                C.setdefault(u,{})
                for v in users:
                    if u==v:
                        continue
                    C[u].setdefault(v,0)
                    if not useIIF:
                        C[u][v]+=1
                    else:
                        C[u][v]+=1 / math.log(1 + len(users))#惩罚用户u和用户v共同兴趣列表中热门物品
        W=C.copy()
        for u,related_users in C.items():
            for v,cuv in related_users.items():
                W[u][v]=cuv/math.sqrt(N[u]*N[v])
        return W,user_item



if __name__ == '__main__':
    moviesPath = './data/ml-1m/movies.dat'
    ratingsPath = './data/ml-1m/ratings.dat'
    usersPath = './data/ml-1m/users.dat'

    # usersDF = pd.read_csv(usersPath,index_col=None,sep='::',header=None,names=['user_id', 'gender', 'age', 'occupation', 'zip'])
    # moviesDF = pd.read_csv(moviesPath,index_col=None,sep='::',header=None,names=['movie_id', 'title', 'genres'])
    ratingsDF = pd.read_csv(ratingsPath, index_col=None, sep='::', header=None,names=['user_id', 'movie_id', 'rating', 'timestamp'])
    X=ratingsDF['user_id'][:100000]
    Y=ratingsDF['movie_id'][:100000]
    rank = UserCF(X,Y).recommend(1,K=10,N=10,useIIF=True)#输出对用户1推荐的 top10 item
    print('UserCF result',rank)











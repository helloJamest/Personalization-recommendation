#-*-coding:utf-8-*-
"""
author:jamest
date:20190310
PersonalRank function #without MF
"""
import pandas as pd
import time

class PersonalRank:
    def __init__(self,X,Y):
        X,Y = ['user_'+str(x) for x in X],['item_'+str(y) for y in Y]
        self.G = self.get_graph(X,Y)

    def get_graph(self,X,Y):
        """
        Args:
            X: user id
            Y: item id
        Returns:
            graph:dic['user_id1':{'item_id1':1},  ... ]
        """
        item_user = dict()
        for i in range(len(X)):
            user = X[i]
            item = Y[i]
            if item not in item_user:
                item_user[item] = {}
            item_user[item][user]=1

        user_item = dict()
        for i in range(len(Y)):
            user = X[i]
            item = Y[i]
            if user not in user_item:
                user_item[user] = {}
            user_item[user][item]=1
        G = dict(item_user,**user_item)
        return G


    def recommend(self, alpha, userID, max_depth,K=10):
        # rank = dict()
        userID = 'user_' + str(userID)
        rank = {x: 0 for x in self.G.keys()}
        rank[userID] = 1
        # 开始迭代
        begin = time.time()
        for k in range(max_depth):
            tmp = {x: 0 for x in self.G.keys()}
            # 取出节点i和他的出边尾节点集合ri
            for i, ri in self.G.items():
                # 取节点i的出边的尾节点j以及边E(i,j)的权重wij,边的权重都为1，归一化后就是1/len(ri)
                for j, wij in ri.items():
                    tmp[j] += alpha * rank[i] / (1.0 * len(ri))
            tmp[userID] += (1 - alpha)
            rank = tmp
        end = time.time()
        print('use_time', end - begin)
        lst = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:K]
        for ele in lst:
            print("%s:%.3f, \t" % (ele[0], ele[1]))

if __name__ == '__main__':
    moviesPath = '../data/ml-1m/movies.dat'
    ratingsPath = '../data/ml-1m/ratings.dat'
    usersPath = '../data/ml-1m/users.dat'

    # usersDF = pd.read_csv(usersPath,index_col=None,sep='::',header=None,names=['user_id', 'gender', 'age', 'occupation', 'zip'])
    # moviesDF = pd.read_csv(moviesPath,index_col=None,sep='::',header=None,names=['movie_id', 'title', 'genres'])
    ratingsDF = pd.read_csv(ratingsPath, index_col=None, sep='::', header=None,names=['user_id', 'movie_id', 'rating', 'timestamp'])
    X=ratingsDF['user_id'][:1000]
    Y=ratingsDF['movie_id'][:1000]
    PersonalRank(X,Y).recommend(alpha=0.8,userID=1,max_depth=50,K=30)#输出对用户1推荐的 top10 item
    # print('PersonalRank result',rank)









#-*-coding:utf-8-*-
"""
author:jamest
date:20190310
PersonalRank function with Matrix
"""
import pandas as pd
import numpy as np
import time
import operator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import gmres


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


    def graph_to_m(self):
        """
        Returns:
            a coo_matrix sparse mat M
            a list,total user item points
            a dict,map all the point to row index
        """

        graph = self.G
        vertex = list(graph.keys())
        address_dict = {}
        total_len = len(vertex)
        for index in range(len(vertex)):
            address_dict[vertex[index]] = index
        row = []
        col = []
        data = []
        for element_i in graph:
            weight = round(1/len(graph[element_i]),3)
            row_index=  address_dict[element_i]
            for element_j in graph[element_i]:
                col_index = address_dict[element_j]
                row.append(row_index)
                col.append(col_index)
                data.append(weight)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        m = coo_matrix((data,(row,col)),shape=(total_len,total_len))
        return m,vertex,address_dict


    def mat_all_point(self,m_mat,vertex,alpha):
        """
        get E-alpha*m_mat.T
        Args:
            m_mat
            vertex:total item and user points
            alpha:the prob for random walking
        Returns:
            a sparse
        """
        total_len = len(vertex)
        row = []
        col = []
        data = []
        for index in range(total_len):
            row.append(index)
            col.append(index)
            data.append(1)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        eye_t = coo_matrix((data,(row,col)),shape=(total_len,total_len))
        return eye_t.tocsr()-alpha*m_mat.tocsr().transpose()

    def recommend_use_matrix(self, alpha, userID, K=10,use_matrix=True):
        """
        Args:
            alpha:the prob for random walking
            userID:the user to recom
            K:recom item num

        Returns:
            a dic,key:itemid ,value:pr score
        """
        m, vertex, address_dict = self.graph_to_m()
        userID = 'user_' + str(userID)
        print('add',address_dict)
        if userID not in address_dict:
            return []
        score_dict = {}
        recom_dict = {}
        mat_all = self.mat_all_point(m,vertex,alpha)
        index = address_dict[userID]
        initial_list = [[0] for row in range(len(vertex))]
        initial_list[index] = [1]
        r_zero = np.array(initial_list)
        res = gmres(mat_all,r_zero,tol=1e-8)[0]
        for index in range(len(res)):
            point = vertex[index]
            if len(point.strip().split('_'))<2:
                continue
            if point in self.G[userID]:
                continue
            score_dict[point] = round(res[index],3)
        for zuhe in sorted(score_dict.items(),key=operator.itemgetter(1),reverse=True)[:K]:
            point,score = zuhe[0],zuhe[1]
            recom_dict[point] = score
        return recom_dict




if __name__ == '__main__':
    moviesPath = '../data/ml-1m/movies.dat'
    ratingsPath = '../data/ml-1m/ratings.dat'
    usersPath = '../data/ml-1m/users.dat'

    # usersDF = pd.read_csv(usersPath,index_col=None,sep='::',header=None,names=['user_id', 'gender', 'age', 'occupation', 'zip'])
    # moviesDF = pd.read_csv(moviesPath,index_col=None,sep='::',header=None,names=['movie_id', 'title', 'genres'])
    ratingsDF = pd.read_csv(ratingsPath, index_col=None, sep='::', header=None,names=['user_id', 'movie_id', 'rating', 'timestamp'])
    X=ratingsDF['user_id'][:1000]
    Y=ratingsDF['movie_id'][:1000]
    rank = PersonalRank(X,Y).recommend_use_matrix(alpha=0.8,userID=1,K=30)
    print('PersonalRank result',rank)









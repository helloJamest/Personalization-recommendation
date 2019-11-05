#-*-coding:utf-8-*-
"""
author:jamest
date:20190405
CBOW function
"""
import pandas as pd
from gensim.models import Word2Vec
import multiprocessing
import os

class CBOW:
    def __init__(self,input_file):
        self.model = self.get_train_data(input_file)

    def get_train_data(self,input_file,L=100):
        if not os.path.exists(input_file):
            return
        score_thr = 4.0
        ratingsDF = pd.read_csv(input_file, index_col=None, sep='::', header=None,
                                names=['user_id', 'movie_id', 'rating', 'timestamp'])
        ratingsDF = ratingsDF[ratingsDF['rating']>score_thr]
        ratingsDF['movie_id'] = ratingsDF['movie_id'].apply(str)
        movie_list = ratingsDF.groupby('user_id')['movie_id'].apply(list).values
        print('training...')
        model = Word2Vec(movie_list, size=L, window=5, sg=0, hs=0, min_count=1, workers=multiprocessing.cpu_count(),
                         iter=10)
        return model

    def recommend(self,movieId,K):
        """
         Args:
             movieId:the movieId to find similar
             K:recom item num

         Returns:
             a dic,key:itemid ,value:sim score
         """
        movieId = str(movieId)
        rank = self.model.most_similar(movieId,topn=K)
        return rank

if __name__ == '__main__':
    moviesPath = '../data/ml-1m/movies.dat'
    ratingsPath = '../data/ml-1m/ratings.dat'
    usersPath = '../data/ml-1m/users.dat'

    rank = CBOW(ratingsPath).recommend(movieId=1,K=30)
    print('CBOW result',rank)









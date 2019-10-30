#-*-coding:utf-8-*-
"""
author:jamest
date:20190405
content based function
"""
import pandas as pd
import numpy as np
import time
import os

class contentBased:
    def __init__(self,rating_file,item_file):
        if not os.path.exists(rating_file) or not os.path.exists(item_file):
            print('the file not exists')
            return
        self.moviesDF = pd.read_csv(item_file, index_col=None, sep='::', header=None, names=['movie_id', 'title', 'genres'])
        self.ratingsDF = pd.read_csv(rating_file, index_col=None, sep='::', header=None,
                                names=['user_id', 'movie_id', 'rating', 'timestamp'])
        self.item_cate, self.cate_item = self.get_item_cate()
        self.up = self.get_up()

    def get_item_cate(self,topK = 10):
        """
         Args:
             topK:nums of items in cate_item
         Returns:
             item_cate：a dic,key:itemid ,value:ratio
             cate_item：a dic:key:cate vale:[item1,item2,item3]
         """
        movie_rating_avg = self.ratingsDF.groupby('movie_id')['rating'].agg({'item_ratings_mean': np.mean}).reset_index()
        movie_rating_avg.head()
        items = movie_rating_avg['movie_id'].values
        scores = movie_rating_avg['item_ratings_mean'].values

        #得到item的平均评分
        item_score_veg = {}
        for item, score in zip(items, scores):
            item_score_veg[item] = score

        #得到item中不同种类的得分
        item_cate = {}
        items = self.moviesDF['movie_id'].values
        genres = self.moviesDF['genres'].apply(lambda x: x.split('|')).values
        for item, genres_lis in zip(items, genres):
            radio = 1 / len(genres_lis)
            item_cate[item] = {}
            for genre in genres_lis:
                item_cate[item][genre] = radio

        recode = {}
        for item in item_cate:
            for genre in item_cate[item]:
                if genre not in recode:
                    recode[genre] = {}
                recode[genre][item] = item_score_veg.get(item, 0)

        # 不同种类item的倒排
        cate_item = {}
        for cate in recode:
            if cate not in cate_item:
                cate_item[cate] = []
            for zuhe in sorted(recode[cate].items(), key=lambda x: x[1], reverse=True)[:topK]:
                cate_item[cate].append(zuhe[0])

        return item_cate, cate_item


    def get_time_score(self,timestamp,fix_time_stamp):
        """
         Args:
             timestamp:the timestamp of user-item
             fix_time_stamp:the max timestamp of the timestamps
         Returns:
             a time_score:fixed range in (0,1]
         """
        total_sec = 24*60*60
        delta = (fix_time_stamp-timestamp)/total_sec/100
        return round(1/(1+delta),3)

    def get_up(self,score_thr=4.0,topK=5):
        """
         Args:
             score_thr:select the score>=score_thr of ratingsDF
             topK:the number of item in up
         Returns:
             a dic,key:userid ,value[(category,ratio),(category1,ratio1)]
         """
        ratingsDF = self.ratingsDF[self.ratingsDF['rating'] > score_thr]
        fix_time_stamp = ratingsDF['timestamp'].max()
        ratingsDF['time_score'] = ratingsDF['timestamp'].apply(lambda x: self.get_time_score(x,fix_time_stamp))

        users = ratingsDF['user_id'].values
        items = ratingsDF['movie_id'].values
        ratings = ratingsDF['rating'].values
        scores = ratingsDF['time_score'].values

        recode = {}
        up = {}
        for userid, itemid, rating, time_score in zip(users, items, ratings, scores):
            if userid not in recode:
                recode[userid] = {}

            for cate in self.item_cate[itemid]:
                if cate not in recode[userid]:
                    recode[userid][cate] = 0
                recode[userid][cate] += rating * time_score * self.item_cate[itemid][cate]
        for userid in recode:
            if userid not in up:
                up[userid] = []
            total_score = 0
            for zuhe in sorted(recode[userid].items(), key=lambda x: x[1], reverse=True)[:topK]:
                up[userid].append((zuhe[0], zuhe[1]))
                total_score += zuhe[1]
            for index in range(len(up[userid])):
                up[userid][index] = (up[userid][index][0], round(up[userid][index][1] / total_score, 3))
        return up


    def recommend(self, userID, K=10):
        """
         Args:
             userID: the user to recom
             K: the num of recom item

         Returns:
             a dic,key:userID ,value:recommend itemid
         """
        if userID not in self.up:
            return
        recom_res = {}
        if userID not in recom_res:
            recom_res[userID] = []

        for zuhe in self.up[userID]:
            cate, ratio = zuhe
            num = int(K * ratio) + 1
            if cate not in self.cate_item:
                continue
            rec_list = self.cate_item[cate][:num]
            recom_res[userID] += rec_list
        return recom_res

if __name__ == '__main__':
    moviesPath = '../data/ml-1m/movies.dat'
    ratingsPath = '../data/ml-1m/ratings.dat'
    usersPath = '../data/ml-1m/users.dat'
    recom_res = contentBased(ratingsPath,moviesPath).recommend(userID=1,K=30)
    print('content based result',recom_res)









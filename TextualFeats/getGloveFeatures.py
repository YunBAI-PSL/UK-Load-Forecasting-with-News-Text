#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:15:18 2022

@author: yunbai
"""

import pandas as pd
import numpy as np
import pickle
from get_topics import build_dictionary
import os 
os.getcwd()

#load GloVe pretrained model
def loadGloVe(path):
    embeddings_index = {}
    f = open(path+'glove.6B.100d.txt',encoding = 'utf-8')
    for line in f:
        values = line.split(' ')
        word = values[0] ## The first entry is the word
        coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index
  
class Word2VecVectorizer:
    def __init__(self, GloVe):
        print("Loading in word vectors...")
        self.word_vectors = GloVe
        print("Finished loading in word vectors")
    
    def fit(self, data):
        pass
    
    def transform(self, pathData):
        data = pd.read_csv(pathData)
        data.dropna()
        Date = data.iloc[:,0].tolist()
        Text = data.iloc[:,1].tolist()
        # determine the dimensionality of vectors
        v = self.word_vectors['king']
        self.D = v.shape[0]
    
        X = np.zeros((len(Text), self.D))
        n = 0
        emptycount = 0
        for sentence in Text:
            if type(sentence) == str:
                tokens = sentence.split()
                vecs = []
                m = 0
                if 'title' in pathData:
                    tokens = tokens[:-2]
                for word in tokens:
                    try:
                        # throws KeyError if word not found
                        vec = self.word_vectors[word]
                        vecs.append(vec)
                        m += 1
                    except KeyError:
                        pass
                if len(vecs) > 0:
                    vecs = np.array(vecs)
                    X[n] = vecs.mean(axis=0)
                else:
                    emptycount += 1
                n += 1
        print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
        
        GloveDf = pd.DataFrame(X)
        GloveDf.columns = ['dim'+str(i) for i in range(X.shape[1])]
        GloveDf['Date'] = Date
        return GloveDf
       
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
 
pathGlove = '/Users/yunbai/Library/CloudStorage/OneDrive-Personal/My-PhD/Experiments-Paper1/07text-features-for-load/sentiment-features-all/'
pathData = './RegionalNews/EastMidlands/EMnews_clean.csv'

GloVe = loadGloVe(pathGlove+'glove/')

vectorizer = Word2VecVectorizer(GloVe)
title_features_GloVe = vectorizer.fit_transform(pathData+'title_clean.csv')
des_features_GloVe = vectorizer.fit_transform(pathData+'des_clean.csv')
body_features_GloVe = vectorizer.fit_transform(pathData+'body_clean_2023.csv')

title_features_GloVe_grouped = title_features_GloVe.groupby('Date').mean()
des_features_GloVe_grouped = des_features_GloVe.groupby('Date').mean()
body_features_GloVe_grouped = body_features_GloVe.groupby('Date').mean()

title_features_GloVe_grouped.to_csv('title_features_GloVe_grouped.csv')
des_features_GloVe_grouped.to_csv('des_features_GloVe_grouped.csv')
body_features_GloVe_grouped.to_csv('body_features_GloVe_grouped.csv')






















    
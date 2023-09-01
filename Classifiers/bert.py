from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from gensim.models import KeyedVectors
 
warnings.filterwarnings(action = 'ignore')
 
import gensim
from gensim.models import Word2Vec


import numpy as np
import pandas as pd
import json
import pickle
import heapq
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from nltk.tokenize import TweetTokenizer
import re
import sys
import random
import heapq 
from get_abstract import count_shared_papers

from sklearn.decomposition import PCA
from sklearn.svm import SVC


from sklearn.model_selection import KFold


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')



with open('../MADStat-dataset-final-version/data.json') as json_file:
    data = json.load(json_file)
    
'''load list of authors'''
with open('../author_name.txt') as f:
    authors = f.readlines()
authors = [author.strip() for author in authors]

'''load papers info'''
papers = pd.read_csv("../paper.csv")

"""load list of authors having at least 30 papers"""
with open("../../authors","rb") as fp:
    author_l = pickle.load(fp)




def bert(data_train,data_test):
    author1, author2 = set(data_train["author"])
    text1 = data_train[data_train["author"]==author1]
    text2 = data_train[data_train["author"]==author2]

    t1 = data_test[data_test["author"]==author1]
    t2 = data_test[data_test["author"]==author2]

    n, m = text1.shape[0], text2.shape[0]
    embeddings = model.encode(list(text1.text)+list(text2.text))
    embed = np.zeros((n+m,384))
    for i in range(n+m):
        embed[i,:] = embeddings[i]
        
    pca = PCA(n_components=10)
    X_train = pca.fit_transform(embed)
    y_train = np.concatenate((np.ones(n),np.zeros(m)))
    clf = SVC()
    clf.fit(X_train,y_train)


    nt, mt = t1.shape[0], t2.shape[0]
    embed_t = np.zeros((nt+mt,384))
    for i in range(nt):
        embeddings = model.encode([t1.text.iloc[i]])
        embed_t[i,:] = embeddings[0]
    for j in range(mt):
        embeddings = model.encode([t2.text.iloc[j]])
        embed_t[nt+j,:] = embeddings[0]
        
    X_test = pca.transform(embed_t)
    y_test = np.concatenate((np.ones(nt),np.zeros(mt)))
    y_pred = clf.predict(X_test)
    return y_pred, y_test



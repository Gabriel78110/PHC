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
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import re
import sys
import random
import heapq 
from get_abstract import count_shared_papers


# N = 32
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


# with open('../MADStat-dataset-final-version/data.json') as json_file:
#     data = json.load(json_file)
    
# '''load list of authors'''
# with open('../author_name.txt') as f:
#     authors = f.readlines()
# authors = [author.strip() for author in authors]

# '''load papers info'''
# papers = pd.read_csv("../paper.csv")

# """load list of authors having at least 30 papers"""
# with open("../../authors","rb") as fp:
#     author_l = pickle.load(fp)


# def clean_text(data) :
#     #data.text = data.text.apply(remove_hexa_symbols)
#     #data.text = data.text.apply(remove_digits)
#     data = data.filter(['author', 'title', 'text']).rename(columns = {'title' : 'doc_id'})
#     data["len"] = data.text.apply(lambda x: len(x))
#     data.text = data.text.apply(lambda x: re.sub("All rights","",x))
#     data.text = data.text.apply(lambda x: re.sub("reserved","",x))
# #         data.text = data.text.apply(lambda x: re.sub("[0-9]","",x))
#     data.text = data.text.apply(lambda x: re.sub("[^A-Za-z ]","",x))
#     data.text = data.text.apply(lambda x: re.sub("copyright","",x))
#     data.text = data.text.apply(lambda x: x.lower())
#     data = data.loc[data.len > 10].reset_index()
#     data.drop(columns=["len"],inplace=True)
#     return data
    
# def topKFrequent(nums, k):
#     dic=Counter(nums)
#     heapmax=[[-freq,num] for num,freq in dic.items()]
#     heapq.heapify(heapmax)
#     list1=[]
#     for i in range(k):
#         poping=heapq.heappop(heapmax)
#         list1.append(poping[1])
#     return list1


# def get_vocab(text, max_length=200):
# #     clf = CountVectorizer(lowercase=True)
# #     clf.fit([text])
# #     vocab = list(clf.vocabulary_.keys())
# #     print("vocab before = ",vocab)
#     vocab_f = []
#     vocab = text.split()
#     for word in set(vocab):
#         if word in model:
#             vocab_f.append(word)
#     return vocab_f


def doc_to_vec(doc):
    cur = np.zeros(300)
    i = 0
    for word in doc.split():
        if word in model:
            i+=1
            cur+=model[word]
    return cur/i


def svm(data_train,data_test):
    author1, author2 = set(data_train["author"])
    #vocab = get_vocab(''.join([doc + " " for doc in list(data_train["text"])]), max_length=400)
    text1 = data_train[data_train["author"]==author1]
    text2 = data_train[data_train["author"]==author2]
    # sentences = [i.split() for i in list(text1.text)+list(text2.text)]
    # w2v = Word2Vec(sentences, min_count = 1, vector_size = N)
    #print("model = ", w2v.wv.keys())
    
    n, m = text1.shape[0], text2.shape[0]
    embed = np.zeros((n+m,300))
    for i in range(n):
        embed[i,:] = doc_to_vec(text1.text.iloc[i])
    for j in range(n,n+m):
        embed[j,:] = doc_to_vec(text2.text.iloc[j-n])
            
    pca = PCA(n_components=10)
    X_train = pca.fit_transform(embed)
    y_train = np.concatenate((np.ones(n),np.zeros(m)))
    clf = SVC()
    clf.fit(X_train,y_train)
    #print("Training error svm = ", clf.score(X_train,y_train))

    t1 = data_test[data_test["author"]==author1]
    t2 = data_test[data_test["author"]==author2]

    nt, mt = t1.shape[0], t2.shape[0]
    embed_t = np.zeros((nt+mt,300))
    for i in range(nt):
        embed_t[i,:] = doc_to_vec(t1.text.iloc[i])
    for j in range(nt,nt+mt):
        embed_t[j,:] = doc_to_vec(t2.text.iloc[j-nt])
        
    X_test = pca.transform(embed_t)
    y_test = np.concatenate((np.ones(nt),np.zeros(mt)))
    y_pred = clf.predict(X_test)
    return y_pred, y_test
    

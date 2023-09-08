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
import re
import random
import heapq
from get_abstract import count_shared_papers

with open('MADStat-dataset-final-version/data.json') as json_file:
    data = json.load(json_file)
    
'''load list of authors'''
with open('author_name.txt') as f:
    authors = f.readlines()
authors = [author.strip() for author in authors]

'''load papers info'''
papers = pd.read_csv("paper.csv")

"""load list of authors having at least 30 papers"""
with open("authors","rb") as fp:
    author_l = pickle.load(fp)


def clean_text(data) :
        data = data.filter(['author', 'title', 'text']).rename(columns = {'title' : 'doc_id'})
        data["len"] = data.text.apply(lambda x: len(x))
        data.text = data.text.apply(lambda x: re.sub("All rights","",x))
        data.text = data.text.apply(lambda x: re.sub("reserved","",x))
        data.text = data.text.apply(lambda x: re.sub("[^A-Za-z ]","",x))
        data.text = data.text.apply(lambda x: re.sub("copyright","",x))
        data.text = data.text.apply(lambda x: x.lower())
        data = data.loc[data.len > 10].reset_index()
        data.drop(columns=["len"],inplace=True)
        return data
    
def topKFrequent(nums, k):
    dic=Counter(nums)
    heapmax=[[-freq,num] for num,freq in dic.items()]
    heapq.heapify(heapmax)
    list1=[]
    for i in range(k):
        poping=heapq.heappop(heapmax)
        list1.append(poping[1])
    return list1


def get_vocab(text, max_length=200):
#     clf = CountVectorizer(lowercase=True)
#     clf.fit([text])
#     vocab = list(clf.vocabulary_.keys())
#     print("vocab before = ",vocab)
    vocab = text.split()
    k = min(max_length, len(set(vocab)))
#     return heapq.nlargest(k, vocab, key=vocab.get)
#     print(vocab)
    return topKFrequent(vocab,k)

"""
Input:  - text is a list of strings corresponding to documents
        - vocab is the vocabulary used for the problem
"""
def doc_to_dtm(text, vocab):
    vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split(),vocabulary=vocab) #tokenizer=tk.tokenize,
    X = vectorizer.transform(text)
    return X.toarray()

def estimate_poisson(corpus):
    return np.mean(corpus,axis=0)

def accuracy(y_preds,y_true):
    return np.mean(y_preds==y_true)

def chi2(data_train,data_test):
    author1, author2 = set(data_train["author"])
    vocab = get_vocab(''.join([doc for doc in list(data_train["text"])]))

    text1 = data_train[data_train["author"]==author1]
    text2 = data_train[data_train["author"]==author2]

    corpus1 = doc_to_dtm(list(text1.text),vocab=vocab)
    corpus2 = doc_to_dtm(list(text2.text),vocab=vocab)

    lam_1 = estimate_poisson(corpus1)
    lam_2 = estimate_poisson(corpus2)
    y_pred = []
    for doc in list(data_test["text"]):
        dtm = doc_to_dtm([doc],vocab=vocab)
        if np.sum((dtm - lam_1)**2) < np.sum((dtm - lam_2)**2):
            y_pred.append(author1)
        else:
            y_pred.append(author2)

    y_true = list(data_test["author"])
    y_pred = [0 if item==author1 else 1 for item in y_pred]
    y_true = [0 if item==author1 else 1 for item in y_true]
    return np.array(y_pred), np.array(y_true)
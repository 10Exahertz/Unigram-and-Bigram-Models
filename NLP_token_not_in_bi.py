# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 19:56:07 2019

@author: EPCOT
"""

#import pandas as pd
#import numpy as np
import pickle
with open('tokenized_train_data.pkl', 'rb') as fp:
    listtt = pickle.load(fp)
    
bi_gram_listt = []
#for token in listt:
for idx, elem in enumerate(listtt):
    this_token = elem
    next_token = listtt[(idx + 1) % len(listtt)]
    new_token = str(this_token+","+next_token)
    bi_gram_listt.append(new_token)

listtt_of_bi_features = {}
listts_of_bi_feat = []
indexx = 0
for token in bi_gram_listt:
    if token not in listtt_of_bi_features:
        listtt_of_bi_features[token] = indexx
        listts_of_bi_feat.append(token)
        indexx += 1
      
def features_list_bi(document):
    listt= []
    file = open(document, 'r')
    
    for line in file:
        listt.append('<s>')
        for word in line.split():
            listt.append(word.lower())
        listt.append('</s>') 
    for token in listt:
        counter = listt.count(token)
        if counter == 1:
            for n, i in enumerate(listt):
                if i == token:
                    listt[n] = '<unk>'
    bi_gram_list = []
    #for token in listt:
    for idx, elem in enumerate(listt):
        this_token = elem
        next_token = listt[(idx + 1) % len(listt)]
        new_token = str(this_token+","+next_token)
        bi_gram_list.append(new_token)

    listt_of_bi_features = {}
    lists_of_bi_feat = []
    indexx = 0
    for token in bi_gram_list:
        if token not in listt_of_bi_features:
            listt_of_bi_features[token] = indexx
            lists_of_bi_feat.append(token)
            indexx += 1
      
    return listt_of_bi_features
def listt_bi(document):
    listt= []
    file = open(document, 'r')
    
    for line in file:
        listt.append('<s>')
        for word in line.split():
            listt.append(word.lower())
        listt.append('</s>') 
    for token in listt:
        counter = listt.count(token)
        if counter == 1:
            for n, i in enumerate(listt):
                if i == token:
                    listt[n] = '<unk>'
    bi_gram_list = []
    #for token in listt:
    for idx, elem in enumerate(listt):
        this_token = elem
        next_token = listt[(idx + 1) % len(listt)]
        new_token = str(this_token+","+next_token)
        bi_gram_list.append(new_token)
    return bi_gram_list
                
A = listtt_of_bi_features
B = features_list_bi('brown-test.txt')
C = features_list_bi('learner-test.txt')
Btrain = bi_gram_listt
Btest = listt_bi('brown-test.txt')
Ltest = listt_bi('learner-test.txt')
count_1 = 0
for token in B:
    if token not in A:
        count_1 += 1
        
count_2 = 0
for token in C:
    if token not in A:
        count_2 += 1

count_3 = 0
for token in Btest:
    if token not in Btrain:
        count_3 += 1
        
count_4 = 0
for token in Ltest:
    if token not in Btrain:
        count_4 += 1
        
token_brown_training_test_type = count_1/(len(B))*100
print(count_1)
token_brown_learner_type = count_2/(len(C))*100

word_token_btest_not_in_train = count_3/(len(Btest))*100

word_token_ltest_not_in_train = count_4/(len(Ltest))*100

print("% of bigram word types in brown test not in training: ",token_brown_training_test_type)
print("% of bigram word types in learner test not in training: ",token_brown_learner_type)
print("% of bigram word tokens in brown test not in training: ",word_token_btest_not_in_train)
print("% of bigram word tokens in learner test not in training: ",word_token_ltest_not_in_train)


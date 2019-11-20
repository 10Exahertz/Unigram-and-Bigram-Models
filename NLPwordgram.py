#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:56:16 2019

@author: stevenalsheimer
"""
import pandas as pd
import numpy as np

listt = []
file = open('train_small.txt', 'r')

for line in file:
    listt.append('<s>')
    for word in line.split():
        listt.append(word.lower())
    listt.append('</s>') 
listt_of_features = {}
lists_of_feat = []
index=0
for token in listt:
    counter = listt.count(token)
    if counter == 1:
        for n, i in enumerate(listt):
            if i == token:
                listt[n] = '<unk>'
    
for token in listt:
  if token not in listt_of_features:
      listt_of_features[token] = index
      lists_of_feat.append(token)
      index += 1
    
def unigram_vector_creator(some_text, features_dictionary):
  unigram_vector = len(features_dictionary)*[0]
  tokens = some_text
  for token in tokens:
    if token in features_dictionary:
      index = features_dictionary[token]
      unigram_vector[index] += 1
  return unigram_vector, features_dictionary
unigram_vec = unigram_vector_creator(listt, listt_of_features)
print(unigram_vec)
unigram_DF = pd.DataFrame({'token':lists_of_feat, 'count':unigram_vec[0]})

def unigram_word_probability(token):
    if token not in listt_of_features:
        token = '<unk>'
    index_token = unigram_DF.index[unigram_DF['token'] == token].tolist()
    index_token = index_token[0]
    summ = unigram_DF['count'].sum()
    word_count = unigram_DF.loc[index_token,'count']
    prob = word_count/summ
    return prob

###BIGRAM MODELS BELOW
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
        
        
new_list_bi_dat = []
final_list_bi_dat = []
for idx, elem in enumerate(listt):
    this_token = elem
    next_token = listt[(idx + 1) % len(listt)]
    new_token = [this_token,next_token]
    new_list_bi_dat.append(new_token)
for token in new_list_bi_dat:
    if token not in final_list_bi_dat:
        final_list_bi_dat.append(token)
new_list_bi_dataframe = pd.DataFrame(final_list_bi_dat, columns = ['contok','tok'])
#print(new_list_bi_dataframe)  

    
def bigram_vector_creator(some_text, features_bi_dictionary):
  bigram_vector = len(features_bi_dictionary)*[0]
  #index=0
  tokens = some_text
  for token in tokens:
    if token in features_bi_dictionary:
        indexx = features_bi_dictionary[token]
        bigram_vector[indexx] += 1
  return bigram_vector, features_bi_dictionary[token]
#print(len(bi_gram_list))
bigram_vec = bigram_vector_creator(bi_gram_list, listt_of_bi_features)
bigram_DF = pd.DataFrame({'token':lists_of_bi_feat, 'count':bigram_vec[0]})
katz_dataframe_list = []


def bigram_word_probability(conditional_token,token):
    if token not in listt_of_features:
        token = '<unk>'
    if conditional_token not in listt_of_features:
        conditional_token = '<unk>'
    index_token = unigram_DF.index[unigram_DF['token'] == conditional_token].tolist()
    index_token = index_token[0]
    conditional_word_count = unigram_DF.loc[index_token,'count']
    
    index_bi_word = conditional_token+','+token
    if index_bi_word not in listt_of_bi_features:
        prob = 0
    else:
        
        index_bi_token = bigram_DF.index[bigram_DF['token'] == index_bi_word].tolist()
        index_bi_token = index_bi_token[0]
        bigram_word_count = bigram_DF.loc[index_bi_token,'count']
        prob = bigram_word_count/conditional_word_count
        #log_prob = np.log2(prob)
    return prob
###Prob of a sentence entered###
def Sentence_unigram_prob(sentence_here):
    sentence_list = []
    sentence_list.append('<s>')
    for word in sentence.split():
        sentence_list.append(word.lower())
    sentence_list.append('</s>')
    sentence_log_prob = 0
    for token in sentence_list:
        prob = unigram_word_probability(token)
        log_prob = np.log2(prob)
        sentence_log_prob += log_prob
    sentence_prob = 2**sentence_log_prob
    return sentence_prob

def Sentence_bigram_prob(sentence_here):
    sentence_list = []
    sentence_log_prob = 0
    sentence_list.append('<s>')
    for word in sentence.split():
        sentence_list.append(word.lower())
    sentence_list.append('</s>')
    for idx, elem in enumerate(sentence_list):
        this_token = elem
        next_token = sentence_list[(idx + 1) % len(sentence_list)]
        prob = bigram_word_probability(this_token,next_token)
        if prob == 0:
            sentence_log_prob = "-inf"
            break
        log_prob = np.log2(prob)
        sentence_log_prob += log_prob
    return sentence_log_prob

###Bi_gram with add 1 smoothing###
def bigram_word_probability_add1(conditional_token,token):
    if token not in listt_of_features:
        token = '<unk>'
    if conditional_token not in listt_of_features:
        conditional_token = '<unk>'
    index_token = unigram_DF.index[unigram_DF['token'] == conditional_token].tolist()
    index_token = index_token[0]
    conditional_word_count = unigram_DF.loc[index_token,'count']
    
    index_bi_word = conditional_token+','+token
    if index_bi_word not in listt_of_bi_features:
        prob = 1/(len(listt_of_features))
    else:
        
        index_bi_token = bigram_DF.index[bigram_DF['token'] == index_bi_word].tolist()
        index_bi_token = index_bi_token[0]
        bigram_word_count = bigram_DF.loc[index_bi_token,'count']
        prob = (bigram_word_count+1)/(conditional_word_count+len(listt_of_features))
        #log_prob = np.log2(prob)
    return prob
def Sentence_bigram_prob_add1(sentence_here):
    sentence_list = []
    sentence_log_prob = 0
    sentence_list.append('<s>')
    for word in sentence.split():
        sentence_list.append(word.lower())
    sentence_list.append('</s>')
    for idx, elem in enumerate(sentence_list):
        this_token = elem
        next_token = sentence_list[(idx + 1) % len(sentence_list)]
        prob = bigram_word_probability_add1(this_token,next_token)
        if prob == 0:
            sentence_log_prob = "-inf"
            break
        log_prob = np.log2(prob)
        sentence_log_prob += log_prob
    return sentence_log_prob

###Katz discount and back-off method###

#print(listt_of_bi_features)
    
def bigram_token_probability_katz(conditional_token, token):
    if token not in listt_of_features:
        token = '<unk>'
    if conditional_token not in listt_of_features:
        conditional_token = '<unk>'
    count_BB_prob = 0
    index_token = unigram_DF.index[unigram_DF['token'] == conditional_token].tolist()
    index_token = index_token[0]
    word_count = unigram_DF.loc[index_token,'count']
    New_list_dataframe = new_list_bi_dataframe[new_list_bi_dataframe['contok'] == conditional_token]
    New_list_dataframe = New_list_dataframe.reset_index()
    countts = len(New_list_dataframe)
    alpha = 1- countts*(0.5/(word_count))
    final_bi = str(conditional_token+","+token)
    for i in range(countts):
        token_b = New_list_dataframe.loc[i,'tok']
        prob = unigram_word_probability(token_b)
        prob = 1-prob
        logprob = np.log2(prob)
        count_BB_prob += logprob
    count_B_prob = count_BB_prob
    #count_B_prob = np.log2(count_B_prob)
    if final_bi in listt_of_bi_features:
        index_bi_token = bigram_DF.index[bigram_DF['token'] == final_bi].tolist()
        index_bi_token = index_bi_token[0]
        bigram_word_count = bigram_DF.loc[index_bi_token,'count']
            
        index_token = unigram_DF.index[unigram_DF['token'] == conditional_token].tolist()
        index_token = index_token[0]
        conditional_word_count = unigram_DF.loc[index_token,'count']
        count_bi  = (bigram_word_count-0.5)/(conditional_word_count)
        prob = np.log2(count_bi)
        return prob
    if final_bi not in listt_of_bi_features:
        word_prob = unigram_word_probability(token)
        word_probl = np.log2(word_prob)
        probb = alpha*word_probl/(count_B_prob)
        prob = probb
        return prob
    
def Sentence_bigram_prob_katz(sentence_here):
    sentence_list = []
    sentence_log_prob = 0
    sentence_list.append('<s>')
    for word in sentence.split():
        sentence_list.append(word.lower())
    sentence_list.append('</s>')
    for idx, elem in enumerate(sentence_list):
        this_token = elem
        next_token = sentence_list[(idx + 1) % len(sentence_list)]
        print(this_token,next_token)
        prob = bigram_token_probability_katz(this_token,next_token)
        if prob == 0:
            sentence_log_prob = "-inf"
            break
        #log_prob = np.log2(prob)
        sentence_log_prob += prob
    return sentence_log_prob

sentence = "The Fulton County Grand Jury said Friday an investigation of Atlanta's ."
        
#print(Sentence_bigram_prob_katz(sentence))
#print(np.log2(bigram_word_probability('<unk>','<unk>')))
#fp = open("demofile2.txt", "a")
#fp.write(str(listt))
#fp.close()
###Tokenized Training Set and prob definitions###
test_brown_listt = []
file_bt = open('train_small.txt', 'r')

for line in file_bt:
    test_brown_listt.append('<s>')
    for word in line.split():
        test_brown_listt.append(word.lower())
    test_brown_listt.append('</s>') 
for token in test_brown_listt:
    counter = test_brown_listt.count(token)
    if counter == 1:
        for n, i in enumerate(test_brown_listt):
            if i == token:
                test_brown_listt[n] = '<unk>'
#print("BT_M:",len(test_brown_listt))
                
test_learner_listt = []
file_lt = open('train_small.txt', 'r')

for line in file_lt:
    test_learner_listt.append('<s>')
    for word in line.split():
        test_learner_listt.append(word.lower())
    test_learner_listt.append('</s>') 
for token in test_learner_listt:
    counter = test_learner_listt.count(token)
    if counter == 1:
        for n, i in enumerate(test_learner_listt):
            if i == token:
                test_learner_listt[n] = '<unk>'
#print("LT_M:",len(test_learner_listt))
                
def test_set_unigram_prob(tokenized_test_here):
    sentence_log_prob = 0
    for token in tokenized_test_here:
        prob = unigram_word_probability(token)
        log_prob = np.log2(prob)
        sentence_log_prob += log_prob
    return sentence_log_prob

def test_set_bigram_prob(test_list):
    sentence_log_prob = 0
    for idx, elem in enumerate(test_list):
        this_token = elem
        next_token = test_list[(idx + 1) % len(test_list)]
        prob = bigram_word_probability(this_token,next_token)
        if prob == 0:
            sentence_log_prob = "-inf"
            break
        log_prob = np.log2(prob)
        sentence_log_prob += log_prob
    return sentence_log_prob
    
def test_set_bigram_prob_add1(sentence_list):
    sentence_log_prob = 0
    for idx, elem in enumerate(sentence_list):
        this_token = elem
        next_token = sentence_list[(idx + 1) % len(sentence_list)]
        prob = bigram_word_probability_add1(this_token,next_token)
        if prob == 0:
            sentence_log_prob = "-inf"
            break
        log_prob = np.log2(prob)
        sentence_log_prob += log_prob
    return sentence_log_prob
def test_set_bigram_prob_katz(sentence_list):
    sentence_log_prob = 0
    for idx, elem in enumerate(sentence_list):
        this_token = elem
        next_token = sentence_list[(idx + 1) % len(sentence_list)]
        prob = bigram_token_probability_katz(this_token,next_token)
        if prob == 0:
            sentence_log_prob = "-inf"
            break
        #log_prob = np.log2(prob)
        sentence_log_prob += prob
    return sentence_log_prob
#import time
#start_time = time.time()
#print("here")
#BT_add1 = Sentence_bigram_prob_katz(sentence)
##BT_add1 = bigram_token_probability_katz('the','<unk>')
#print(BT_add1)
#print("--- %s seconds ---" % (time.time() - start_time))
#import pickle
#
#with open('test.pkl', 'wb') as fp:
#    pickle.dump(listt, fp)
            
            
            
#print(bigram_word_probability_add1('<unk>','to'))
#print(Sentence_bigram_prob_add1(sentence))






























































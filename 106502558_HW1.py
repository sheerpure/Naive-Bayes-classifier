#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas
import numpy as np
from itertools import groupby

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def gen_train_test_data():
    test_ratio = .3 
    df = pandas.read_csv('./yelp_labelled.txt', sep='\t', header=None, encoding='utf-8')
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(df[0])
    y = df[1].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
    dist_count = len(count_vect.transform(df[0]).toarray().sum(axis = 0)) # Total number of distinct words 

    return X_train, X_test, y_train, y_test, dist_count



def multinomial_nb(X_train, X_test, y_train, y_test):
    index = max(X_train.indices) + 1
    posi_cl = 0
    negi_cl = 0
    posi_cl_le = [0] * index
    negi_cl_le = [0] * index
    for i in range(len(y_train)):
        if y_train[i] == 1:
            for j in range(len(X_train[i].data)):
                posi_cl += X_train[i].data[j]
                temp = X_train[i].indices[j]
                posi_cl_le[temp] += X_train[i].data[j]
        else:
            for j in range(len(X_train[i].data)):
                negi_cl += X_train[i].data[j]
                temp = X_train[i].indices[j]
                negi_cl_le[temp] += X_train[i].data[j]
    exc = 0
    for i in range(index):
        if posi_cl_le[i] or negi_cl_le[i]:
            exc += 1
    pos_of_d = posi_cl + exc
    neg_of_d = negi_cl + exc
    p_pos = [0] * index
    p_neg = [0] * index
    for i in range(index):
        p_pos[i] = (posi_cl_le[i]+1) / pos_of_d
        p_neg[i] = (negi_cl_le[i]+1) / neg_of_d
    x = 0
    for i in range(len(y_train)):
        ans = y_train[i]
        pos = 1
        neg = 1
        for j in range(len(X_train[i].indices)):
            temp = X_train[i].indices[j]
            for k in range(X_train[i].data[j]):
                pos *= p_pos[temp]
                neg *= p_neg[temp]
        if (pos > neg and ans == 0) or (pos < neg and ans == 1):
            x += 1
    print("multinomial_nb - training : ")
    print("Accuracy : = ", 1-x/len(y_train))

    # test
    x = 0
    for i in range(len(y_test)):
        ans = y_test[i]
        pos, neg  = 1, 1
        for j in range(len(X_test[i].indices)):
            temp = X_test[i].indices[j]
            for k in range(X_test[i].data[j]):
                pos *= p_pos[temp]
                neg *= p_neg[temp]
        if (pos > neg and ans == 0) or (pos < neg and ans == 1):
            x += 1
    print("multinomial_nb - testing : ")
    print("Accuracy : = ", 1-x/len(y_test))
    pass


def bernoulli_nb(X_train, X_test, y_train, y_test, dist_count):
    list1 = [] 
    list2 = []
    for i in range(len(y_train)):
        if(y_train[i] == 1):
            list1 += X_train[i].indices.tolist()
        else:
            list2 += X_train[i].indices.tolist()

    list2.sort()
    lst_0_freq = [len(list(group)) for key, group in groupby(list2)]
    lst_0_freq_id = [list(pair) for pair in zip(list(set(list2)), lst_0_freq)]

    list1.sort()
    lst_1_freq = [len(list(group)) for key, group in groupby(list1)]
    lst_1_freq_id = [list(pair) for pair in zip(list(set(list1)), lst_1_freq)]

    for element_0 in lst_0_freq_id:
        element_0.append((element_0[1]+1)/(y_train.count(0) + dist_count))
    for element_1 in lst_1_freq_id:
        element_1.append((element_1[1]+1)/(y_train.count(1) + dist_count))

    y = 0
    for i in range(len(y_train)):
        p1, p2 = 1, 1
        for element in X_train[i].indices.tolist():
            try:
                temp0 = [x for x in lst_0_freq_id if element in x][0]
                p1 *= temp0[2]
                
            except IndexError:
                p1 *= 1/(y_train.count(0) + dist_count)

            try:
                temp1 = [x for x in lst_1_freq_id if element in x][0]
                p2 *= temp1[2]
                
            except IndexError:
                p2 *= 1/(y_train.count(1) + dist_count)
        predict = 1 if p2 > p1 else 0
        if (predict != y_train[i]):
            y += 1
    print("bernoulli_nb - training : ")
    print("Accuracy: ", 1-(y/len(y_train)))

    y = 0
    for i in range(len(y_test)):
        p1, p2 = 1, 1
        for element in X_test[i].indices.tolist():
            try:
                temp0 = [x for x in lst_0_freq_id if element in x][0]
                p1 *= temp0[2]
                
            except IndexError:
                p1 *= 1/(y_train.count(0) + dist_count)

            try:
                temp1 = [x for x in lst_1_freq_id if element in x][0]
                p2 *= temp1[2]

            except IndexError:
                p2 *= 1/(y_train.count(1) + dist_count)

        predict = 1 if p2 > p1 else 0
        if (predict != y_test[i]):
            y += 1
    print("bernoulli_nb - testing : ")
    print("Accuracy: ", 1-(y/len(y_test)))

    pass

def main(argv):
    X_train, X_test, y_train, y_test, dist_count = gen_train_test_data()
    multinomial_nb(X_train, X_test, y_train, y_test)
    bernoulli_nb(X_train, X_test, y_train, y_test, dist_count)


if __name__ == "__main__":
    main(sys.argv)

# -*- coding: utf-8 -*-

from sklearn import preprocessing


def normalize(arr):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(arr)

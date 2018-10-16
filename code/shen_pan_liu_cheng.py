# -*- coding: utf-8 -*-

import pandas as pd
from code.lib.adjudicative_documents import *
import numpy as np


def shen_pan_liu_cheng(path_prefix='../data/train/'):
    """
    裁判流程 数据处理
    :return: pd.DataFrame
    """

    data = pd.read_csv(path_prefix + '3.csv')

    # print('处理 裁判流程 数据...')

    # data.insert(7, '裁判流程次数', 1)
    data = data.fillna('未知')
    data['审理机关'] = data.apply(fill_institution, axis=1)
    data['公告类型'] = data.apply(fill_gong_gao, axis=1)
    data['诉讼地位'] = data.apply(fill_role, axis=1)
    data['诉讼地位3'] = data['诉讼地位']

    data = data.drop(columns=['日期类别','具体日期','涉案事由'])
    data = pd.get_dummies(data, prefix=['诉讼地位3','审理机关','公告类型'], columns=['诉讼地位','审理机关','公告类型'])
    data = data.groupby('小微企业ID').sum().reset_index()
    return data


def fill_role(row):
    return adjudicative_documents_fole(row['诉讼地位'])


def fill_institution(row):
    return adjudicative_documents_institution(row['审理机关'])


def fill_gong_gao(row):
    return gong_gao(row['公告类型'])


if __name__ == '__main__':
    data = shen_pan_liu_cheng()
    data.to_csv('../data/processed/3.csv')
    data_t = shen_pan_liu_cheng('../data/test/')
    data_t.to_csv('../data/processed/3_test.csv')
    # print(len(data['诉讼地位'].unique()))
    # print(data['诉讼地位'].unique())
    # print(np.setdiff1d(data['诉讼地位'].values, data_t['诉讼地位'].values))
    # print(np.setdiff1d(data_t['诉讼地位'].values, data['诉讼地位'].values))

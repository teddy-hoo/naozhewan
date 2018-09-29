# -*- coding: utf-8 -*-

import pandas as pd
import jenkspy
from code.lib.adjudicative_documents import *
from code.lib.utils import normalize
import numpy as np


def cai_pan_wen_shu(path_prefix='../data/train/'):
    """
    裁判文书 数据处理
    :return: pd.DataFrame
    """

    data = pd.read_csv(path_prefix + '2.csv')

    # print('处理 裁判文书 数据...')

    data.insert(8, '裁判文书次数', 1)
    data = data.fillna('未知')
    #print(data.head())
    data['诉讼地位'] = data.apply(fill_role, axis=1)
    data['审理机关'] = data.apply(fill_institution, axis=1)
    data['审理程序'] = data.apply(fill_cheng_xu, axis=1)
    data['文书类型'] = data.apply(fill_wen_shu, axis=1)
    # print(data['文书类型'].unique())
    # data['审理程序'] = data.apply(fille)

    #print(data['涉案事由'].unique())#后期处理

    #删除日期 、涉案事由
    data = data.drop(columns=['涉案事由','结案时间'])

    breaks = jenkspy.jenks_breaks(data['涉案金额(元)'].values, nb_class=15)
    # print(breaks)
    nv = np.zeros(len(data['涉案金额(元)']))
    for i in range(0, len(breaks) - 1):
        c = 1
        for j, x in data['涉案金额(元)'].iteritems():
            if breaks[i] <= x < breaks[i + 1]:
                if i > 9:
                    nv[j] = 10
                else:
                    nv[j] = i+1
                c += 1
    data['涉案金额(元)'] = nv

    # print(data)
    data = pd.get_dummies(data, prefix=['诉讼地位', '审理机关', '文书类型', '审理程序'], columns=['诉讼地位', '审理机关','文书类型','审理程序'])
    data = data.groupby('小微企业ID').sum().reset_index()
    # print(data)
    return data


def fill_role(row):
    return adjudicative_documents_fole(row['诉讼地位'])


def fill_institution(row):
    return adjudicative_documents_institution(row['审理机关'])


def fill_cheng_xu(row):
    return shen_pan_cheng_xu(row['审理程序'])


def fill_wen_shu(row):
    return wen_shu_lei_xing(row['文书类型'])


if __name__ == '__main__':
    data = cai_pan_wen_shu()
    data.to_csv('../data/processed/2.csv')
    data_t = cai_pan_wen_shu('../data/test/')
    data_t.to_csv('../data/processed/2_test.csv')
    # print(len(data['文书类型'].unique()))
    # print(np.setdiff1d(data['文书类型'].values, data_t['文书类型'].values))
    # print(np.setdiff1d(data_t['文书类型'].values, data['文书类型'].values))

# -*- coding: utf-8 -*-

import pandas as pd
import jenkspy
from code.lib.adjudicative_documents import *
from code.lib.utils import normalize
import numpy as np


def cai_pan_wen_shu():
    """
    裁判文书 数据处理
    :return: pd.DataFrame
    """

    data = pd.read_csv('../data/train/2.csv')

    print('处理 裁判文书 数据...')

    data.insert(8, '裁判文书次数', 1)
    data = data.fillna('未知')
    #print(data.head())
    data['诉讼地位'] = data.apply(fill_role, axis=1)
    data['审理机关'] = data.apply(fill_institution, axis=1)

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

    data['涉案金额(元)'] = normalize(data['涉案金额(元)'])
    # print(data)
    data = pd.get_dummies(data, prefix=['诉讼地位', '审理机关', '文书类型', '审理程序'], columns=['诉讼地位', '审理机关','文书类型','审理程序'])
    data = data.groupby('小微企业ID').sum().reset_index()
    # print(data)
    return data


def fill_role(row):
    return adjudicative_documents_fole(row['诉讼地位'])


def fill_institution(row):
    return adjudicative_documents_institution(row['审理机关'])


if __name__ == '__main__':
    cai_pan_wen_shu()

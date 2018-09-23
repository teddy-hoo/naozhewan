# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from code.lib.adjudicative_documents import *


def cai_pan_wen_shu():
    """
    裁判文书 数据处理
    :return: pd.DataFrame
    """

    data = pd.read_csv('../data/train/2.csv')

    print('处理 裁判文书 数据...')

    data.insert(8, '裁判文书次数', 1)
    data = data.fillna('未知')
    print(data.head())
    #print(data['诉讼地位'].unique())
    data['诉讼地位'] = data.apply(fill_role, axis=1)


    print('处理 审理机关')
    data['审理机关'] = data.apply(fill_institution, axis=1)

    #print(data['涉案事由'].unique())#后期处理

    #删除日期 、涉案事由
    data = data.drop(columns=['涉案事由','结案时间'])
    data = pd.get_dummies(data, prefix=['诉讼地位','审理机关','文书类型','审理程序'], columns=['诉讼地位','审理机关','文书类型','审理程序'])
    data = data.groupby('小微企业ID').sum()

    #pd.options.display.max_rows = 10
    #pd.options.display.max_columns = 50
    print(data.head())
    return data


def fill_role(row):
    return adjudicative_documents_fole(row['诉讼地位'])
def fill_institution(row):
    return adjudicative_documents_institution(row['审理机关'])

if __name__ == '__main__':
    cai_pan_wen_shu()
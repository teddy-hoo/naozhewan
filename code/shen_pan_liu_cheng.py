# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from code.lib.adjudicative_documents import *


def shen_pan_liu_cheng():
    """
    裁判流程 数据处理
    :return: pd.DataFrame
    """

    data = pd.read_csv('../data/train/3.csv')

    print('处理 裁判流程 数据...')

    data.insert(7, '裁判流程次数', 1)
    data = data.fillna('未知')
    data['审理机关'] = data.apply(fill_institution, axis=1)

    data = data.drop(columns=['日期类别','具体日期','涉案事由'])
    data = pd.get_dummies(data, prefix=['诉讼地位','审理机关','公告类型'], columns=['诉讼地位','审理机关','公告类型'])
    data = data.groupby('小微企业ID').sum()
    return data



def fill_role(row):
    return adjudicative_documents_fole(row['诉讼地位'])
def fill_institution(row):
    return adjudicative_documents_institution(row['审理机关'])

if __name__ == '__main__':
    shen_pan_liu_cheng()
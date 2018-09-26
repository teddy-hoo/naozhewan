# -*- coding: utf-8 -*-

import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from code.lib.administrative_sanction import sanction_organization_value


def xing_zheng_wei_fa_ji_lu():
    """
    行政处罚机关 数据处理
    :return: pd.DataFrame
    """

    data = pd.read_csv('../data/train/4.csv')

    data.insert(4, '行政处罚次数', 1)
    # 现在不知道认定日期是做什么用的，所以暂时先去掉，放在后面优化
    data = data.drop(columns=['日期类别'])
    #data = data.drop(columns=['具体日期'])

    #data.info()
    #print(data.describe())


    # 将  执法/复议/审判机关 转化为数字 然后 向量化
    # print('将 执法/复议/审判机关 转化为数字')
    data = data.fillna('未知机关')
    date_year(data)
    # print(data.head())
    #print(data['执法/复议/审判机关'].unique())
    data['处罚机关'] = data.apply(fill, axis=1)
    data = data.drop(columns=['执法/复议/审判机关'])
    data = pd.get_dummies(data, prefix=['处罚机关','具体日期'], columns=['处罚机关','具体日期'])
    data = data.groupby('小微企业ID').sum()
    #print(data)
    return data

def date_year(df):
    for i, row in df.iterrows():
        t = row['具体日期'][0:4]
        df.at[i, '具体日期'] = t

def fill(row):
    return sanction_organization_value(row['执法/复议/审判机关'])


if __name__ == '__main__':
    xing_zheng_wei_fa_ji_lu()
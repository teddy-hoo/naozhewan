# -*- coding: utf-8 -*-

import pandas as pd
from code.lib.tax import tax_organization_value


def na_shui_fei_zheng_change_hu(path_prefix='../data/train/'):
    """
    纳税非正常户 数据处理
    :return: pd.DataFrame
    """

    data = pd.read_csv(path_prefix + '6.csv')

    # 现在不知道认定日期是做什么用的，所以暂时先去掉，放在后面优化
    data = data.drop(columns=['认定日期'])

    # 将 主管税务机关 转化为数字
    # print('将 主管税务机关 转化为数字')
    data = data.fillna('未知机关')
    # print(data.head())
    data['主管税务机关数字'] = data.apply(fill, axis=1)
    data = data.drop(columns=['主管税务机关'])
    # print(data.head(10))
    data = pd.get_dummies(data, prefix=['主管税务机关数字'], columns=['主管税务机关数字'])
    # print(data['小微企业ID'])
    data = data.groupby('小微企业ID').sum().reset_index()
    # print(data['小微企业ID'])

    return data


def fill(row):
    return tax_organization_value(row['主管税务机关'])


if __name__ == '__main__':
    data = na_shui_fei_zheng_change_hu()
    data.to_csv('../data/processed/6.csv')
    data = na_shui_fei_zheng_change_hu('../data/test/')
    data.to_csv('../data/processed/6_test.csv')

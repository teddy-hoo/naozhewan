# -*- coding: utf-8 -*-

import pandas as pd
# import matplotlib.pyplot as plt
from code.lib.tax import tax_organization_value, tax_type_value
from code.lib.utils import normalize


pd.options.display.max_rows = 10
pd.options.display.max_columns = 50


def qian_shui_ming_dan():
    """
    欠税名单 数据处理
    :return:
    """

    data = pd.read_csv('../data/train/5.csv')

    # 欠税属期  具体日期  目前不明白其业务意义 暂时先去掉
    data = data.drop(columns=['欠税属期', '具体日期'])

    # 将 主管税务机关 转化为数字
    # print('将 主管税务机关 转化为数字')
    data['主管税务机关'] = data['主管税务机关'].fillna('未知机关')
    # print(data.head())
    data['主管税务机关数字'] = data.apply(fill_tax_org, axis=1)
    data = data.drop(columns=['主管税务机关'])
    # print(data.head())

    # 将 所欠税种 转化为数字
    # print('将 所欠税种 转化为数字')
    data['所欠税种'] = data['所欠税种'].fillna('未知税种')
    # print(data.head())
    data['所欠税种数字'] = data.apply(fill_tax_type, axis=1)
    data = data.drop(columns=['所欠税种'])
    # print(data)

    # 加和相同税种的余额  然后将其打平 每个id变成一行
    data = data.groupby(['小微企业ID', '主管税务机关数字', '所欠税种数字'], as_index=False).sum()
    data['主管税务机关-所欠税种'] = '税务' + data['主管税务机关数字'].map(str) + '-' + data['所欠税种数字'].map(str)
    data = data.drop(columns=['主管税务机关数字', '所欠税种数字'])
    data['欠税余额(元)'] = normalize(data['欠税余额(元)'])
    data = pd.pivot_table(data, index='小微企业ID', columns=['主管税务机关-所欠税种'], values='欠税余额(元)')
    return data


def fill_tax_org(row):
    return tax_organization_value(row['主管税务机关'])


def fill_tax_type(row):
    return tax_type_value(row['所欠税种'])


if __name__ == '__main__':
    qian_shui_ming_dan()

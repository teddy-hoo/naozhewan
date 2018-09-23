# -*- coding: utf-8 -*-

import pandas as pd
from code.lib.tax import tax_organization_value


def na_shui_fei_zheng_change_hu():
    """
    纳税非正常户 数据处理
    :return: pd.DataFrame
    """

    data = pd.read_csv('../data/train/6.csv')

    # 现在不知道认定日期是做什么用的，所以暂时先去掉，放在后面优化
    data = data.drop(columns=['认定日期'])

    # 将 主管税务机关 转化为数字
    print('将 主管税务机关 转化为数字')
    data = data.fillna('未知机关')
    print(data.head())
    data['主管税务机关数字'] = data.apply(fill, axis=1)
    print(data.head())
    data = data.drop(columns=['主管税务机关'])

    return data


def fill(row):
    return tax_organization_value(row['主管税务机关'])


if __name__ == '__main__':
    na_shui_fei_zheng_change_hu()

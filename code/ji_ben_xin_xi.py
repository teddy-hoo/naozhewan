# -*- coding: utf-8 -*-

import pandas as pd
from code.lib.currency_exchange import currency_exchange
from code.lib.utils import normalize
import jenkspy
import numpy as np

pd.options.display.max_rows = 10
pd.options.display.max_columns = 50


def ji_ben_xin_xi(path_prefix='../data/train/'):
    """
    企业基本信息 数据处理
    :return:
    """

    data = pd.read_csv(path_prefix + '1.csv')

    # print(data['行业门类代码'].unique())
    data['行业门类代码'], _ = pd.factorize(data['行业门类代码'])
    # print(data['行业门类代码'].unique())

    currency_exchange(data, '注册资金(万元)', '注册资本(金)币种')
    # data['注册资金(万元)'].plot(kind='line', title='注册资金（万元）人民币')
    data = data.drop(columns=['注册资本(金)币种'])
    # print(data['注册资金(万元)'].values)
    breaks = jenkspy.jenks_breaks(data['注册资金(万元)'].values, nb_class=15)
    # print(breaks)
    nv = np.zeros(len(data['注册资金(万元)']))
    for i in range(0, len(breaks) - 1):
        c = 1
        for j, x in data['注册资金(万元)'].iteritems():
            if breaks[i] <= x < breaks[i + 1]:
                if i > 9:
                    nv[j] = 9
                else:
                    nv[j] = i
                c += 1
    data['注册资金(万元)'] = nv

    # print(data.head())

    # print(data['投资总额币种'].unique())
    currency_exchange(data, '投资总额(万元)', '投资总额币种')
    # data['投资总额(万元)'].plot(kind='line', title='投资总额(万元)人民币')
    data = data.drop(columns=['投资总额币种'])

    breaks = jenkspy.jenks_breaks(data['投资总额(万元)'].values, nb_class=15)
    # print(breaks)
    nv = np.zeros(len(data['投资总额(万元)']))
    for i in range(0, len(breaks) - 1):
        c = 1
        for j, x in data['投资总额(万元)'].iteritems():
            if breaks[i] <= x < breaks[i + 1]:
                if i > 9:
                    nv[j] = 9
                else:
                    nv[j] = i
                c += 1
    data['投资总额(万元)'] = nv

    # print(data.head())

    # print(data['企业(机构)类型'].unique())
    data['企业(机构)类型'], _ = pd.factorize(data['企业(机构)类型'])
    # print(data['企业(机构)类型'].unique())


    data = data.drop(columns=['许可经营项目','一般经营项目','经营(业务)范围','成立日期'])

    # print(data)

    return data


if __name__ == '__main__':
    data = ji_ben_xin_xi()
    data.to_csv('../data/processed/1.csv')
    data = ji_ben_xin_xi('../data/test/')
    data.to_csv('../data/processed/1_test.csv')

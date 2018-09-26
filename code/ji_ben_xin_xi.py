# -*- coding: utf-8 -*-

import pandas as pd
from code.lib.currency_exchange import currency_exchange
from code.lib.utils import normalize

pd.options.display.max_rows = 10
pd.options.display.max_columns = 50


def ji_ben_xin_xi():
    """
    企业基本信息 数据处理
    :return:
    """

    data = pd.read_csv('../data/train/1.csv')

    # print(data['行业门类代码'].unique())
    data['行业门类代码'], _ = pd.factorize(data['行业门类代码'])
    # print(data['行业门类代码'].unique())

    currency_exchange(data, '注册资金(万元)', '注册资本(金)币种')
    # data['注册资金(万元)'].plot(kind='line', title='注册资金（万元）人民币')
    data = data.drop(columns=['注册资本(金)币种'])
    data['注册资金(万元)'] = normalize(data['注册资金(万元)'])
    # print(data.head())

    # print(data['投资总额币种'].unique())
    currency_exchange(data, '投资总额(万元)', '投资总额币种')
    # data['投资总额(万元)'].plot(kind='line', title='投资总额(万元)人民币')
    data = data.drop(columns=['投资总额币种'])
    data['投资总额(万元)'] = normalize(data['投资总额(万元)'])
    # print(data.head())

    # print(data['企业(机构)类型'].unique())
    data['企业(机构)类型'], _ = pd.factorize(data['企业(机构)类型'])
    data['企业(机构)类型'] = normalize(data['企业(机构)类型'])
    # print(data['企业(机构)类型'].unique())

    data['从业人数'] = normalize(data['从业人数'])

    data = data.drop(columns=['许可经营项目','一般经营项目','经营(业务)范围','成立日期'])

    # print(data)

    return data


if __name__ == '__main__':
    ji_ben_xin_xi()

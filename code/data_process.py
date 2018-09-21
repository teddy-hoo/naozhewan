# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from code.lib.currency_exchange import currency_exchange
from code.lib.utils import normalize

import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'SimHei'


def ji_ben_xin_xi():
    data_ji_ben_xin_xi = pd.read_csv('../data/train/1.csv',
                                     keep_default_na=False,
                                     na_values=[""])

    print(data_ji_ben_xin_xi.head())

    currency_exchange(data_ji_ben_xin_xi, '注册资金(万元)', '注册资本(金)币种')
    data_ji_ben_xin_xi['注册资金(万元)'].plot(kind='line', title='注册资金（万元）人民币')
    plt.show(block=True)
    data_ji_ben_xin_xi = data_ji_ben_xin_xi.drop(columns=['注册资本(金)币种'])
    data_ji_ben_xin_xi['注册资金(万元)'] = normalize(data_ji_ben_xin_xi['注册资金(万元)'])
    print(data_ji_ben_xin_xi.head())

    print(data_ji_ben_xin_xi['投资总额币种'].unique())
    currency_exchange(data_ji_ben_xin_xi, '投资总额(万元)', '投资总额币种')
    data_ji_ben_xin_xi['投资总额(万元)'].plot(kind='line', title='投资总额(万元)人民币')
    plt.show(block=True)
    data_ji_ben_xin_xi = data_ji_ben_xin_xi.drop(columns=['投资总额币种'])
    data_ji_ben_xin_xi['投资总额(万元)'] = normalize(data_ji_ben_xin_xi['投资总额(万元)'])
    print(data_ji_ben_xin_xi.head())

    print(data_ji_ben_xin_xi['企业(机构)类型'].unique())
    data_ji_ben_xin_xi['企业(机构)类型'], _ = pd.factorize(data_ji_ben_xin_xi['企业(机构)类型'])
    print(data_ji_ben_xin_xi['企业(机构)类型'].unique())


def data_process():

    ji_ben_xin_xi()

    qian_shui_ming_dan = pd.read_csv('../data/train/5.csv',
                                     keep_default_na=False,
                                     na_values=[""])

    min_shang_shi_cai_pan_wen_shu = pd.read_csv('../data/train/2.csv',
                                                keep_default_na=False,
                                                na_values=[""])


# 将数据合并到一个二维数组里面
# print(basic_info)
# basic_info = pd.merge(left=basic_info, right=qian_shui_ming_dan, how='left',
#                       left_on='小微企业ID', right_on='小微企业ID')
# print(basic_info)


if __name__ == '__main__':
    data_process()

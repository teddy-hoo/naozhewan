# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from code.lib.currency_exchange import currency_exchange
from code.lib.utils import normalize
import seaborn as sns

import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'SimHei'

def ji_ben_xin_xi():
    data_ji_ben_xin_xi = pd.read_csv('../data/train/1.csv',
                                     keep_default_na=False,
                                     na_values=[""])


    #将老赖的信息并入基本信息中
    lao_lai = pd.read_csv('../data/train/8.csv',
                                     keep_default_na=False,
                                     na_values=[""])
    lao_lai.insert(1,'是否老赖',1)
    outfile = pd.merge(data_ji_ben_xin_xi, lao_lai, how='left', left_on='小微企业ID', right_on='小微企业ID')
    def clear_Nan(s):
        if pd.isna(s):
            return 0
        else:
            return 1
    outfile['是否老赖'] = outfile['是否老赖'].apply(clear_Nan)


    #分析注册资金
    def money_label(s):
        if (s <= 1000):
            return 1
        elif ((s > 1000) & (s <= 5000)):
            return 2
        elif ((s > 5000) & (s <= 10000)):
            return 3
        elif ((s > 10000) & (s <= 20000)):
            return 4
        elif ((s > 20000) & (s <= 30000)):
            return 5
        elif (s > 30000):
            return 6
    outfile['注册资金分段'] = outfile['注册资金(万元)'].apply(money_label)
    #sns.barplot(x="注册资金分段", y="是否老赖", data=outfile, palette='Set3')
    plt.show()

    #分析注册资金类型
    #sns.barplot(x="注册资本(金)币种", y="是否老赖", data=outfile, palette='Set3')
    plt.show()

    sns.barplot(x="企业(机构)类型", y="是否老赖", data=outfile, palette='Set3')
    plt.show()

    print(data_ji_ben_xin_xi.head())

    outfile.to_csv('outfile.csv', index=False)

    print(data_ji_ben_xin_xi['企业(机构)类型'].unique())

    exit(1)

    print(data_ji_ben_xin_xi.head())

    currency_exchange(data_ji_ben_xin_xi, '注册资金(万元)', '注册资本(金)币种')
    print(data_ji_ben_xin_xi)
    exit(1)
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

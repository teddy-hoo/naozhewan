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


    #企业性质
    outfile['企业性质'] = outfile['企业(机构)类型'].apply(lambda x:x.split('(')[0])
    Title_Dict = {}
    Title_Dict.update(dict.fromkeys(['有限责任公司', '有限责任', '有限责任公司分公司'], '有限责任公司'))
    Title_Dict.update(dict.fromkeys(['集体所有制'], '集体所有制'))
    Title_Dict.update(dict.fromkeys(['股份合作制'], '股份合作制'))
    Title_Dict.update(dict.fromkeys(['股份有限公司', '股份','股份有限','股份有限公司分公司'], '股份有限公司'))
    Title_Dict.update(dict.fromkeys(['农民专业合作经济组织'], '农民专业合作经济组织'))
    Title_Dict.update(dict.fromkeys(['集体分支机构'], '集体分支机构'))
    Title_Dict.update(dict.fromkeys(['全民所有制', '全民所有制分支机构'], '全民所有制'))
    Title_Dict.update(dict.fromkeys(['联营'], '联营'))
    Title_Dict.update(dict.fromkeys(['外商投资企业分公司'], '外商投资企业分公司'))
    Title_Dict.update(dict.fromkeys(['个人独资企业'], '个人独资企业'))
    Title_Dict.update(dict.fromkeys(['台、港、澳投资企业分公司', '投资企业分公司'], '台、港、澳投资企业分公司'))
    outfile['企业性质'] = outfile['企业性质'].map(Title_Dict)
    #sns.barplot(x="企业性质", y="是否老赖", data=outfile, palette='Set3')
    plt.show()


    #企业控制人类型 (未完,从独资合资角度分析，从国内国外组成成分分析)
    outfile['企业控制人类型'] = outfile['企业(机构)类型'].apply(lambda x:x.split('(')[0])
    Control_Dict = {}
    Control_Dict.update(dict.fromkeys(['自然人投资或控股', '自然人独资', '外国自然人独资','台港澳自然人独资','外国法人独资','台港澳法人独资','非自然人投资或控股的法人独资'], '独资'))
    Control_Dict.update(dict.fromkeys(['中外合资','台港澳与境内合资','台港澳合资','台港澳与境内合资','台港澳与外国投资者合资',''], '合资'))
    Control_Dict.update(dict.fromkeys([''], ''))
    Control_Dict.update(dict.fromkeys(['非上市'], '非上市'))
    Control_Dict.update(dict.fromkeys([''], ''))
    outfile['企业控制人类型'] = outfile['企业控制人类型'].map(Title_Dict)
    #sns.barplot(x="企业控制人类型", y="是否老赖", data=outfile, palette='Set3')
    plt.show()



    #行业门类代码
    #sns.barplot(x="行业门类代码", y="是否老赖", data=outfile, palette='Set3')
    plt.show()


    #成立日期
    outfile['哪年成立'] = outfile['成立日期'].apply(lambda x:x.split('/')[0])
    print(outfile['哪年成立'])
    sns.barplot(x="哪年成立", y="是否老赖", data=outfile, palette='Set3')
    plt.show()

    outfile.to_csv('outfile.csv', index=False)


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

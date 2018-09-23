# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from code.lib.currency_exchange import currency_exchange
from code.lib.utils import normalize
import seaborn as sns

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'SimHei'


pd.options.display.max_rows = 10
pd.options.display.max_columns = 50


from code.na_shui_fei_zheng_chang_hu import na_shui_fei_zheng_change_hu
from code.xing_zheng_wei_fa_ji_lu import xing_zheng_wei_fa_ji_lu
from code.qian_shui_ming_dan import qian_shui_ming_dan
from code.xian_zhi_gao_xiao_fei_ming_dan import xian_zhi_gao_xiao_fei_ming_dan
from code.ji_ben_xin_xi import ji_ben_xin_xi


# def ji_ben_xin_xi():
#     data_ji_ben_xin_xi = pd.read_csv('../data/train/1.csv',
#                                      keep_default_na=False,
#                                      na_values=[""])
#
#
#     #将老赖的信息并入基本信息中
#     lao_lai = pd.read_csv('../data/train/8.csv',
#                                      keep_default_na=False,
#                                      na_values=[""])
#     lao_lai.insert(1,'是否老赖',1)
#     outfile = pd.merge(data_ji_ben_xin_xi, lao_lai, how='left', left_on='小微企业ID', right_on='小微企业ID')
#     def clear_Nan(s):
#         if pd.isna(s):
#             return 0
#         else:
#             return 1
#     outfile['是否老赖'] = outfile['是否老赖'].apply(clear_Nan)
#
#     #outfile.info()
#     #print(outfile.describe())
#
#     fig = plt.figure()
#     fig.set(alpha=0.2)
#     plt.subplot2grid((2, 3), (0, 0))
#
#
#     outfile['注册资本(金)币种'].value_counts().plot(kind='bar')
#
#
#     #分析注册资金
#     def money_label(s):
#         if (s <= 1000):
#             return 1
#         elif ((s > 1000) & (s <= 5000)):
#             return 2
#         elif ((s > 5000) & (s <= 10000)):
#             return 3
#         elif ((s > 10000) & (s <= 20000)):
#             return 4
#         elif ((s > 20000) & (s <= 30000)):
#             return 5
#         elif (s > 30000):
#             return 6
#     outfile['注册资金分段'] = outfile['注册资金(万元)'].apply(money_label)
#     #sns.barplot(x="注册资金分段", y="是否老赖", data=outfile, palette='Set3')
#     plt.show()
#
#
#
#
#     #分析注册资金类型
#     #sns.barplot(x="注册资本(金)币种", y="是否老赖", data=outfile, palette='Set3')
#     plt.show()
#
#
#     #企业性质
#     outfile['企业性质'] = outfile['企业(机构)类型'].apply(lambda x:x.split('(')[0])
#     Title_Dict = {}
#     Title_Dict.update(dict.fromkeys(['有限责任公司', '有限责任', '有限责任公司分公司'], '有限责任公司'))
#     Title_Dict.update(dict.fromkeys(['集体所有制'], '集体所有制'))
#     Title_Dict.update(dict.fromkeys(['股份合作制'], '股份合作制'))
#     Title_Dict.update(dict.fromkeys(['股份有限公司', '股份','股份有限','股份有限公司分公司'], '股份有限公司'))
#     Title_Dict.update(dict.fromkeys(['农民专业合作经济组织'], '农民专业合作经济组织'))
#     Title_Dict.update(dict.fromkeys(['集体分支机构'], '集体分支机构'))
#     Title_Dict.update(dict.fromkeys(['全民所有制', '全民所有制分支机构'], '全民所有制'))
#     Title_Dict.update(dict.fromkeys(['联营'], '联营'))
#     Title_Dict.update(dict.fromkeys(['外商投资企业分公司'], '外商投资企业分公司'))
#     Title_Dict.update(dict.fromkeys(['个人独资企业'], '个人独资企业'))
#     Title_Dict.update(dict.fromkeys(['台、港、澳投资企业分公司', '投资企业分公司'], '台、港、澳投资企业分公司'))
#     outfile['企业性质'] = outfile['企业性质'].map(Title_Dict)
#     #sns.barplot(x="企业性质", y="是否老赖", data=outfile, palette='Set3')
#     plt.show()
#
#
#     #企业控制人类型 (未完,从独资合资角度分析，从国内国外组成成分分析)
#     outfile['企业控制人类型'] = outfile['企业(机构)类型'].apply(lambda x:x.split('(')[0])
#     Control_Dict = {}
#     Control_Dict.update(dict.fromkeys(['自然人投资或控股', '自然人独资', '外国自然人独资','台港澳自然人独资','外国法人独资','台港澳法人独资','非自然人投资或控股的法人独资'], '独资'))
#     Control_Dict.update(dict.fromkeys(['中外合资','台港澳与境内合资','台港澳合资','台港澳与境内合资','台港澳与外国投资者合资',''], '合资'))
#     Control_Dict.update(dict.fromkeys([''], ''))
#     Control_Dict.update(dict.fromkeys(['非上市'], '非上市'))
#     Control_Dict.update(dict.fromkeys([''], ''))
#     outfile['企业控制人类型'] = outfile['企业控制人类型'].map(Title_Dict)
#     #sns.barplot(x="企业控制人类型", y="是否老赖", data=outfile, palette='Set3')
#     plt.show()
#
#
#
#     #行业门类代码
#     #sns.barplot(x="行业门类代码", y="是否老赖", data=outfile, palette='Set3')
#     plt.show()
#
#
#     #成立日期
#     outfile['哪年成立'] = outfile['成立日期'].apply(lambda x:x.split('/')[0])
#     #sns.barplot(x="哪年成立", y="是否老赖", data=outfile, palette='Set3')
#     plt.show()
#
#
#
#
#     #从业人数
#     def people_label(s):
#         if (s <= 10):
#             return 1
#         elif ((s > 10) & (s <= 20)):
#             return 2
#         elif ((s > 20) & (s <= 30)):
#             return 3
#         elif ((s > 30) & (s <= 40)):
#             return 4
#         elif ((s > 40) & (s <= 50)):
#             return 5
#         elif ((s > 50) & (s <= 60)):
#             return 6
#         elif ((s > 60) & (s <= 70)):
#             return 7
#         elif ((s > 70) & (s <= 80)):
#             return 8
#         elif ((s > 80) & (s <= 90)):
#             return 9
#         elif ((s > 90) & (s <= 100)):
#             return 10
#         elif ((s > 100) & (s <= 110)):
#             return 11
#         elif ((s > 110) & (s <= 120)):
#             return 12
#         elif ((s > 120) & (s <= 130)):
#             return 13
#         elif ((s > 130) & (s <= 140)):
#             return 14
#         elif ((s > 140) & (s <= 150)):
#             return 15
#         elif ((s > 150) & (s <= 160)):
#             return 16
#         elif ((s > 160) & (s <= 170)):
#             return 17
#         elif (s > 170):
#             return 18
#     outfile['从业人数分段'] = outfile['从业人数'].apply(people_label)
#     #sns.barplot(x="从业人数分段", y="是否老赖", data=outfile, palette='Set3')
#     plt.show()
#
#
#     #投资总额(万元)
#     outfile['投资总额分段'] = outfile['投资总额(万元)'].apply(money_label)
#     #sns.barplot(x="投资总额分段", y="是否老赖", data=outfile, palette='Set3')
#     plt.show()
#
#
#     print(outfile['投资总额币种'])
#
#
#     outfile.to_csv('outfile.csv', index=False)
#
#     print(data_ji_ben_xin_xi.head())
#
#     currency_exchange(data_ji_ben_xin_xi, '注册资金(万元)', '注册资本(金)币种')
#     print(data_ji_ben_xin_xi)
#     exit(1)
#     data_ji_ben_xin_xi['注册资金(万元)'].plot(kind='line', title='注册资金（万元）人民币')
#     plt.show(block=True)
#     data_ji_ben_xin_xi = data_ji_ben_xin_xi.drop(columns=['注册资本(金)币种'])
#     data_ji_ben_xin_xi['注册资金(万元)'] = normalize(data_ji_ben_xin_xi['注册资金(万元)'])
#     print(data_ji_ben_xin_xi.head())
#
#     print(data_ji_ben_xin_xi['投资总额币种'].unique())
#     currency_exchange(data_ji_ben_xin_xi, '投资总额(万元)', '投资总额币种')
#     data_ji_ben_xin_xi['投资总额(万元)'].plot(kind='line', title='投资总额(万元)人民币')
#     plt.show(block=True)
#     data_ji_ben_xin_xi = data_ji_ben_xin_xi.drop(columns=['投资总额币种'])
#     data_ji_ben_xin_xi['投资总额(万元)'] = normalize(data_ji_ben_xin_xi['投资总额(万元)'])
#     print(data_ji_ben_xin_xi.head())
#
#     print(data_ji_ben_xin_xi['企业(机构)类型'].unique())
#     data_ji_ben_xin_xi['企业(机构)类型'], _ = pd.factorize(data_ji_ben_xin_xi['企业(机构)类型'])
#     print(data_ji_ben_xin_xi['企业(机构)类型'].unique())


def data_process():

    data_ji_ben_xin_xi = ji_ben_xin_xi()
    data_qian_shui_ming_dan = qian_shui_ming_dan()
    data_na_shui_fei_zheng_chang_hu = na_shui_fei_zheng_change_hu()
    data_xing_zheng_wei_fa_ji_lu = xing_zheng_wei_fa_ji_lu()
    data_xian_zhi_gao_xiao_fei_ming_dan = xian_zhi_gao_xiao_fei_ming_dan()

    # 将数据合并到一个二维数组里面
    all_info = pd.merge(left=data_ji_ben_xin_xi, right=data_qian_shui_ming_dan, how='left',
                          left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_na_shui_fei_zheng_chang_hu, how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_xing_zheng_wei_fa_ji_lu, how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_xian_zhi_gao_xiao_fei_ming_dan, how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    # print(all_info)

    # 将老赖的信息并入基本信息中
    lao_lai = pd.read_csv('../data/train/8.csv',
                                     keep_default_na=False,
                                     na_values=[""])
    lao_lai['是否老赖'] = 1
    all_info = pd.merge(all_info, lao_lai, how='left', left_on='小微企业ID', right_on='小微企业ID')
    all_info['是否老赖'] = all_info['是否老赖'].fillna(0)
    all_info = all_info.fillna(0)
    # print(all_info)

    return all_info


def build_model(init_size):
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(keras.layers.Dense(init_size, activation='relu'))
    # Add another:
    model.add(keras.layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(keras.layers.Dense(10, activation='relu'))

    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    all_info = data_process()
    all_info.to_csv('./process.csv', index=False)
    all_info = pd.read_csv('./process.csv')

    # print(all_info.shape)
    label = pd.get_dummies(all_info['是否老赖'])
    data = all_info.drop(columns=['小微企业ID', '是否老赖'])

    msk = np.random.random(len(data)) < 0.8
    train_label = label[msk]
    train_data = data[msk]

    test_label = label[~msk]
    test_data = data[~msk]

    # print(train_data.shape, test_data.shape)

    model = build_model(47)
    model.fit(train_data.values, train_label.values, epochs=1, batch_size=10)

    test_loss, test_acc = model.evaluate(test_data.values, test_label.values)

    print('Test accuracy:', test_acc)

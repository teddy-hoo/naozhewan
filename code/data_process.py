# -*- coding: utf-8 -*-

import pandas as pd
from code.na_shui_fei_zheng_chang_hu import na_shui_fei_zheng_change_hu
from code.xing_zheng_wei_fa_ji_lu import xing_zheng_wei_fa_ji_lu
from code.qian_shui_ming_dan import qian_shui_ming_dan
from code.xian_zhi_gao_xiao_fei_ming_dan import xian_zhi_gao_xiao_fei_ming_dan
from code.ji_ben_xin_xi import ji_ben_xin_xi
from code.cai_pan_wen_shu import cai_pan_wen_shu
from code.shen_pan_liu_cheng import shen_pan_liu_cheng

import numpy as np
import tensorflow as tf
from tensorflow import keras

# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = 'SimHei'


pd.options.display.max_rows = 10
pd.options.display.max_columns = 50


def data_process():

    data_ji_ben_xin_xi = ji_ben_xin_xi()
    data_qian_shui_ming_dan = qian_shui_ming_dan()
    data_na_shui_fei_zheng_chang_hu = na_shui_fei_zheng_change_hu()
    data_xing_zheng_wei_fa_ji_lu = xing_zheng_wei_fa_ji_lu()
    data_xian_zhi_gao_xiao_fei_ming_dan = xian_zhi_gao_xiao_fei_ming_dan()
    data_cai_pan_wen_shu = cai_pan_wen_shu()
    data_shen_pan_liu_cheng = shen_pan_liu_cheng()

    # 将数据合并到一个二维数组里面
    all_info = pd.merge(left=data_ji_ben_xin_xi, right=data_qian_shui_ming_dan, how='left',
                          left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_na_shui_fei_zheng_chang_hu, how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_xing_zheng_wei_fa_ji_lu, how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_xian_zhi_gao_xiao_fei_ming_dan, how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_cai_pan_wen_shu, how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_shen_pan_liu_cheng, how='left',
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
    model.add(keras.layers.Dense(init_size, activation='relu'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
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
    lao_lai_data = all_info.loc[all_info['是否老赖'] == 1]
    m = np.random.random(len(all_info)) < 0.05
    all_info = all_info[m]
    all_info = pd.concat([all_info, lao_lai_data])
    label = pd.get_dummies(all_info['是否老赖'])
    data = all_info.drop(columns=['小微企业ID', '是否老赖'])
    print(data.shape)
    # data = data.sample(frac=1)
    # print(data.head())
    data.to_csv('./sample.csv')

    msk = np.random.random(len(data)) < 0.8
    train_label = label[msk]
    train_data = data[msk]

    test_label = label[~msk]
    test_data = data[~msk]

    # print(train_data.shape, test_data.shape)

    model = build_model(196)
    model.fit(train_data.values, train_label.values, epochs=20, batch_size=10000)

    test_loss, test_acc = model.evaluate(test_data.values, test_label.values)

    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_acc)

    predictions = model.predict(test_data.values)

    print(test_label.shape, predictions.shape)
    for i in range(0, len(test_label.values)):
        print('真实数据: ', test_label.values[i][0], test_label.values[i][1])
        print('预测数据: ', predictions[i][0], predictions[i][1])


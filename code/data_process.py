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

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'


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

    # 只使用特征比较齐全的数据
    ids = pd.concat([data_qian_shui_ming_dan['小微企业ID'],
                     data_na_shui_fei_zheng_chang_hu['小微企业ID'],
                     data_xing_zheng_wei_fa_ji_lu['小微企业ID'],
                     data_xian_zhi_gao_xiao_fei_ming_dan['小微企业ID'],
                     data_cai_pan_wen_shu['小微企业ID'],
                     data_shen_pan_liu_cheng['小微企业ID']
                     ])

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
    # # print(all_info)

    # 将老赖的信息并入基本信息中
    lao_lai = pd.read_csv('../data/train/8.csv',
                                     keep_default_na=False,
                                     na_values=[""])
    lao_lai['是否老赖'] = 1
    all_info = pd.merge(all_info, lao_lai, how='left', left_on='小微企业ID', right_on='小微企业ID')
    all_info['是否老赖'] = all_info['是否老赖'].fillna(0)
    all_info = all_info.fillna(0)
    # print(all_info)

    ids = pd.concat([ids, lao_lai['小微企业ID']])
    # print(all_info.shape)
    all_info = all_info.loc[all_info['小微企业ID'].isin(ids)]

    # lao_lai_data = all_info.loc[all_info['是否老赖'] == 1]

    # print(all_info.shape)
    # print(lao_lai_data.shape)
    # print(lao_lai.shape)
    # exit()

    return all_info


# def build_model(shape):
#     model = keras.Sequential([
#         keras.layers.Dense(128, activation=tf.nn.relu,
#                            input_shape=(shape,)),
#         keras.layers.Dropout(0.5),
#         keras.layers.Dense(64, activation=tf.nn.relu),
#         keras.layers.Dropout(0.5),
#         keras.layers.Dense(32, activation=tf.nn.relu),
#         keras.layers.Dropout(0.5),
#         keras.layers.Dense(2)
#       ])
#
#     optimizer = tf.train.AdamOptimizer(0.001)
#
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer=optimizer,
#                   metrics=['accuracy'])
#     return model


def build_model(init_size):
    model = keras.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', input_shape=(init_size, )))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


EPOCHS = 200


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['loss']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_loss']),
           label='Val Loss')
  plt.plot(history.epoch, np.array(history.history['acc']),
           label='Train Acc')
  plt.plot(history.epoch, np.array(history.history['val_acc']), label='Val Acc')
  plt.legend()
  # plt.ylim([0, 1])
  plt.show()


if __name__ == '__main__':
    # all_info = data_process()
    # all_info.to_csv('./process.csv', index=False)
    all_info = pd.read_csv('./process.csv')
    print(all_info.shape)
    lao_lai_data = all_info.loc[all_info['是否老赖'] == 1]
    print(lao_lai_data.shape)
    all_info = all_info.loc[all_info['是否老赖'] != 1]
    print(all_info.shape)
    # 让老赖和非老赖的数据量平均一些
    m = np.random.random(len(all_info)) < 0.2
    all_info = all_info[m]
    all_info = pd.concat([all_info, lao_lai_data])
    all_info = all_info.sample(frac=1)
    all_info.to_csv('./sample.csv')
    label = pd.get_dummies(all_info['是否老赖'])
    data = all_info.drop(columns=['小微企业ID', '是否老赖'])
    print(data.shape)
    data = data.sample(frac=1)

    msk = np.random.random(len(data)) < 0.8
    train_label = label[msk]
    train_data = data[msk]

    test_label = label[~msk]
    test_data = data[~msk]

    # print(train_data.shape, test_data.shape)

    model = build_model(len(train_data.columns))
    # model.summary()

    history = model.fit(train_data.values, train_label.values, epochs=EPOCHS, validation_split=0.2, verbose=0)
    plot_history(history)

    # model.fit(train_data.values, train_label.values, epochs=10, batch_size=1000)

    test_loss, test_acc = model.evaluate(test_data.values, test_label.values)

    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_acc)

    predictions = model.predict(test_data.values)

    print(test_label.shape, predictions.shape)
    for i in range(0, len(test_label.values)):
        if test_label.values[i][1] == 0:
            continue
        print('真实数据: ', test_label.values[i][0], test_label.values[i][1])
        print('预测数据: ', predictions[i][0], predictions[i][1])

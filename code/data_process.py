# -*- coding: utf-8 -*-

import pandas as pd
from code.na_shui_fei_zheng_chang_hu import na_shui_fei_zheng_change_hu
from code.xing_zheng_wei_fa_ji_lu import xing_zheng_wei_fa_ji_lu
from code.qian_shui_ming_dan import qian_shui_ming_dan
from code.xian_zhi_gao_xiao_fei_ming_dan import xian_zhi_gao_xiao_fei_ming_dan
from code.ji_ben_xin_xi import ji_ben_xin_xi
from code.cai_pan_wen_shu import cai_pan_wen_shu
from code.shen_pan_liu_cheng import shen_pan_liu_cheng
from code.lib.utils import normalize
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'


pd.options.display.max_rows = 10
pd.options.display.max_columns = 50


def train_data_process():

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

    return all_info


def test_data_process():
    data_ji_ben_xin_xi = ji_ben_xin_xi('../data/test/')
    data_qian_shui_ming_dan = qian_shui_ming_dan('../data/test/')
    data_na_shui_fei_zheng_chang_hu = na_shui_fei_zheng_change_hu('../data/test/')
    data_xing_zheng_wei_fa_ji_lu = xing_zheng_wei_fa_ji_lu('../data/test/')
    data_xian_zhi_gao_xiao_fei_ming_dan = xian_zhi_gao_xiao_fei_ming_dan('../data/test/')
    data_cai_pan_wen_shu = cai_pan_wen_shu('../data/test/')
    data_shen_pan_liu_cheng = shen_pan_liu_cheng('../data/test/')

    # 将数据合并到一个二维数组里面
    all_info = pd.merge(left=data_ji_ben_xin_xi, right=data_qian_shui_ming_dan,
                        how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_na_shui_fei_zheng_chang_hu,
                        how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_xing_zheng_wei_fa_ji_lu,
                        how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info,
                        right=data_xian_zhi_gao_xiao_fei_ming_dan, how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_cai_pan_wen_shu, how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = pd.merge(left=all_info, right=data_shen_pan_liu_cheng,
                        how='left',
                        left_on='小微企业ID', right_on='小微企业ID')
    all_info = all_info.fillna(0)

    return all_info


def get_train_data():
    all_info = pd.read_csv('../data/processed/train_data.csv')
    all_info = all_info.sample(frac=1)
    label = pd.get_dummies(all_info['是否老赖']).values
    data = all_info.drop(columns=['小微企业ID', '是否老赖'])
    data = normalize(data.values)
    return data, label


def get_test_data():
    all_info = pd.read_csv('../data/processed/test_data.csv')
    ids = all_info['小微企业ID'].values
    data = all_info.drop(columns=['小微企业ID'])
    data = normalize(data.values)
    return data, ids


if __name__ == '__main__':
    train = train_data_process()
    print(train.shape)
    train.to_csv('../data/processed/train_data.csv', index=False)

    test = test_data_process()
    print(test.shape)
    test.to_csv('../data/processed/test_data.csv', index=False)

    # print(train.columns.values)
    # print(test.columns.values)
    # print(np.setdiff1d(train.columns.values, test.columns.values))
    # print(np.setdiff1d(test.columns.values, train.columns.values))

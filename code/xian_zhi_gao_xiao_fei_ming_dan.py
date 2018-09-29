# -*- coding: utf-8 -*-

import pandas as pd


def xian_zhi_gao_xiao_fei_ming_dan(path_prefix='../data/train/'):
    """
    限制高消费名单 数据处理
    :return:
    """

    data = pd.read_csv(path_prefix + '7.csv')

    # print(data)
    data = data.drop(columns=['执行法院', '执行内容', '日期类别', '具体日期'])
    data['限制高消费'] = 1
    data = data.groupby('小微企业ID').sum().reset_index()
    # print(data)
    # print(data)

    return data


if __name__ == '__main__':
    data = xian_zhi_gao_xiao_fei_ming_dan()
    data.to_csv('../data/processed/7.csv')
    data_t = xian_zhi_gao_xiao_fei_ming_dan('../data/test/')
    data_t.to_csv('../data/processed/7_test.csv')

# -*- coding: utf-8 -*-

import pandas as pd
from code.lib.utils import normalize


def xian_zhi_gao_xiao_fei_ming_dan():
    """
    限制高消费名单 数据处理
    :return:
    """

    data = pd.read_csv('../data/train/7.csv')

    # print(data)
    data = data.drop(columns=['执行法院', '执行内容', '日期类别', '具体日期'])
    data['限制高消费'] = 1
    data = data.groupby('小微企业ID').sum().reset_index()
    # print(data)
    data['限制高消费'] = normalize(data['限制高消费'])
    # print(data)

    return data


if __name__ == '__main__':
    xian_zhi_gao_xiao_fei_ming_dan()

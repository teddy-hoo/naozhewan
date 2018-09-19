# -*- coding: utf-8 -*-

import pandas as pd

basic_info = pd.read_csv('../data/train/企业基本信息1.csv')
qian_shui_ming_dan = pd.read_csv('../data/train/欠税名单5.csv')

# print(basic_info.sample())
#
# print(basic_info.小微企业ID.unique)

print(qian_shui_ming_dan.所欠税种.unique())

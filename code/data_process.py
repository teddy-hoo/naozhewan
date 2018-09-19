# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager

basic_info = pd.read_csv('../data/train/企业基本信息1.csv', keep_default_na=False, na_values=[""])
qian_shui_ming_dan = pd.read_csv('../data/train/欠税名单5.csv', keep_default_na=False, na_values=[""])
min_shang_shi_cai_pan_wen_shu = pd.read_csv('../data/train/民商事裁判文书2.csv', keep_default_na=False, na_values=[""])

# 画出来税种的类型，未解决问题中文没显示出来
x = qian_shui_ming_dan.groupby('所欠税种')['小微企业ID'].nunique()
y = x.plot(kind='bar')
plt.show(block=True)

# 将数据合并到一个二维数组里面
print(basic_info)
basic_info = pd.merge(left=basic_info, right=qian_shui_ming_dan, how='left', left_on='小微企业ID', right_on='小微企业ID')
print(basic_info)

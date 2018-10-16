import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt
from code.lib.utils import normalize

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000


def data_analyze():
    data = pd.read_csv('../data/processed/train_data.csv')
    x = data.drop(columns=['是否老赖', '小微企业ID'])
    y = data.是否老赖

    # 是否老赖的比例
    # ax = sns.countplot(y, label="Count")  # M = 212, B = 357
    # plt.show()

    # 数据的分布情况
    # print(x.describe())

    data = x
    data_n_2 = (data - data.mean()) / (data.std())              # standardization
    data = pd.concat([y, data_n_2.iloc[:, 64:]], axis=1)
    data = pd.melt(data, id_vars="是否老赖",
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.violinplot(x="features", y="value", hue="是否老赖", data=data,
                   split=True, inner="quart")
    # sns.swarmplot(x="features", y="value", hue="是否老赖", data=data)
    plt.xticks(rotation=90)
    plt.ylim((-4, 8))

    # sns.jointplot(x.loc[:, 'concavity_worst'], x.loc[:, 'concave points_worst'],
    #               kind="regg", color="#ce1414")

    plt.show()
    """
    分析结果：
    # 基本信息
    1. 行业门类代码
    2. 投资总额（万元）
    # 欠税名单
    3. 所欠税务机关
    # 行政违法记录
    4. 行政处罚次数
    5. 处罚机关_市场
    6. 处罚机关_邮政
    # 
    7. 审判机关_6
    8. 审判机关_7
    9. 文书类型_3
    10. 文书类型_8
    11. 诉讼地位_0
    12. 诉讼地位_10
    13. 审判机关_0
    14. 诉讼地位_15
    15. 审理程序_1
    #
    16. 审判机关_1
    16. 审判机关_2
    18. 公告类型_1
    19. 公告类型_2
    20. 公告类型_3
    21. 公告类型_12
    21. 公告类型_13
    21. 公告类型_14
    21. 公告类型_15
    21. 公告类型_16
    21. 公告类型_17
    21. 公告类型_18
    21. 公告类型_19
    21. 公告类型_20
    21. 公告类型_21
    21. 公告类型_22
    21. 公告类型_23
    21. 公告类型_25
    22. 诉讼地位3_6
    23. 诉讼地位3_7
    """


if __name__ == '__main__':
    data_analyze()

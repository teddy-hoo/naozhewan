import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from code.lib.plot import plot_history, plot_roc

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000


def random_forest():

    x = pd.read_csv('../data/processed/train_data.csv')
    print(x.columns)

    drop_list = [
    '行业门类代码',
    '投资总额(万元)',
    '欠税税务机关',
    '行政处罚次数',
    '处罚机关_市场',
    '处罚机关_邮政',
    '审理机关_6_x',
    '审理机关_7_x',
    '文书类型_3.0',
    '文书类型_8.0',
    '诉讼地位_10',
    '审理机关_0_x',
    '诉讼地位_15',
    '审理程序_1',
    '审理机关_1_x',
    '审理机关_2_x',
    '公告类型_1',
    '公告类型_2',
    '公告类型_3',
    '公告类型_12',
    '公告类型_13',
    '公告类型_14',
    '公告类型_15',
    '公告类型_16',
    '公告类型_17',
    '公告类型_18',
    '公告类型_19',
    '公告类型_21',
    '公告类型_22',
    '公告类型_23',
    '公告类型_25',
    '诉讼地位3_6',
    '诉讼地位3_7'
    ]
    # x = x.drop(drop_list, axis=1)  # do not modify x, we will use it later
    X = np.asarray(x.values[:,2:-1], np.float32)
    Y = np.asarray(x.values[:,-1], np.float32)

    # split data train 70 % and test 30 %
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=42)

    # random forest classifier with n_estimators=10 (default)
    clf_rf = RandomForestClassifier(random_state=43)
    clr_rf = clf_rf.fit(x_train, y_train)

    y_pred = clr_rf.predict(x_test)
    ac = accuracy_score(y_test, y_pred)
    print('Accuracy is: ', ac)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")

    # result_analyze(eval_data, eval_label, predictions)

    print(y_test)
    print(y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    print("Test Auc: ", auc)
    plot_roc(fpr, tpr, auc)
    plt.show()


if __name__ == '__main__':
    random_forest()

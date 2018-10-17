import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from code.lib.plot import plot_history, plot_roc
import seaborn as sns
import matplotlib.pyplot as plt


pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000


def gdbt():

    df = pd.read_csv('../data/processed/train_data.csv')

    X = np.asarray(df.values[:,2:-1], np.float32)
    Y = np.asarray(df.values[:,-1], np.float32)

    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=432)

    x = pd.read_csv('../data/processed/train_data.csv')
    data = x.drop(columns=['小微企业ID']).values

    msk = np.random.random(len(data)) < 0.8

    train_data = data[msk]
    eval_data = data[~msk]
    class_1 = train_data[np.where(train_data[:, -1] == 1)[0]]
    # 调整class为1的样本量来解决样本量不平均的问题
    for i in range(10):
        train_data = np.concatenate((train_data, class_1), axis=0)
    np.random.shuffle(train_data)

    train_label = np.asarray(train_data[:, -1], np.float32)
    train_data = np.asarray(train_data[:, 1:-1], np.float32)
    train_data = preprocessing.scale(train_data)
    # print("train: ", train_data)
    # print("X: ", X_train)
    # print("train: ", train_label)
    # print("y: ", y_train[:1000])
    # exit()

    eval_label = np.asarray(eval_data[:, -1], np.float32)
    eval_data = np.asarray(eval_data[:, 1:-1], np.float32)

    clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01,
            max_depth=1, random_state=432)
    clf.fit(train_data, train_label)

    y_pred = clf.predict(eval_data)
    print("Text Acc: ", accuracy_score(eval_label, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(eval_label, y_pred)
    auc = metrics.auc(fpr, tpr)
    print("Test Auc: ", auc)
    plot_roc(fpr, tpr, auc)
    cm = metrics.confusion_matrix(eval_label, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()


if __name__ == '__main__':
    gdbt()

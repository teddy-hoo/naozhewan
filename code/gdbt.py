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
    print(df.values.shape)

    print(df.values)
    X = np.asarray(df.values[:,2:-1], np.float32)
    Y = np.asarray(df.values[:,-1], np.float32)

    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=432)

    clf = GradientBoostingClassifier(n_estimators=10000, learning_rate=0.01,
            max_depth=1, random_state=432)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(y_pred[:100])
    print(y_test[:100])
    print(accuracy_score(y_test, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    print("Test Auc: ", auc)
    plot_roc(fpr, tpr, auc)
    cm = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()


if __name__ == '__main__':
    gdbt()

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


def svm_al():

    df = pandas.read_csv('../data/processed/train_data.csv')

    X = np.asarray(df.values[:,2:-1], np.float32)
    Y = np.asarray(df.values[:,-1], np.float32)

    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    clf = svm.SVC(kernel='sigmoid', gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    # print(fpr)
    # print(tpr)
    auc = metrics.auc(fpr, tpr)
    print("Test Auc: ", auc)
    # plot_roc(fpr, tpr, auc)

    print(np.count_nonzero(y_test))
    print(np.count_nonzero(y_pred))
    cm = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()


if __name__ == "__main__":
    svm_al()

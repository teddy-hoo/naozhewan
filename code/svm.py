import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


def svm_al():

    df = pandas.read_csv('../data/processed/train_data.csv')

    X = np.asarray(df.values[:,2:-1], np.float32)
    Y = np.asarray(df.values[:,-1], np.float32)

    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    clf = svm.SVC(gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    svm_al()

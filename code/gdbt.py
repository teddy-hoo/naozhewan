import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier


def gdbt():

    df = pandas.read_csv('../data/processed/train_data.csv')

    X = np.asarray(df.values[:,2:-1], np.float32)
    Y = np.asarray(df.values[:,-1], np.float32)

    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,
            max_depth=1, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    PC = len(y_test[y_test[:] == 1])
    NC = len(y_test[y_test[:] == 0])
    TP = 0
    FP = 0
    for i in range(0, len(y_pred)):
        if y_test[i] == 1 and y_pred[i] > 0:
            # print('TP: ', predictions[i][0], ' ', predictions[i][1])
            TP += 1
        if y_test[i] == 0 and y_pred[i] > 0:
            # print('FP: ', predictions[i][0], ' ', predictions[i][1])
            FP += 1
    print('Total: ', len(y_test))
    print('PC: ', PC)
    print('NC: ', NC)
    print('TP: ', TP)
    print('FP: ', FP)


if __name__ == '__main__':
    gdbt()

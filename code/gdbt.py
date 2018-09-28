import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

df = pandas.read_csv('sample.csv')

X = np.asarray(df.values[:,2:-1], np.float32)
Y = np.asarray(df.values[:,-1], np.float32)

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

#clf = svm.SVC(gamma='auto')
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0)
clf.fit(X_train, y_train)

#clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
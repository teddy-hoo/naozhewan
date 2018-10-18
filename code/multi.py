import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import metrics
from code.lib.plot import plot_roc
import seaborn as sns

"""
A comparison of a several classifiers in scikit-learn on synthetic datasets. 
The point of this example is to illustrate the nature of decision boundaries of
different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by these 
examples does not necessarily carry over to real datasets.
Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.
The plots show training points in solid colors and testing points semi-transparent.
The lower right shows the classification accuracy on the test set.
"""

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

x = pd.read_csv('../data/processed/train_data.csv')
data = x.drop(columns=['小微企业ID']).values

msk = np.random.random(len(data)) < 0.8

train_data = data[msk]
eval_data = data[~msk]

print('train data 1s: ', len(np.where(train_data[:, -1] == 1)[0]))
print('train data 0s: ', len(np.where(train_data[:, -1] == 0)[0]))
print('eval data 1s: ', len(np.where(eval_data[:, -1] == 1)[0]))
print('eval data 0s: ', len(np.where(eval_data[:, -1] == 0)[0]))

train_class_0 = train_data[np.where(train_data[:, -1] == 0)[0]]
train_class_1 = train_data[np.where(train_data[:, -1] == 1)[0]]

down_rate = np.random.random(len(train_class_0)) < 0.2

train_class_0 = train_class_0[down_rate]

train_data = train_class_0
for i in range(5):
    train_data = np.concatenate((train_data, train_class_1), axis=0)
np.random.shuffle(train_data)

train_label = np.asarray(train_data[:, -1], np.float32)
train_data = np.asarray(train_data[:, 1:-1], np.float32)
train_data = preprocessing.scale(train_data)

eval_label = np.asarray(eval_data[:, -1], np.float32)
eval_data = np.asarray(eval_data[:, 1:-1], np.float32)
eval_data = preprocessing.scale(eval_data)

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(train_data, train_label)
    score = clf.score(eval_data, eval_label)

    # if hasattr(clf, "decision_function"):
    #     Z = clf.decision_function(X_test)
    # else:
    #     Z = clf.predict_proba(X_test)

    try:
        y_pred = clf.predict_proba(eval_data)
    except:
        continue
    print('alg name: ', name)
    print("Text Acc: ", accuracy_score(eval_label, np.argmax(y_pred, axis=1)))
    fpr, tpr, thresholds = metrics.roc_curve(eval_label, y_pred[:, -1])
    auc = metrics.auc(fpr, tpr)
    print("Test Auc: ", auc)
    plot_roc(fpr, tpr, auc)
    cm = metrics.confusion_matrix(eval_label, np.argmax(y_pred, axis=1))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

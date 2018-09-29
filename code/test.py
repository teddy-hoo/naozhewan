import pandas as pd
from sklearn import metrics
import numpy as np

test_label = np.array([[1, 1], [0, 2]])
predictions = np.array([[0.1, 0.9], [0.9, 0.1]])

print(test_label[:, -1], predictions[:, -1],)

test_label = [0,1,0,0,1,0,1,0,0,0,1,1]
predictions = [0.1, 0.2, 0.1, 0.2, 0.1, 0.7, 0.8, 0.4, 0.5, 0.6, 0.8, 0.3]


fpr, tpr, thresholds = metrics.roc_curve(test_label, predictions)
print("Test Auc: ", metrics.auc(fpr, tpr))

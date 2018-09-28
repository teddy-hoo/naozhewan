# -*- coding: utf-8 -*-


import jenkspy
from sklearn.cluster import KMeans
import numpy as np
data = np.concatenate([np.random.randint(0, 4, 15), np.random.randint(5, 11, 20), np.random.randint(15, 21, 15)])


print(data)
breaks = jenkspy.jenks_breaks(data, nb_class=5)
print(breaks)

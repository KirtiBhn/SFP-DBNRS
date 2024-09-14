import pandas as pd
import statistics
import numpy as np


def NCL(dataset):

    X = dataset.loc[dataset.iloc[:, -1] == 0]
    Y = dataset.loc[dataset.iloc[:, -1] == 1]

    if len(X) > len(Y):
        majority = X
        minority = Y
    else:
        minority = X
        majority = Y

    X = dataset.iloc[:, 0:-1]
    Y = dataset.iloc[:, -1]

    data = []
    distances_majority = []
    R = []

    for i in range(0, len(minority)):
        new_data_point = np.array(X.loc[minority.index[i]])
        distances = np.linalg.norm(X - new_data_point, axis=1)

        k = 3
        n_neighbor_ids = distances.argsort()[1:k+1]

        for j in n_neighbor_ids:
            if(Y[i] == Y[majority.index[0]]):
                data.append(X.loc[i])
                distances_majority.append(distances[i])
                R.append(i)

    R = list(set(R))
    R.sort(reverse=True)
    for i in R:
        dataset.drop(dataset.index[int(i)], inplace=True)

    return dataset

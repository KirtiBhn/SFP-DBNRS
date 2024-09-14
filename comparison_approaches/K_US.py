import pandas as pd
import statistics
import numpy as np
import math

def KUS(dataset):
    X = dataset.loc[dataset.iloc[:, -1] == 0]
    Y = dataset.loc[dataset.iloc[:, -1] == 1]

    if len(X) > len(Y):
        NonDefective = X
        Defective = Y
    else:
        Defective = X
        NonDefective = Y

    k = math.ceil(len(NonDefective)/len(Defective))

    X = dataset.iloc[:, 0:-1]
    Y = dataset.iloc[:, -1]

    To_Be_Removed = pd.DataFrame(columns=['index','membership_count'])
    data = []
    distances_NonDefective = []
    NonDefective_neighbors_ids = []

    for i in range(0,len(Defective)):
        # KNN search function to find n nearest points using eucliedian dist and
        # placing them in colonial neigbhors
        new_data_point = np.array(X.loc[Defective.index[i]])
        distances = np.linalg.norm(X - new_data_point, axis=1)
        # k = 10
        K_neighbors = distances.argsort()[1:k+1]
        # finding the NonDefective colonial neighbors from all the colonial neighbors that
        # were found earlier
        for i in K_neighbors:
            if(Y[i] == Y[NonDefective.index[0]]):
                data.append(X.loc[i])
                distances_NonDefective.append(distances[i])
                NonDefective_neighbors_ids.append(i)

    Neighbors = pd.DataFrame()
    Neighbors[0] = NonDefective_neighbors_ids
    Neighbors[1] = distances_NonDefective

    for j in range(0,len(Neighbors)):
        i = Neighbors.loc[j][0]
        if To_Be_Removed['index'].isin([i]).any().any():
            To_Be_Removed.loc[To_Be_Removed['index']==i,'membership_count']+=1
        else:
            new_row = {'index':i,'membership_count':1}
            To_Be_Removed=To_Be_Removed.append(new_row, ignore_index=True)

    To_Be_Removed = To_Be_Removed.loc[To_Be_Removed['membership_count']>1]
    To_Be_Removed = To_Be_Removed.sort_values(["index"], ascending=False)
    for i in To_Be_Removed['index']:
        dataset.drop(dataset.index[int(i)],inplace=True)
    return dataset     

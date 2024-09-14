import pandas as pd
import statistics
import numpy as np
import math


def NUS(data):
    dataset = data.copy(deep=True)
    X = dataset.loc[dataset.iloc[:, -1] == 0]
    Y = dataset.loc[dataset.iloc[:, -1] == 1]

    if len(X) > len(Y):
        clean = X
        buggy = Y
    else:
        buggy = X
        clean = Y

    k = math.ceil(len(clean)/len(buggy))

    X = dataset.iloc[:, 0:-1]
    Y = dataset.iloc[:, -1]

    to_eliminate = pd.DataFrame(columns=['index', 'membership_count'])
    for i in range(0, len(buggy)):
        # KNN search function to find n nearest points using eucliedian dist and
        # placing them in colonial neigbhors
        new_data_point = np.array(X.loc[buggy.index[i]])
        distances = np.linalg.norm(X - new_data_point, axis=1)

        colonial_neighbor_ids = distances.argsort()[1:k+1]

        data = []
        distances_clean = []
        clean_neighbors_ids = []
        # finding the clean colonial neighbors from all the colonial neighbors that
        # were found earlier

        for i in colonial_neighbor_ids:
            if(Y[i] == Y[clean.index[0]]):
                data.append(X.loc[i])
                distances_clean.append(distances[i])
                clean_neighbors_ids.append(i)

        # Calculating median of all the distances of clean neighbors
        if(len(distances_clean)==0):
            med = 0
        else:
            med = statistics.median(distances_clean)

        df = pd.DataFrame()
        df[0] = clean_neighbors_ids
        df[1] = distances_clean

        Nominated_for_elimination = []
        # if the distance is less than median , then nominate those for elimination
        for i in range(0, len(clean_neighbors_ids)):
            if(df.loc[i][1] <= med):
                Nominated_for_elimination.append(df.loc[i][0])

        # calculating the membership count
        for i in Nominated_for_elimination:
            if to_eliminate['index'].isin([i]).any().any():
                to_eliminate.loc[to_eliminate['index']
                                 == i, 'membership_count'] += 1
            else:
                new_row = {'index': i, 'membership_count': 1}
                to_eliminate = to_eliminate.append(new_row, ignore_index=True)

    final_elimination = to_eliminate.loc[to_eliminate['membership_count'] > 1]
    sorted_df = final_elimination.sort_values(["index"], ascending=False)

    for i in sorted_df['index']:
        dataset.drop(dataset.index[int(i)], inplace=True)

    return dataset

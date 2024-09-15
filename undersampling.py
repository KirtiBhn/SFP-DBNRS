import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler

RAD = 0

def scale_dataset(dataset):
    scaler = MinMaxScaler()
    last_column_name = dataset.columns[-1]
    temp1 = dataset.drop(columns=[last_column_name])
    temp2 = dataset[last_column_name]
    col = temp1.columns
    x = scaler.fit_transform(temp1.to_numpy())
    dataset = pd.DataFrame(x, columns=col)
    dataset = pd.concat([dataset, temp2], axis=1)
    return dataset


def eucliden_distance(x, X):
    i = 0
    sum = 0
    while i < len(x):
        sum += ((x[i]-X[i])**2)
        i += 1
    return math.sqrt(sum)


def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list


def calc_r(df):
    df = df.copy()
    dis_sum = 0
    df = df.sample(frac=1)
    D_random = df.iloc[:, 0:-1]
    count = int(len(D_random)*0.05)
    for i in range(0, count+1):
        data = np.array(D_random.loc[i])
        distances = np.linalg.norm(D_random - data, axis=1)
        distances.sort()
        dis_sum += float(distances[6])
    return dis_sum/count


def find_k_nearest_neighbours(list_df, k, x):
    result = []
    dist_dict = {}
    for i in range(0, len(list_df)):
        dist_dict[i] = eucliden_distance(x, list_df[i])
    new_dict = {k: v for k, v in sorted(
        dist_dict.items(), key=lambda item: item[1])}
    key_list = getList(new_dict)
    for i in range(0, k):
        result.append(list_df[key_list[i]])
    return result


def find_neighbours_in_radius(list_D_neg, r_neg, x):
    inside = []
    for i in list_D_neg:
        d = eucliden_distance(i, x)
        if d <= r_neg:
            inside.append(i)
    return inside

def remove_overlap_new_df(dataset):
    last_column = dataset.iloc[:, -1]
    last_column_name = dataset.columns[-1]
    items_count = last_column.value_counts()

    neg = ''
    neg_count = 0
    pos = ''
    pos_count = 0

    # finding majority and miniority
    if items_count[0] > items_count[1]:
        neg = items_count.keys()[0]
        neg_count = items_count[0]
        pos = items_count.keys()[1]
        pos_count = items_count[1]
    else:
        pos = items_count.keys()[0]
        pos_count = items_count[0]
        neg = items_count.keys()[1]
        neg_count = items_count[1]

    # scaling dataset
    new_df = scale_dataset(dataset)

    # General parameters used in algo
    total_instance = neg_count+pos_count
    imb = neg_count/pos_count
    k = int(math.sqrt(total_instance) + math.sqrt(imb))
    global RAD
    RAD = calc_r(new_df)
    r_neg = 2.5*RAD
    r_pos = 2.5*r_neg
    minp_pos = 2

    # storing pos and neg data points (without taget value)
    temp_new_df = new_df.drop(last_column_name, axis=1)
    temp_D_pos = new_df[new_df[last_column_name] == pos].drop(last_column_name, axis=1)
    temp_D_neg = new_df[new_df[last_column_name] == neg].drop(last_column_name, axis=1)

    # storing all column names
    column_name_list = list(temp_new_df.columns)

    # changing data in form of (list of tuples)
    list_df = [tuple(x) for x in temp_new_df.values.tolist()]
    list_D_pos = [tuple(x) for x in temp_D_pos.values.tolist()]
    list_D_neg = [tuple(x) for x in temp_D_neg.values.tolist()]

    D_overlap = []
    D_noise = []
    
    # print("\n" + str(len(list_D_pos)) + "\n")
    # print("\n" + str(len(list_D_neg)) + "\n")

    # identifying noise and core
    for x in list_D_pos:
        Neighbors_main = find_neighbours_in_radius(list_df, r_pos, x)
        Neighbors_pos = []
        Neighbors_neg = []
        for item in Neighbors_main:
            if item in list_D_pos:
                Neighbors_pos.append(item)
            else:
                Neighbors_neg.append(item)

        if(len(Neighbors_pos) < minp_pos):
            D_noise.append(x)
        else:
            # this is core
            for y in Neighbors_neg:
                if eucliden_distance(y, x) <= r_pos/4:
                    D_overlap.append(y)

    # removing noise from list_D_pos
    # print("\n Noise ka size = " + str(len(D_noise)))
    # print("\n Before list_D_pos ka size = " + str(len(list_D_pos)))
    for item in D_noise:
        list_D_pos.remove(item)
    # print("\n After list_D_pos ka size = " + str(len(list_D_pos)) + "\n")

    # recursive knn based search
    freq_table = {}
    X = []

    for y in D_overlap:
        new_neighbours = find_k_nearest_neighbours(list_df, k, y)
        for i in new_neighbours:
            if i in list_D_neg:
                if i in freq_table:
                    freq_table[i] += 1
                else:
                    freq_table[i] = 1

    for key in freq_table:
        if freq_table[key] > 1:
            # print("thak gya")
            X.append(key)

    X_final = []

    for z in D_overlap:
        if (z not in X_final):
            X_final.append(z)

    for z in X:
        if (z not in X_final):
            X_final.append(z)

    # removing combined overlap from list_D_neg
    for instance in X_final:
        list_D_neg.remove(instance)
        
    # print("\n" + str(len(list_D_pos)) + "\n")
    # print("\n" + str(len(list_D_neg)) + "\n")

    final_neg_df = pd.DataFrame(list_D_neg, columns=column_name_list)
    final_neg_df = final_neg_df.assign(last_column_name=neg)

    final_pos_df = pd.DataFrame(list_D_pos, columns=column_name_list)
    final_pos_df = final_pos_df.assign(last_column_name=pos)

    frames = [final_pos_df, final_neg_df]
    final_df = pd.concat(frames, ignore_index=True)
    final_df.rename(
        columns={'last_column_name': last_column_name}, inplace=True)

    return final_df

def remove_overlap_new(dataset):
    last_column = dataset.iloc[:, -1]
    last_column_name = dataset.columns[-1]
    items_count = last_column.value_counts()

    neg = ''
    neg_count = 0
    pos = ''
    pos_count = 0

    # finding majority and miniority
    if items_count[0] > items_count[1]:
        neg = items_count.keys()[0]
        neg_count = items_count[0]
        pos = items_count.keys()[1]
        pos_count = items_count[1]
    else:
        pos = items_count.keys()[0]
        pos_count = items_count[0]
        neg = items_count.keys()[1]
        neg_count = items_count[1]

    # scaling dataset
    new_df = scale_dataset(dataset)

    # General parameters used in algo
    total_instance = neg_count+pos_count
    imb = neg_count/pos_count
    k = int(math.sqrt(total_instance) + math.sqrt(imb))
    global RAD
    RAD = calc_r(new_df)
    r_neg = 2.5*RAD
    r_pos = 4*r_neg
    minp_pos = 2

    # storing pos and neg data points (without taget value)
    temp_new_df = new_df.drop(last_column_name, axis=1)
    temp_D_pos = new_df[new_df[last_column_name] == pos].drop(last_column_name, axis=1)
    temp_D_neg = new_df[new_df[last_column_name] == neg].drop(last_column_name, axis=1)

    # storing all column names
    column_name_list = list(temp_new_df.columns)

    # changing data in form of (list of tuples)
    list_df = [tuple(x) for x in temp_new_df.values.tolist()]
    list_D_pos = [tuple(x) for x in temp_D_pos.values.tolist()]
    list_D_neg = [tuple(x) for x in temp_D_neg.values.tolist()]

    D_overlap = []
    D_noise = []

    # identifying noise and core
    for x in list_D_pos:
        Neighbors_main = find_neighbours_in_radius(list_df, r_pos, x)
        Neighbors_pos = []
        Neighbors_neg = []
        for item in Neighbors_main:
            if item in list_D_pos:
                Neighbors_pos.append(item)
            else:
                Neighbors_neg.append(item)

        if(len(Neighbors_pos) < minp_pos):
            D_noise.append(x)
        else:
            # this is core
            for y in Neighbors_neg:
                if eucliden_distance(y, x) <= r_pos/4:
                    D_overlap.append(y)

    # removing noise from list_D_pos
    # print("\n Noise ka size = " + str(len(D_noise)))
    # print("\n Before list_D_pos ka size = " + str(len(list_D_pos)))
    for item in D_noise:
        list_D_pos.remove(item)
    # print("\n After list_D_pos ka size = " + str(len(list_D_pos)) + "\n")

    # recursive knn based search
    freq_table = {}
    X = []

    for y in D_overlap:
        new_neighbours = find_k_nearest_neighbours(list_df, k, y)
        for i in new_neighbours:
            if i in list_D_neg:
                if i in freq_table:
                    freq_table[i] += 1
                else:
                    freq_table[i] = 1

    for key in freq_table:
        if freq_table[key] > 1:
            # print("thak gya")
            X.append(key)

    X_final = []

    for z in D_overlap:
        if (z not in X_final):
            X_final.append(z)

    for z in X:
        if (z not in X_final):
            X_final.append(z)

    # removing combined overlap from list_D_neg
    for instance in X_final:
        list_D_neg.remove(instance)

    final_neg_df = pd.DataFrame(list_D_neg, columns=column_name_list)
    final_neg_df = final_neg_df.assign(last_column_name=neg)

    final_pos_df = pd.DataFrame(list_D_pos, columns=column_name_list)
    final_pos_df = final_pos_df.assign(last_column_name=pos)

    frames = [final_pos_df, final_neg_df]
    final_df = pd.concat(frames, ignore_index=True)
    final_df.rename(
        columns={'last_column_name': last_column_name}, inplace=True)

    X_train = final_df.iloc[:, :-1].values
    y_train = final_df.iloc[:, -1].values
    return X_train, y_train


def perform_undersampling_new(dataset):
    last_column = dataset.iloc[:, -1]
    last_column_name = dataset.columns[-1]
    items_count = last_column.value_counts()

    neg = ''
    neg_count = 0
    pos = ''
    pos_count = 0

    # finding majority and miniority
    if items_count[0] > items_count[1]:
        neg = items_count.keys()[0]
        neg_count = items_count[0]
        pos = items_count.keys()[1]
        pos_count = items_count[1]
    else:
        pos = items_count.keys()[0]
        pos_count = items_count[0]
        neg = items_count.keys()[1]
        neg_count = items_count[1]

    # storing pos and neg data points (with taget value)
    D_pos = dataset[dataset[last_column_name] == pos]
    D_neg = dataset[dataset[last_column_name] == neg]

    # General parameters used in algo
    total_instance = neg_count+pos_count
    imb = neg_count/pos_count
    k = int(math.sqrt(total_instance) + math.sqrt(imb))
    r_neg = RAD
    minp_neg = 2
    ngh = int(minp_neg * imb * 1.5)

    # storing pos and neg data points (without taget value)
    temp_new_df = dataset.drop(last_column_name, axis=1)
    temp_D_pos = dataset[dataset[last_column_name] == pos].drop(last_column_name, axis=1)
    temp_D_neg = dataset[dataset[last_column_name] == neg].drop(last_column_name, axis=1)

    # storing all column names
    column_name_list = list(temp_new_df.columns)

    # changing data in form of (list of tuples)
    list_df = [tuple(x) for x in temp_new_df.values.tolist()]
    list_D_pos = [tuple(x) for x in temp_D_pos.values.tolist()]
    list_D_neg = [tuple(x) for x in temp_D_neg.values.tolist()]
    D_noise = []
    visited = set()
    irremovable_core = set()
    
    # print("\nUndersample wala " + str(len(list_D_pos)) + "\n")
    # print("\nUndersample wala" + str(len(list_D_neg)) + "\n")

    for x in list_D_neg:
        if x not in visited:
            visited.add(x)
            neighbours = find_neighbours_in_radius(list_D_neg, r_neg, x)
            num_neighbours = len(neighbours)
            
            # print("\n num Neighbours = " + str(num_neighbours) + "\n")

            if num_neighbours < minp_neg:
                border = x
                num_border = 0

                for y in neighbours:
                    if y == border:
                        num_border += 1

                if num_border == num_neighbours:
                    D_noise = np.append(D_noise, neighbours)

            else:
                removable_core = x
                neighbour_removable = []

                for z in neighbours:
                    if z not in irremovable_core:
                        if z == removable_core:
                            neighbour_removable.append(z)

                        if z not in visited:
                            neighbours_z = find_neighbours_in_radius(list_D_neg, r_neg, z)
                            num_neighbours_z = len(neighbours_z)

                            if num_neighbours_z >= minp_neg:
                                # z must be added in neighbour_removable
                                neighbour_removable.append(z)
                        
                        # print("\nneighbour_removable = " + str(len(neighbour_removable)) + "\n")
                        
                temp = []
                for itr in range(0, len(neighbour_removable)):
                    temp.append([eucliden_distance(neighbour_removable[itr], x), neighbour_removable[itr]])

                temp.sort()
                neighbour_removable.clear()

                for itr in range(0, len(temp)):
                    neighbour_removable.append(temp[itr][1])

                num_remove = num_neighbours - minp_neg
                
                # print("\nRemove hona wala Neighbours = " + str(num_remove) + "\n")
                # print("\nngh ki values = " + str(ngh) + "\n")
                # print("\nneighbour_removable = " + str(len(neighbour_removable)) + "\n")
                
                for k in range(0, min(num_remove,ngh)):
                    if len(neighbour_removable) != 0:
                        selected_data = neighbour_removable.pop(0)
                        for find in list_D_neg:
                            if find == selected_data:
                                list_D_neg.remove(find)

                irremovable_core.add(x)
                for r in neighbour_removable:
                    irremovable_core.add(r)

    # print("\nUndersample wala " + str(len(list_D_pos)) + "\n")
    # print("\nUndersample wala" + str(len(list_D_neg)) + "\n")
    
    final_neg_df = pd.DataFrame(list_D_neg, columns=column_name_list)
    final_neg_df = final_neg_df.assign(last_column_name=neg)

    final_pos_df = pd.DataFrame(list_D_pos, columns=column_name_list)
    final_pos_df = final_pos_df.assign(last_column_name=pos)

    frames = [final_pos_df, final_neg_df]
    final_df = pd.concat(frames, ignore_index=True)
    final_df.rename(
        columns={'last_column_name': last_column_name}, inplace=True)

    X_train = final_df.iloc[:, :-1].values
    y_train = final_df.iloc[:, -1].values
    return X_train, y_train

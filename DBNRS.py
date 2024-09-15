# Importing the libraries
# import shapegraph
import matplotlib.pyplot as plt

from math import pi
from math import sqrt
import sys
from imblearn.metrics import geometric_mean_score

# from sklearn.cluster import DBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score as roc
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.pairwise import euclidean_distances as e_dist
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as NaiveBayes
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

from undersampling import remove_overlap_new
from undersampling import perform_undersampling_new
from undersampling_old import perform_undersampling_old
from undersampling_old import remove_overlap_old
from undersampling import remove_overlap_new_df
from undersampling_old import remove_overlap_old_df

import warnings
import numpy as np
import os
import pandas as pd
from math import ceil

from numpy import average, std
from statistics import mean
import ast

base_path = os.path.dirname(os.path.abspath(__file__))
results_folder = "results_folder"
results_base_path = os.path.join(base_path, results_folder)

# db_scan_analysis_csv_path = os.path.join(
#     results_base_path, "db_scan_fx_metric_analysis.csv"
# )
# open(db_scan_analysis_csv_path, "w+").write(
#     f"fx,dataset_name,apr,pass_no,epsilon,min_samples_dbscan\n"
# )

sys.path.append(base_path)

fx_tuple = ("f1",)

datasets = [
    #"mw1",
    #"kc3",
    "cm1",
    #"kc2",
    #"pc1",
    #"pc4",
    #"pc3",
    #"kc1",
    # "pc2",
    # "jm1",
    # "jedit-4.2",
    # "jedit-4.1",
    #"camel-1.6",
    #"camel-1.4",
    #"prop-2",
    #"prop-3",
    #"prop-4",
]

approaches = [
    #"Normal",
    # "DBScan",
    # "ShapeGraph",
    # "DbScan+ShapeGraph",
    #"overlap_old",
    #"undersampling_old",
    #"overlap_old + undersampling_old",
    "DBNRS",
    #"undersampling_new",
    #"overlap_new + undersampling_new",
]

classifier_techniques = [
    "DecisionTreeClassifier",
    "KNN",
    "SVM",
    "NaiveBayes",
    "RandomForest",
    "Boosting",
]


def prob_f_al(confusion_matrix_ravel):
    """
    Calculates the probability of false alarm.
    """
    tn, fp, fn, tp = confusion_matrix_ravel
    return fp / (fp + tn)


def analysis_finder(dataset_df_X, dataset_df_y):
    dataset_train = pd.DataFrame(dataset_df_X)
    dataset_train["defects"] = pd.DataFrame(dataset_df_y)

    dr_vals = list(dataset_train["defects"].unique())

    if (
        dataset_train["defects"].value_counts()[dr_vals[0]]
        >= dataset_train["defects"].value_counts()[dr_vals[1]]
    ):
        mj_ind = 0
        mn_ind = 1
    else:
        mj_ind = 1
        mn_ind = 0

    data_majority = dataset_train[dataset_train.defects == dr_vals[mj_ind]]
    data_minority = dataset_train[dataset_train.defects == dr_vals[mn_ind]]

    analysis_finder_dict = {
        "Total": len(dataset_train),
        "Majorty": len(data_majority),
        "Minority": len(data_minority),
        "IR": (len(data_majority) / len(data_minority)),
        "Majority_Class": dr_vals[mj_ind],
        "Minority_Class": dr_vals[mn_ind],
    }

    return analysis_finder_dict


if __name__ == "__main__":
    # Importing the dataset
    IRNow = 1

    project_base_path = os.path.join(base_path, "data")

    for fx in fx_tuple:
        print("Using Formula:", fx)
        try:
            # if True:
            results_fx_base_path = os.path.join(results_base_path, fx)
            warnings.filterwarnings("ignore")

            results = dict()
            analysis_dict = dict()

            for dsstr in datasets:
                print(dsstr)
                if dsstr not in analysis_dict:
                    analysis_dict[dsstr] = dict()

                for apr in approaches:
                    print("\t", apr)
                    if apr not in analysis_dict[dsstr]:
                        analysis_dict[dsstr][apr] = dict()

                    dataset = pd.read_csv(f"{project_base_path}/{dsstr}.csv")
                    column_name_list = list(dataset.columns)

                    # Creating the training set
                    X = dataset.iloc[:, :-1].values
                    y = dataset.iloc[:, -1].values
                    analysis_dict[dsstr][apr]["Initial"] = analysis_finder(X, y)

                    X_df = pd.DataFrame(X, columns=column_name_list[:-1])
                    y_df = pd.DataFrame(y, columns=column_name_list[-1:])
                    dataset1 = pd.concat([X_df, y_df], axis=1)

                    if apr == "Normal":
                        X_new, y_new = X, y
                        analysis_dict[dsstr][apr]["Final_Same_as_Initial"] = (
                            analysis_finder(X_new, y_new)
                        )
                    elif apr == "overlap_old":
                        X_new, y_new = remove_overlap_old(dataset1)
                        analysis_dict[dsstr][apr]["Final_After_Oversampling"] = (
                            analysis_finder(X_new, y_new)
                        )
                    elif apr == "undersampling_old":
                        X_new, y_new = perform_undersampling_old(dataset1)
                        analysis_dict[dsstr][apr]["Final_After_Undersampling"] = (
                            analysis_finder(X_new, y_new)
                        )
                    elif apr == "overlap_old + undersampling_old":
                        temp1 = remove_overlap_old_df(dataset1)
                        X_new, y_new = perform_undersampling_old(temp1)
                        analysis_dict[dsstr][apr]["Final_After_Undersampling"] = (
                            analysis_finder(X_new, y_new)
                        )
                    elif apr == "DBNRS":
                        X_new, y_new = remove_overlap_new(dataset1)
                        analysis_dict[dsstr][apr]["Final_After_Overlap"] = (
                            analysis_finder(X_new, y_new)
                        )
                    elif apr == "undersampling_new":
                        X_new, y_new = perform_undersampling_new(dataset1)
                        analysis_dict[dsstr][apr]["Final_After_Undersampling"] = (
                            analysis_finder(X_new, y_new)
                        )
                    elif apr == "overlap_new + undersampling_new":
                        temp2 = remove_overlap_new_df(dataset1)
                        X_new, y_new = perform_undersampling_new(temp2)
                        analysis_dict[dsstr][apr]["Final_After_Undersampling"] = (
                            analysis_finder(X_new, y_new)
                        )

                    for classifier_str in classifier_techniques:
                        print("\t\t", classifier_str, end=":: ")
                        skf = StratifiedKFold(n_splits=5)
                        cm = []
                        mathew_corr_coef_val = []
                        g_m_score = []
                        precision = []
                        recall = []
                        f1 = []
                        kappa = []
                        acc_score = []
                        roc_a = []
                        pfa = []
                        intini = 1

                        X_copy = X_new.copy()
                        y_copy = y_new.copy()
                        # print("Length of X_train:", len(X_train_copy))

                        print("Fold:", end=" ")
                        for train_index, test_index in skf.split(X_copy, y_copy):
                            print(intini, end=";")
                            intini += 1
                            X_train, X_test = (
                                X_copy[train_index],
                                X_copy[test_index],
                            )

                            y_train, y_test = (
                                y_copy[train_index],
                                y_copy[test_index],
                            )

                            try:
                                if classifier_str == "DecisionTreeClassifier":
                                    classifier = DecisionTreeClassifier(
                                        criterion="entropy", random_state=0
                                    )
                                elif classifier_str == "KNN":
                                    classifier = KNN(n_neighbors=5)
                                elif classifier_str == "SVM":
                                    classifier = SVM(random_state=0)
                                elif classifier_str == "NaiveBayes":
                                    classifier = NaiveBayes()
                                elif classifier_str == "RandomForest":
                                    classifier = RandomForest(
                                        n_estimators=10,
                                        criterion="entropy",
                                        random_state=0,
                                    )
                                elif classifier_str == "Boosting":
                                    classifier = AdaBoostClassifier(
                                        n_estimators=10, random_state=0
                                    )

                                classifier.fit(X_train, y_train)
                                y_pred = classifier.predict(X_test)

                                cm.append(
                                    list(confusion_matrix(y_test, y_pred).ravel())
                                )
                                mathew_corr_coef_val.append(
                                    matthews_corrcoef(y_test, y_pred)
                                )
                                g_m_score.append(geometric_mean_score(y_test, y_pred))
                                precision.append(
                                    precision_score(y_test, y_pred, zero_division=0)
                                )
                                recall.append(
                                    recall_score(y_test, y_pred, zero_division=0)
                                )
                                f1.append(f1_score(y_test, y_pred, zero_division=0))
                                kappa.append(cohen_kappa_score(y_test, y_pred))
                                acc_score.append(accuracy_score(y_pred, y_test))
                                pfa.append(
                                    prob_f_al(confusion_matrix(y_test, y_pred).ravel())
                                )

                                try:
                                    roc_a.append(roc(y_test, y_pred, average="micro"))
                                except ValueError:
                                    roc_a.append("N.A.")

                            except Exception as e:
                                print("ClassifierException:", e)

                        print()
                        # print("\nFold wali loop ka bhar")

                        if dsstr not in results:
                            results[dsstr] = dict()
                        if apr not in results[dsstr]:
                            results[dsstr][apr] = dict()
                        if classifier_str not in results[dsstr][apr]:
                            results[dsstr][apr][classifier_str] = {
                                "Accuracy": str(acc_score).replace(",", " "),
                                "Precision": str(precision).replace(",", " "),
                                "Recall": str(recall).replace(",", " "),
                                "F1": str(f1).replace(",", " "),
                                "Cohens kappa": str(kappa).replace(",", " "),
                                "Area Under ROC": str(roc_a).replace(",", " "),
                                "Matthews Correlation Coefficient": str(
                                    mathew_corr_coef_val
                                ).replace(",", " "),
                                "Geometric Mean Score": str(g_m_score).replace(
                                    ",", " "
                                ),
                                "Confusion Matrix": str(cm)
                                .replace("\n", " ")
                                .replace(",", " "),
                                "Probability of False Alarm": str(pfa).replace(
                                    ",", " "
                                ),
                            }

                    # print("Classifier loop\n")

                # print("approaches loop\n")

            # print("dataset loop\n")

            # print("Analyser Results:", analysis_dict)
            csv_str_analysis = "Dataset,Approach,State,TotalCount,MajorityCount,MinorityCount,IR,MajorityClass,MinorityClass"
            for dataset in analysis_dict:
                for apr in analysis_dict[dataset]:
                    for state in analysis_dict[dataset][apr]:
                        csv_str_analysis += f"\n{dataset},{apr},{state},{analysis_dict[dataset][apr][state]['Total']},{analysis_dict[dataset][apr][state]['Majorty']},{analysis_dict[dataset][apr][state]['Minority']},{round(analysis_dict[dataset][apr][state]['IR'],2)},{analysis_dict[dataset][apr][state]['Majority_Class']},{analysis_dict[dataset][apr][state]['Minority_Class']}"

            os.makedirs(os.path.join(results_fx_base_path, "Analysis"), exist_ok=True)

            with open(
                os.path.join(
                    results_fx_base_path, "Analysis", "Dataset_Reduction_Analysis.csv"
                ),
                "w+",
            ) as f:
                f.write(csv_str_analysis)

            print("\nExporting Results", fx)

            tables = [
                #"Accuracy",
                #"Precision",
                "Recall",
                "F1",
                #"Matthews Correlation Coefficient",
                "Geometric Mean Score",
                "Area Under ROC",
                "Confusion Matrix",
                "Probability of False Alarm",
            ]
            domains = approaches

            teckz = classifier_techniques
            for tb in tables:
                c_str = tb + "\nDataset,Classifier," + ",".join(domains) + "\n"
                for x in results:
                    for y in teckz:
                        c_str += x + "," + y
                        for domain in domains:
                            try:
                                c_str += f",{results[x][domain][y][tb]}"
                            except Exception as e:
                                print(f"Error in {tb}:", e)
                                c_str += f",N.A."
                        c_str += "\n"

                with open(
                    os.path.join(results_fx_base_path, tb + ".csv"),
                    "w",
                ) as f:
                    f.write(c_str)

            print("Computing Averages", fx)

            res_len = 5

            plusminus = "\u00b1"

            def conf_matr_avg(list_of_conf_matrices):
                av_list = []
                list_of_conf_matrices = [
                    x for x in list_of_conf_matrices if len(x) == 4
                ]
                for i in range(4):

                    av_list.append(
                        str(mean([x[i] for x in list_of_conf_matrices]))[:res_len]
                        + plusminus
                        + str(std([x[i] for x in list_of_conf_matrices]))[:res_len]
                    )
                return av_list

            for r in os.listdir(results_fx_base_path):
                if r.endswith(".csv"):
                    # print("Average_" + r, end="\t")
                    df = pd.read_csv(os.path.join(results_fx_base_path, r), header=1)
                    if r == "Confusion Matrix.csv":
                        for c in df.columns:
                            if c in ("Dataset", "Classifier"):
                                continue
                            df[c] = df[c].apply(
                                lambda x: str(
                                    conf_matr_avg(
                                        ast.literal_eval(x.replace("  ", ","))
                                    )
                                ).replace(",", " ")
                            )
                    else:
                        for c in df.columns:
                            if c in ("Dataset", "Classifier"):
                                continue
                            try:
                                df[c] = df[c].apply(
                                    lambda x: str(
                                        mean(ast.literal_eval(x.replace("  ", ",")))
                                    )[:res_len]
                                    + plusminus
                                    + str(std(ast.literal_eval(x.replace("  ", ","))))[
                                        :res_len
                                    ]
                                )
                            except:
                                print("Error in ", r, c, df[c])
                    if not os.path.exists(
                        os.path.join(results_fx_base_path, "Averaged/")
                    ):
                        os.makedirs(os.path.join(results_fx_base_path, "Averaged/"))
                    df.to_csv(
                        os.path.join(results_fx_base_path, "Averaged", "average_" + r),
                        index=False,
                    )
            print(
                "Results and Average saved for",
                fx,
                "saved at",
                results_fx_base_path,
                end="\n\n",
            )
        except Exception as e:
            print(f"Failed for {fx} with Exception {e}")

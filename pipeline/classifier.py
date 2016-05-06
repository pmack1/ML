import pandas as pd
import pylab as pl
import evaluate
from pylab import figure, axes, pie, title, show
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
import matplotlib.pyplot as plt
import re


##### create models
'''Much has been  borrowed from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py,
Mistakes are likely my own'''

def define_clfs_params():

    clfs = {
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SGD': SGDClassifier(loss="log", penalty="l2"),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
            }

    grid = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]}
           }
    return clfs, grid

def try_models(X,y, models_to_run):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    clfs,  grid = define_clfs_params()
    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        time_start = time.clock()
        parameter_values = grid[models_to_run[index]]
        model = GridSearchCV(clf, parameter_values)
        y_pred = clf.fit(X_train,  y_train).predict(X_test)
        y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
        # evaluate.print_table(clf,y_test, y_pred, time_start)
        plot_precision_recall_n(y_test,y_pred_probs,clf)





        # for p in ParameterGrid(parameter_values):
        #     try:
        #         clf.set_params(**p)
        #         y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)
        #         y_pred = clf.fit(X_train,  y_train).predict(X_test)
        #         print(clf)
                # accuracy = metrics.accuracy_score(y_test, y_pred)
                # precision = metrics.precision_score(y_test,y_pred)
                # recall = metrics.recall_score(y_test,y_pred)
                # auc = metrics.roc_auc_score(y_test, y_pred)
                # f1score = metrics.f1_score(y_test,y_pred)
                # print("accuracy is", accuracy)
                # print("precision is", precision)
                # print("recall is ", recall)
                # print("auc is", auc)
                # print("f1score is",  f1score)
        #         # threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
        #         # print threshold
        #         # print(precision_at_k(y_test,y_pred_probs,.05))
        #         # plot_precision_recall_n(y_test,y_pred_probs,clf)
        #         print("\n")
        #     except IndexError, e:
        #         print 'Error:',e
        #         continue

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')

    name = str(model_name)
    pattern = "^[a-zA-Z]*"
    match = re.search(pattern, name)
    filename = match.group(0)
    plt.title(filename)
    plt.savefig("graphs/" + filename)


    # def evaluate_accuracy(X_train, X_test, y_train, y_test, clf):
    #         y_pred = classifier.logistic(x_train, y_train, x_test)
    #         return metrics.accuracy_score(y_test, y_pred)










# def nearest_neighbor(train_df, test_df, target_col, n_neighbors):
#     #import model
#     from sklearn.neighbors import KNeighborsClassifier
#     #instantiate model
#     knn = KNeighborsClassifier(n_neighbors = n_neighbors)
#     #fit data
#     y = df_train[df_train.columns[target_col]].as_matrix()
#     del df_train[df_train.columns[target_col]]
#     x = df_train.as_matrix()
#     knn.fit(x,y)

#     #predict
#     del df_test[df_test.columns[target_col]]
#     x_new = df_test.as_matrix()
#     return knn.predict(x_new)



    # df.dropna()

    # df_train = df.iloc[0: int(0.8 *len(df)),]
    # df_test = df.iloc[int(0.8 *len(df)):int(len(df)),]

    # target = df_train[df_train.columns[target_col]].as_matrix()
    # del df_train[df_train.columns[target_col]]
    # data = df_train.as_matrix()

    # knn = neighbors.KNeighborsClassifier()
    # knn.fit(data, target)
    # return

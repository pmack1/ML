import read
import summarize
import process
import evaluate
import pandas as pd
from classifier import Model
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

TARGET_COL = 0

#### read data
df_train = read.read_file('data/cs-training.csv', index_col = 0)
df_test = read.read_file('data/cs-test.csv', index_col = 0)
columns = df_train.columns

#### seperate x and y variables from training and testing sets
x_train = df_train.copy()
y_train = x_train[columns[TARGET_COL]]
del x_train[columns[TARGET_COL]]
x_test = df_test.copy()
del x_test[columns[TARGET_COL]]

#### create histograms for each variable in input data
for col in columns:
    summarize.histogram(df_train,col)

### summary tables
for col in columns:
    summarize.summary_table(df_train, col)

#### fill missing values
mean_list = list(set(columns) - set('MonthlyIncome'))
process.fill_missing(df_train = x_train, df_test = x_test, mean = mean_list, median = 'MonthlyIncome')

##### create models
logistic = Model("logistic", x_train, y_train, x_test)









def define_clfs_params:
    '''Borrowed from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py'''

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3)
            }

    grid = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    return  clfs, grid







# ###### evaluate models on training data by splitting training dataset
# logistic_model_accuracy = evaluate.evaluate_accuracy(x_train, y_train, "logistic")
# print(logistic_model_accuracy)
#
# ## Decide which variables to convert into quartiles based on whether converting
# ## each individually improves the accuracy of using all of the variables without transformation
#
# df_train_categorical = df_train.copy()
#
# include = []
# for i in range(len(columns)):
#     if i!= TARGET_COL:
#         if df_train[columns[i]].min() != np.percentile(df_train[columns[i]],25) \
#         != np.percentile(df_train[columns[i]],50) != np.percentile(df_train[columns[i]],75) \
#         != df_train[columns[i]].max():
#             process.categorize_quartile(df_train_categorical,i)
#             x_train_categorical = df_train_categorical.copy()
#             y_train_categorical = df_train_categorical[columns[TARGET_COL]]
#             del x_train_categorical[columns[TARGET_COL]]
#
#             model_accuracy = evaluate.evaluate_accuracy(x_train_categorical, y_train_categorical, "logistic")
#             if model_accuracy > logistic_model_accuracy:
#                 include.append(i)
#             df_train_categorical = df_train.copy()
#
# #### convert the indexes in the include list above into quartiles
# df_train_categorical = df_train.copy()
# for i in include:
#     process.categorize_quartile(df_train_categorical,i)
#
# x_train_categorical = df_train_categorical.copy()
# y_train_categorical = x_train_categorical[columns[TARGET_COL]]
# del x_train_categorical[columns[TARGET_COL]]
#
# ###### evaluate models on training data by splitting training dataset
# categorical_logistic_model_accuracy = evaluate.evaluate_accuracy(x_train, y_train, "logistic")
#
#
# ### Train on entire training set and return predictions for testing set using the categorical model:
# predictions = classifier.logistic(x_train, y_train, x_new)
# np.round(predictions)
# output = pd.DataFrame(predictions)
#
# #### Save predictions to file
# output.to_csv("predictions.csv", header = ['Delinquent'])



















##### train on entire training set and return predictions for testing set
# predictions = classifier.logistic(x_train, y_train, x_new)

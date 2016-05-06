import read
import summarize
import process
import evaluate
import pandas as pd
import classifier
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import cross_validation
import scipy


TARGET_COL = 0

#### read data
df_train = read.read_file('data/cs-training.csv', index_col = 0)
df_test = read.read_file('data/cs-test.csv', index_col = 0)
columns  = df_train.columns

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#### seperate x and y variables from training and testing sets
X = df_train.copy()
y = X[columns[TARGET_COL]]
del X[columns[TARGET_COL]]
del df_test[columns[TARGET_COL]]

#### create histograms for each variable in input data
for col in columns:
    summarize.histogram(df_train,col)

#### if skew above 2 create logged versions
# df_skew = df_train.copy()
# for col in df_skew.columns:
#     skew = scipy.stats.skew(df_skew[col])
#     if  abs(skew) > 2:
#         print(col, "is being scaled for graphs")
#         process.transform(df_skew, col)
#     for col in df_skew.columns:
#         summarize.histogram(df_skew,col)

### summary tables
for col in columns:
    summarize.summary_table(df_train, col)

#### fill missing values
mean_list = list(set(columns) - set(['MonthlyIncome', 'NumberOfDependents']))
process.fill_missing(df_train = X, df_test = df_test, mean = mean_list, median = 'MonthlyIncome', mode = ['NumberOfDependents'])

##### if skew above 2 add a log feature
# for col in X.columns:
#     process.transform(X,col)
#     process.transform(df_test,col)


##### Scale Data
# for col in X.columns:
#     scaler = preprocessing.StandardScaler().fit(X[col])
#     X[col] = scaler.transform(X[col])
#     df_test[col] = scaler.transform(df_test[col])

# #### Model
classifier.try_models(X,y, ['LR', 'RF', 'SGD', 'DT', 'KNN', 'GB', 'AB'])























##### train on entire training set and return predictions for testing set
# predictions = classifier.logistic(x_train, y_train, x_new)

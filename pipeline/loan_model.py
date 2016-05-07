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


#### seperate x and y variables from training and testing sets
X = df_train.copy()
y = X[columns[TARGET_COL]]
del X[columns[TARGET_COL]]
del df_test[columns[TARGET_COL]]

#### create histograms for each variable in input data
for col in columns:
    summarize.histogram(df_train,col)

### summary tables
for col in columns:
    summarize.summary_table(df_train, col)

#### fill missing values
mean_list = list(set(columns) - set(['MonthlyIncome', 'NumberOfDependents']))
process.fill_missing(df_train = X, df_test = df_test, mean = mean_list, median = 'MonthlyIncome', mode = ['NumberOfDependents'])

#### if skew above 2 add a log feature
for col in X.columns:
    process.transform(X,col)
    process.transform(df_test,col)


# #### Model
classifier.try_models(X,y, ['LR'])
# classifier.try_models(X,y, ['LR', 'RF', 'SGD', 'DT', 'GB', 'AB'])

#### scale data for KNN model
X_scale = X.copy()
preprocessing.robust_scale(X_scale)
classifier.try_models(X_scale, y ,['KNN'])
























##### train on entire training set and return predictions for testing set
# predictions = classifier.logistic(x_train, y_train, x_new)

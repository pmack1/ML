import read
import summarize
import process
import classifier
import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show

TARGET_COL = 0

df_train = read.read_csv('cs-training.csv', index_col = 0)
df_test = read.read_csv('cs-training.csv', index_col = 0)
columns = df_train.columns

# x_train = df_train.copy()
# y_train = x_train[columns[TARGET_COL]]
# del x_train[columns[TARGET_COL]]

#### create histograms for each variable in input data
for col in columns:
    df_train[col].hist()
    plt.title(col)
    plt.savefig(col)
    plt.close()

### summary tables

for col in columns:
    summary = summarize.summary_table(df_train, col)
    summary_title = col + " " + "summary"
    summary.to_csv(summary_title)

#### fill missing values
for i in range(len(df_train.columns)):
    if i != TARGET_COL:
        process.fill_mean(df_train, i)
        process.fill_mean(df_test, i)

#### predict 
predictions = classifier.logistic(df_train,df_test, TARGET_COL)


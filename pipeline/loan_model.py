import read
import summarize
import process
import classifier
import evaluate
import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show

TARGET_COL = 0

df_train = read.read_csv('cs-training.csv', index_col = 0)
df_test = read.read_csv('cs-training.csv', index_col = 0)
columns = df_train.columns


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


### seperate x and y variables from training and testing sets
x_train = df_train.copy()
y_train = x_train[columns[TARGET_COL]]
del x_train[columns[TARGET_COL]]
x_new = df_test.copy()
del x_new[columns[TARGET_COL]]



###### evaluate models on training data by splitting training dataset
logistic_model_accuracy = evaluate.evaluate_accuracy(x_train, y_train, "logistic")

# ##### try alternative logistic model with some features replaced to be categorical
# #### categorize DebtRatio and RevolvingUtilization as they had some of the most skewed distributions

# df_categorical_model1 = df_train.copy()
# process.categorize_quartile(df_categorical_model,1)
# process.categorize_quartile(df_categorical_model,4)
# del df_categorical_model[columns[1]]
# del df_categorical_model[columns[4]]

# x_train_categorical1 = df_categorical_model1.copy()
# y_train_categorical1 = df_categorical_model1[columns[TARGET_COL]]
# del x_train_categorical1[columns[TARGET_COL]]

# model2_accuracy = evaluate.evaluate_accuracy(x_train_categorical, y_train_categorical, "logistic")

###### add a new model with MonthlyIncome converted into a quartile variable

skewness = []
for i in range(len(columns)):
    if i != TARGET_COL:
        if df_train[columns[i]].min() != np.percentile(df_train[columns[i]],25) \
        != np.percentile(df_train[columns[i]],50) != np.percentile(df_train[columns[i]],75) \
        != df_train[columns[i]].max():

            skewness.append((df_train[columns[i]].skew(),i))

skewness.sort(reverse = True)


df_train_categorical = df_train.copy()



best_categorical = 0
for skew,index in skewness:
    process.categorize_quartile(df_train_categorical,index)
    x_train_categorical = df_train_categorical.copy()
    y_train_categorical = df_train_categorical[columns[TARGET_COL]]
    del x_train_categorical[columns[TARGET_COL]]

    # x_train_categorical = df_train.ix[:,df_train.columns != df_train.columns[TARGET_COL]]
    # y_train_categorical = df_train.ix[:,df_train.columns == df_train.columns[TARGET_COL]]
    model_accuracy = evaluate.evaluate_accuracy(x_train_categorical, y_train_categorical, "logistic")
    if model_accuracy > best_categorical:
        last_index = index
        best_categorical = model_accuracy

















##### train on entire training set and return predictions for testing set
# predictions = classifier.logistic(x_train, y_train, x_new)


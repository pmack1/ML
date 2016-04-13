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
df_test = read.read_csv('cs-test.csv', index_col = 0)
columns = df_train.columns


#### create histograms for each variable in input data 
for col in columns:


    df_train[col].hist()
    plt.title(col)
    plt.savefig(col)
    plt.close()

##### create histograms for each variable in input data where Target Variable  = 1
df_pos = df_train[df_train[columns[TARGET_COL]] == 1]
df_neg = df_train[df_train[columns[TARGET_COL]] == 0]
for col in columns:

    df_pos[col].hist()
    plt.title(col +  " " + "For Delinquent Loans")
    plt.savefig(col  +  "Delinquent")
    plt.close()

### summary tables

for col in columns:
    summary = summarize.summary_table(df_train, col)
    summary_title = col + " " + "summary"
    summary.to_csv(summary_title)

##### summary tables where Target Variable = 1
for col in columns:
    summary = summarize.summary_table(df_pos, col)
    summary_title = col + " " + "Summary For Delinquent" 
    summary.to_csv(summary_title)
### summary tables where Target Variable = 0
for col in columns:
    summary = summarize.summary_table(df_neg, col)
    summary_title = col + " " + "Summary For Non Delinquent" 
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

## Decide which variables to convert into quartiles based on whether converting 
## each individually improves the accuracy of using all of the variables without transformation

df_train_categorical = df_train.copy()

include = []
for i in range(len(columns)):
    if i!= TARGET_COL:
        if df_train[columns[i]].min() != np.percentile(df_train[columns[i]],25) \
        != np.percentile(df_train[columns[i]],50) != np.percentile(df_train[columns[i]],75) \
        != df_train[columns[i]].max():
            process.categorize_quartile(df_train_categorical,i)
            x_train_categorical = df_train_categorical.copy()
            y_train_categorical = df_train_categorical[columns[TARGET_COL]]
            del x_train_categorical[columns[TARGET_COL]]

            model_accuracy = evaluate.evaluate_accuracy(x_train_categorical, y_train_categorical, "logistic")
            if model_accuracy > logistic_model_accuracy:
                include.append(i)
            df_train_categorical = df_train.copy()

#### convert the indexes in the include list above into quartiles
df_train_categorical = df_train.copy()
for i in include:
    process.categorize_quartile(df_train_categorical,i)

x_train_categorical = df_train_categorical.copy()
y_train_categorical = x_train_categorical[columns[TARGET_COL]]
del x_train_categorical[columns[TARGET_COL]]

###### evaluate models on training data by splitting training dataset
categorical_logistic_model_accuracy = evaluate.evaluate_accuracy(x_train, y_train, "logistic")


### Train on entire training set and return predictions for testing set using the categorical model:
predictions = classifier.logistic(x_train, y_train, x_new)
np.round(predictions)
output = pd.DataFrame(predictions)

#### Save predictions to file
output.to_csv("predictions.csv", header = ['Delinquent'])



















##### train on entire training set and return predictions for testing set
# predictions = classifier.logistic(x_train, y_train, x_new)


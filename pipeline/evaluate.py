import pandas as pd
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import classifier
import time
from sklearn.externals.six import StringIO
import re

def print_table(model_name, y_test, y_pred, time_start):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test,y_pred)
    current_time = time.clock()
    runtime = current_time - time_start

    name = str(model_name)
    pattern = "^[a-zA-Z]*"
    match = re.search(pattern, name)
    filename = match.group(0)
    with open(filename + "results.txt", 'w') as f:
        f.write("---------------------------------------------")
        f.write(name + "\n")
        f.write("Accuracy: " + str(round(accuracy,2)) + "\n")
        f.write("Precicision: " + str(round(precision,2)) + "\n")
        f.write("Recall: " + str(round(recall,2)) + "\n")
        f.write("AUC: " + str(round(auc, 2)) + "\n")
        f.write("F1 Score: " + str(round(f1score,2)) + "\n")
        f.write("Run Time: " + str(round(runtime,2)) + "\n")
        f.write("--------------------------------------------")

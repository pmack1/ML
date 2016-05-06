import pandas as pd
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import classifier
import time

def print_table(model_name, y_test, y_pred, time_start):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test,y_pred)
    recall = metrics.recall_score(y_test,y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test,y_pred)
    current_time = time.clock()
    runtime = current_time - time_start
    print(model_name)
    print("accuracy is", accuracy)
    print("precision is", precision)
    print("recall is ", recall)
    print("auc is", auc)
    print("f1score is",  f1score)
    print("run time was", runtime)
    print("\n")

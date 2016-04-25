import pandas as pd
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import classifier

def evaluate_accuracy(x, y, model_name):
    if model_name == 'logistic':
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)
        y_pred = classifier.logistic(x_train, y_train, x_test)
        return metrics.accuracy_score(y_test, y_pred)


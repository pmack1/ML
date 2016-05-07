''' Process Data '''
import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show


def transform(df, col, transform = 'log'):
    '''Transforms a column either log, sqrt, or inverse. If the transformation
    cannot be done (e.g. log of 0) the value is transformed into a null value. Adds the new column as a feature'''
    if transform == 'log':
        rv = np.log(np.ma.array(df[col]))
    elif transform == 'sqrt':
        rv = np.sqrt(np.ma.array(df[col]))
    elif transform == 'inverse':
        rv = np.divide(1, np.ma.array(df[col]))
    else:
        print("transform not recognized")
    df[col +  transform] = rv

def fill_missing(df_train, df_test , mean = [], median = [], mode = []):
    ''' Fills in missing values by mean, median, or mode. Does not fill in columns in ignore
    df_test  is  imputed with the  df_train statistics'''
    for col in df_train.columns:
        if col in mean:
            mean_val = df_train[col].mean()
            df_train[col].fillna(mean_val, inplace = True)
            df_test[col].fillna(mean_val, inplace = True)
        if col in mode:
            '''Uses first mode if multiple'''
            mode_val = df_train[col].mode()[0]
            df_train[col].fillna(mode_val, inplace  = True)
            df_test[col].fillna(mean_val, inplace = True)
        if col in median:
            median_val = df_train[col].median()
            df_train[col].fillna(median_val, inplace = True)
            df_test[col].fillna(mean_val, inplace = True)


def quartile_bin(df, col, q_num = 4):
    '''Bins Continuous Data into Quartiles '''
    labels = [col + str(i) for i in range(1,q_num +1)]
    df[col] = pd.qcut(df[col], q = q_num, labels = labels)


def fill_gender(df):
    '''Fills in missing genders using API'''
    missing = df.loc[df['Gender'].isnull(),'First_name']
    infer = []
    for name in missing:
        infer.append(infer_gender(name))
    df.loc[df['Gender'].isnull(),'Gender'] = infer

def infer_gender(name):
    '''Infers Gender using API'''
    url = "https://api.genderize.io/?name="
    request = url + name
    r = requests.get(request)
    gender = r.json()['gender']
    return gender.title()

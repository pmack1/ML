import pandas as pd
import pylab as pl
import numpy as np

def fill_mean(df, col):
    df.loc[df[df.columns[col]].isnull(), df.columns[col]] = df[df.columns[col]].mean()

def categorize_quartile(df, col):
    '''Splits data into 1,2,3,4 quartile'''
    columns = df.columns
    top = df[columns[col]].max()
    bottom = df[columns[col]].min()
    first = np.percentile(df[columns[col]],25)
    second = np.percentile(df[columns[col]],50)
    third = np.percentile(df[columns[col]],75)


   
    

    bins = [bottom] + [first] + [second] + [third] + [top]
    new_col_name = df.columns[col] + "_" + "category"
    df[new_col_name] = pd.cut(x = df[columns[col]], bins = bins, labels = ['first', 'second', 'third', 'fourth'], right = False , include_lowest = True)



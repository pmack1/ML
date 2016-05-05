''' Process Data '''
import pandas as pd
import pylab as pl
import numpy as np

# a = np.array([1,2,3,0,4,-1,-2])
# b = np.log(np.ma.array(a)

def transform(df, col, transform = 'log'):
    '''Transforms a column either log, sqrt, or inverse. If the transformation
    cannot be done (e.g. log of 0) the value is transformed into a null value'''
    if transform == 'log':
        rv = np.log(np.ma.array(df[col]))
    elif transform == 'sqrt':
        rv = np.sqrt(np.ma.array(df[col]))
    elif transform == 'inverse':
        rv = np.divide(1, np.ma.array(df[col]))
    else:
        print("transform not recognized")
    df[col] = rv

def fill_missing(df, col, method = 'mean'):
    if method == 'mean':
        df[col].fillna(df[col].mean(), inplace = True)
    if method == 'mode':
        '''Uses first mode if multiple'''
        df[col].fillna(df[col].mode()[0], inplace  = True)
    if method == 'median':
        df[col].fillna(df[col].median(), inplace = True)

def quartile_bin(df, col, q_num = 4, labels = ['first', 'second', 'third', 'fourth']):
    '''Bins Continuous Data into Quartiles '''
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

def categorize_quartile(df, col):
    '''Splits data into 1,2,3,4 quartile - if quartile are equal to each other does not bin'''
    columns = df.columns
    top = df[columns[col]].max()
    bottom = df[columns[col]].min()
    first = np.percentile(df[columns[col]],25)
    second = np.percentile(df[columns[col]],50)
    third = np.percentile(df[columns[col]],75)






    bins = [bottom] + [first] + [second] + [third] + [top]
    # new_col_name = df.columns[col] + "_" + "category"

    df[columns[col]] = pd.cut(x = df[columns[col]], bins = bins, labels = [1, 2, 3, 4], right = True , include_lowest = True)

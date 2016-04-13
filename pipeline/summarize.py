import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show

def summary_table(df,col): 
    summary = {'mean': np.round(df[col].mean(),2),
                'median':np.round(df[col].median(),2),
                'standard_deviation': np.round(df[col].std(),2),
                'min': np.round(df[col].min(),2),
                'max': np.round(df[col].max(),2),
                'null_count': len(df[col]) - df[col].count()}
    summary_df = pd.DataFrame(summary, index = ['mean', 'median', 'standard_deviation', 'min', 'max', 'null_count'])
    return  summary_df.iloc[0]

def quartile_histogram(df):
    columns = df.columns
    for col in columns:
        top = df[col].max()
        bottom = df[col].min()
        first = np.percentile(df[col],25)
        second = np.percentile(df[col],50)
        third = np.percentile(df[col],75)
        bins = [bottom] + [first] + [second] + [third] + [top]

        plt.hist(df[col], bins = bins)
        plt.title(col)
        plt.savefig(col)
        plt.close()

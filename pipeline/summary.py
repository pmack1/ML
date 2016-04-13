import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, axes, pie, title, show

def summary_table(df,col):
    mode_list = []
    for mode in df[col].mode():
        mode_list.append(mode)
    mode = {'mode': mode_list}
   
    summary = {'mean': df[col].mean(),
                'median':df[col].median(),
                'standard_deviation': df[col].std(),
                'null_count': len(df[col]) - df[col].count()}
    summary_df = pd.DataFrame(summary, index = ['mean', 'median', 'standard_deviation', 'null_count'])
    mode_df = pd.DataFrame(mode, index = ['mode'])
    return  summary_df.iloc[0], mode_df

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

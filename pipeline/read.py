import pandas as pd
import pylab as pl
import numpy as np

def read_csv(file_name, index_col =  None):
    df = pd.read_csv(file_name, index_col = index_col)
    return df
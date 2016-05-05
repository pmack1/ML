''' Module for reading in a  dataset '''

import pandas as pd
import pylab as pl
import numpy as np
import re

##### Regular Expression Pattern to extract  file extension
#### Found here: http://stackoverflow.com/questions/29436088/how-to-use-regex-to-get-file-extension
pattern = "(\\.[^.]+)$"



def read_file(file_name, index_col =  None):
    match = re.search(pattern, file_name)
    ext = match.group(0)

    if ext == ".csv":
        df = pd.read_csv(file_name, index_col = index_col)
        return df
    if ext == '.dat':
        df = pd.read_stata(file_name, index_col = index_col)
        return df
    else:
        print("File Extension not recognized")


def read_space_seperated(file_name, index_col = None):
    df= pd.read_table(file_name, sep = ' ', index_col = index_col)
    return df

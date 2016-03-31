### File for Loading in and Creating Summary Statistics of Fake Student Records

import pandas as pd
import numpy as np

df = pd.read_csv("mock_student_data.csv")
columns = df.columns

summary = []
for column in columns:
    if df[column].dtype == "float64":
        summary.append([column,df[column].mean(), df[column].median(), \
        df[column].mode()[0], df[column].std(), df[column].isnull().sum()
        ])


f = open('summary.txt', 'w')
f.write("variable, mean , median, mode, std , null \n")
for row in summary:
    value = ""
    for i in range(len(row)):
        value = value + str(row[i]) + ", "
    value = value + "\n"

    f.write(str(value))
f.close()


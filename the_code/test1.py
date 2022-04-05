import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import statsmodels.api as smf
import sklearn as sk 
import scipy as sp
import matplotlib as plt


# Dataframe that was the final merge from Project two. (Concat t2, t3), (Merged S3a, S4a) (Merged Concat, Merged S3a, S4A)
df = pd.read_csv('source\merge\data-merge2.csv', low_memory=False)


# Reassigning Varables to Status L:1, M:0, S:0
df1 = df['NEW_STATUS'] = df.STATUS.map({'L':1, 'M':0, 'S':0})
df1.describe()
print(df1)
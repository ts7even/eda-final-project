import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn as sk 
from sklearn.linear_model import LinearRegression
import scipy as sp
import matplotlib as plt

df = pd.read_csv('source/data_cleaning/cleaned_data.csv')



def opportunites():
    print('I was pleased with the opportunities for professional DEVELOPMENT\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('F0119')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Disagree', 'Agree'], 
    title='Linear Regression on the Free Traininng Variable against Leaver'))
    print()

opportunites()

# 
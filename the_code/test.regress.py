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


def linearRegressionModel():
    print('I was pleased with the opportunities for professional DEVELOPMENT.\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0186')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0186)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['No', 'Yes'],  
    title='Linear Regression on the Opportunities for professional development Variable against Leaver'))
    print()

def LogisticRegressionModel():
    print('REQUIRED professional development activities at the school usually closely matched my professional development goals..\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0186')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.logit('NEW_STATUS ~ C(T0186)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['No', 'YEs'],  
    title='Linear Regression on the prof development activities variable against Leaver'))
    print()

linearRegressionModel()
LogisticRegressionModel()
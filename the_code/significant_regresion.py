import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn as sk 
from sklearn.linear_model import LinearRegression
import scipy as sp
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
matplotlib.use('Qt5Agg')

df = pd.read_csv('source/data_cleaning/cleaned_data.csv')

# Count works fine with multi regression except for F0119, 

def regressMulti7():
    model = smf.logit('NEW_STATUS ~ C(S0287) + TOTEXPER_x + T0277 + T0278 + AGE_T_x', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', '4 percent or less free lunch (S0287)', 'Total Years of Experiance (TOTEXPER_x)', 'Total Hours of Preperation (T0277)',
    'How many students where tardy (T0278)', 'Younger than 40 (AGE_T_x)'], 
    title=' Multiple Logistic Regression With High Correlation Test'))
    print()

def regressMulti8():
    model = smf.logit('NEW_STATUS ~ C(T0080) + (EARNALL) + C(T0329) + C(T0330)', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Masters Degree (T0080)', 'Teachers that make less than 39,999k (EARNALL)', 'Student Alcohol Abuse Problem (T0329)',
    'Student Drug Abuse Problem (T0330)'], 
    title=' Multiple Logistic Regression With High Correlation Test'))
    print()




# regressMulti7()
regressMulti8()
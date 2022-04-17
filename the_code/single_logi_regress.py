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

df = pd.read_csv('source/data_cleaning/cleaned_data.csv')
df1 = pd.read_csv('source/data_cleaning/cleaned_data_male.csv')
df2 = pd.read_csv('source/data_cleaning/cleaned_data_female.csv')



# Data for Graph
# Only do graphs for variables that correlate to our project. 

matplotlib.use('Qt5Agg')
sns.set_theme(style='ticks', color_codes=True)



# On statsmodels, It goes by numerical order of smallest to largest. 
def opportunites():
    indep = ['AGE', 'S0287', 'T0080', 'SALARY', 'T0329', 'T0333', 'T0159', 'T0165','EXPER', 'T0356'] 
    for i in indep:
        model = smf.logit(f'LEAVER ~ {i}', data = df ).fit()
        print(model.summary(yname="Status Leaver", xname=['Intercept', i ],  
        title='Single Logistic Regression'))
        print()
opportunites()


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
# Only do graphs for variables that correlate to our project. 

# Order of Variables that we are doing 
# F0119
# F0120
# T0186 ------
# S1628 ------
# T0178 ------
# T0165
# F0120 
# F0115
## F0119 


# On statsmodels, It goes by numerical order of smallest to largest. 
def opportunites():
    print('I was pleased with the opportunities for professional DEVELOPMENT.\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('F0119')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(F0119)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Disagree', 'Agree'],  
    title='Linear Regression on the Opportunities for professional development Variable against Leaver'))
    print()



def IncreseSalaryProfDevelopment(): # Good to go
    print('Increase in salary or other pay increases as a result of participating in professional development activities?\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0186')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0186)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Yes', 'No'],  # Yes is 0, No is 1 
    title='Linear Regression on the pay rewards increase of completing professional development variable against Leaver'))
    print()



def regressFreeTraining():
    print('If free training available in this district? regardless of funding source, to prepare staff members to teach in fields with current or anticipated shortages?\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('S1628')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(S1628)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Yes Free Training is Avaliable', 'No Free Training Avaliable'], # 0 is 
    title='Linear Regression on the Free Traininng Variable against Leaver'))
    print()




def age(): 
    print('What are the ages of the teachers? \n ')
    print("Preliminary Stats")
    pre_stats = df.groupby('AGE_T_x')["NEW_STATUS"].describe()
    print(pre_stats)
    print('\n \n')
    model = smf.ols('NEW_STATUS ~ C(AGE_T_x)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Younger than 40', '40 and older'], # 1 is younger than 40 #0 is older than 40
    title='Linear Regression on the Age Variable against Leaver'))
    print()



def freeLunch(): 
    print('Around the first of October, how many applicants at this school were APPROVED for free or reduced-price lunches?\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('S0287')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(S0287)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Less than 50 percent', '50 percent or more'], # 1 is less than 50% #0 is 50 percent or more
    title='Linear Regression on the Free Lunch Variable against Leaver'))
    print()



opportunites()
IncreseSalaryProfDevelopment()
regressFreeTraining()
age()
freeLunch()




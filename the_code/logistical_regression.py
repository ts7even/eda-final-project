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


def regressAttack():
    print('Have you been attacked before as a teacher? \n')
    print("Preliminary Stats")
    pre_stats = df.groupby('ATTACK')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.logit('NEW_STATUS ~ C(ATTACK)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=["Never Attacked", "Attacked but not in the last 12 Months", "Attacked in the past 12 months"],
    title='Linear Regression on the Attack Variable against Leaver'))
    print()
    

def regressFreeTraining():
    print('If free training available in this district? regardless of funding source, to prepare staff members to teach in fields with current or anticipated shortages?\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('S1628')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.logit('NEW_STATUS ~ C(S1628)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Yes Free Training is Avaliable', 'No Free Training Avaliable'],
    title='Linear Regression on the Free Traininng Variable against Leaver'))
    print()

def jobSecurity():
    print('I worry about the security of my job because of the performance of my students on state or local tests.\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0313')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.logit('NEW_STATUS ~ C(T0313)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Strongly Agree', 'Somewhat Agree', 'Somewhat disagree', 'Stongly Agree'],
    title='Linear Regression on the Job Security Variable against Leaver'))
    print()

def age():
    print('What are the ages of the teachers? \n')
    print("Preliminary Stats")
    pre_stats = df.groupby('AGE_T')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.logit('NEW_STATUS ~ C(AGE_T)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Younger than 30', '30 - 39', '40 - 49', '50 or older'],
    title='Linear Regression on the Age Variable against Leaver'))
    print()

# def totalExperiance(): # This is a catagorical regresing on a continuous variable
#     print("Linear Regression on the Total Expriance Variable against Leaver  ")
#     model = smf.ols('NEW_STATUS ~ TOTEXPER', data = df).fit()
#     print(model.summary())
#     print()

def specialNeedsSupport():
    print('I am given the support I need to teach students with special needs.\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0314')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.logit('NEW_STATUS ~ C(T0314)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Strongly Agree', 'Somewhat Agree', 'Somewhat disagree', 'Stongly Agree'],
    title='Linear Regression on the Support Special Needs Variable against Leaver'))
    print()

def schoolLevel():
    print('What school level does the teacher teach?.\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('SCHLEVEL_x')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.logit('NEW_STATUS ~ C(SCHLEVEL_x)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Elementary', 'Secondary', 'Combined'],
    title='Linear Regression on the School Level Variable against Leaver'))
    print()

def freeLunch():
    print('Around the first of October, how many applicants at this school were APPROVED for free or reduced-price lunches?\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('S0287')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.logit('NEW_STATUS ~ C(S0287)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Less than 1 percent', '1 - 4 percent', '5 - 19 percent', '20 percent or more'],
    title='Linear Regression on the Free Lunch Variable against Leaver'))
    print()

def doBest():
    print("I sometimes feel it is a waste of time to try to do my best as a teacher. \n")
    print("Preliminary Stats")
    pre_stats = df.groupby('T0318')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.logit('NEW_STATUS ~ C(T0318)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Strongly Agree', 'Somewhat Agree', 'Somewhat disagree', 'Stongly Agree'],
    title='Linear Regression on the Do your Best Variable against Leaver'))
    print()


# def StudentswithDisablilies(): # # This is a catagorical regresing on a continuous variable
#     print("Linear Regression on the total students with disabilites variable against Leaver  ")
#     model = smf.logit('NEW_STATUS ~ T0244', data = df).fit()
#     print(model.summary(title='How many students have disabilities or are special education students at your school?'))
#     print()




regressAttack()
regressFreeTraining()
jobSecurity()
age()
# totalExperiance()
specialNeedsSupport()
schoolLevel()
freeLunch()
doBest()
# StudentswithDisablilies()
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
df1 = pd.read_csv('source/data_cleaning/cleaned_data_male.csv')
df2 = pd.read_csv('source/data_cleaning/cleaned_data_female.csv')


# To See if Data Carried over and observations are the same.
test_observation1 = df['AGE_T_x'].describe()
test_observation2 = df['F0119'].describe()
test_observation3 = df['T0330'].count()

print(test_observation1)
print(test_observation2)
print(test_observation3)





def regressMulti7():
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x) + C(S0287) + C(T0178) + C(T0080) + C(EARNALL) + C(T0329) + C(T0330) + C(T0332) + C(T0333) +C(T0336) +C(T0155) +C(T0157) +C(T0182) +C(T0184) +C(T0159) +C(T0165) +C(T0174) + TOTEXPER_x', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Pleased Prof Dev (F0119)', 'Salary increase becasue of prof dev (T0186)', 'Free Training Avaliable (S1628)',
    'Younger than 40 (AGE_T_x)', '4 percent or less free lunch (S0287)','Useful prof dev (T0178)', 'Masters Degree (T0080)',
    'Teachers that make less than 39,999k (EARNALL)', 'Student Alcohol Abuse Problem (T0329)', 'Student Drug Abuse Problem (T0330)', 'Disrespect Towards Teachers (T0332)', 'Problem with dropouts (T0333)',
    'Problem With Poverty (T0336)', 'Mentorship or Coaching (T0155)', 'Prof Dev Workshops (T0157)', 'Tuition reimbersement (T0182)', 'Reimbersement daily expenses (T0184)',
    'Prof Dev Main Assign (T0159)', 'Prof Dev Methods of Teaching (T0165)', 'Prof Dev Student Discipline (T0174)', 'Total Years of Experiance (TOTEXPER_x)'], 
    title=' Multiple Logistic Regression (Test to see observations)'))
    print()


def regressMulti8():
    model = smf.logit('NEW_STATUS ~ F0119 + T0186 + S1628 + AGE_T_x + S0287 + T0178 + T0080 + EARNALL + T0329 + T0330 + T0332 + T0333 + T0336 + T0155 + T0157 + T0182 + T0184 + T0159 + T0165 + T0174 + TOTEXPER_x', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Pleased Prof Dev (F0119)', 'Salary increase becasue of prof dev (T0186)', 'Free Training Avaliable (S1628)',
    'Younger than 40 (AGE_T_x)', '4 percent or less free lunch (S0287)','Useful prof dev (T0178)', 'Masters Degree (T0080)',
    'Teachers that make less than 39,999k (EARNALL)', 'Student Alcohol Abuse Problem (T0329)', 'Student Drug Abuse Problem (T0330)', 'Disrespect Towards Teachers (T0332)', 'Problem with dropouts (T0333)',
    'Problem With Poverty (T0336)', 'Mentorship or Coaching (T0155)', 'Prof Dev Workshops (T0157)', 'Tuition reimbersement (T0182)', 'Reimbersement daily expenses (T0184)',
    'Prof Dev Main Assign (T0159)', 'Prof Dev Methods of Teaching (T0165)', 'Prof Dev Student Discipline (T0174)', 'Total Years of Experiance (TOTEXPER_x)'], 
    title=' Multiple Logistic Regression (Test to see observations)'))
    print()


regressMulti7()
regressMulti8()
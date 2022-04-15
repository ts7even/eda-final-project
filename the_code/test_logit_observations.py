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
test_observation1 = df['NEW_STATUS'].count()
test_observation2 = df['F0119'].count()
test_observation3 = df['T0186'].count()
test_observation4 = df['S1628'].count()
test_observation5 = df['AGE_T_x'].count()
test_observation6 = df['S0287'].count()
test_observation7 = df['T0178'].count()
test_observation8 = df['T0080'].count()
test_observation9 = df['EARNALL'].count()
test_observation10 = df['T0329'].count()
test_observation11 = df['T0330'].count()
test_observation12= df['T0332'].count()
test_observation13 = df['T0333'].count()
test_observation14 = df['T0336'].count()
test_observation15 = df['T0155'].count()
test_observation16 = df['T0157'].count()
test_observation17 = df['T0182'].count()
test_observation18 = df['T0184'].count()
test_observation19 = df['T0159'].count()
test_observation20 = df['T0165'].count()
test_observation21 = df['T0174'].count()
test_observation22 = df['TOTEXPER_x'].count()
print(f'Observations for NEW_STATUS:  {test_observation1}')
print(f'Observations for F0119: {test_observation2}')
print(f'Observations for T0186: {test_observation3}')
print(f'Observations for S1628: {test_observation4}')
print(f'Observations for AGE_T_x: {test_observation5}')
print(f'Observations for S0287: {test_observation6}')
print(f'Observations for T0178: {test_observation7}')
print(f'Observations for T0080: {test_observation8}')
print(f'Observations for EARNALL: {test_observation9}')
print(f'Observations for T0329: {test_observation10}')
print(f'Observations for T0330: {test_observation11}')
print(f'Observations for T0332: {test_observation12}')
print(f'Observations for T0333: {test_observation13}')
print(f'Observations for T0336: {test_observation14}')
print(f'Observations for T0155: {test_observation15}')
print(f'Observations for T0157: {test_observation16}')
print(f'Observations for T0182: {test_observation17}')
print(f'Observations for T0184: {test_observation18}')
print(f'Observations for T0159: {test_observation19}')
print(f'Observations for T0165: {test_observation20}')
print(f'Observations for T0174: {test_observation21}')
print(f'Observations for TOTEXPER_x: {test_observation22}')





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

def regressMulti9():
    model = smf.logit('NEW_STATUS ~ AGE_T_x + S0287 + T0080 + EARNALL + T0329 + T0333 + T0159 + T0165 + TOTEXPER_x ', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept',
    'Younger than 40 (AGE_T_x)', '4 percent or less free lunch (S0287)','Masters Degree (T0080)',
    'Teachers that make less than 39,999k (EARNALL)', 'Student Alcohol Abuse Problem (T0329)', 'Problem with dropouts (T0333)',
    'Prof Dev Main Assign (T0159)', 'Prof Dev Methods of Teaching (T0165)', 'Total Years of Experiance (TOTEXPER_x)'], 
    title=' Multiple Logistic Regression (Test to see observations)'))
    print()


# regressMulti7()
# regressMulti8()
regressMulti9()
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



def regressMulti2():
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186)', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Overall)', 'No salary increase becasue of prof dev (Overall)'], 
    title='Multiple regression of F0119 and T0186 (Overall)'))
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186)', data = df1).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Male)', 'No salary increase becasue of prof dev (Male)'], 
    title='Multiple regression of F0119 and T0186 (Male)'))
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186)', data = df2).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Female)', 'No salary increase becasue of prof dev (Female)'], 
    title='Multiple regression of F0119 and T0186 (Female)'))
    print()

  


def regressMulti3():
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628)', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Overall)', 'No salary increase becasue of prof dev (Overall)', 'No Free Training Avaliable (Overall)'], 
    title=' Multiple Linear Regression on F0119, T0186, and S1628 (Overall)'))
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628)', data = df1).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Male)', 'No salary increase becasue of prof dev (Male)', 'No Free Training Avaliable (Male)'], 
    title=' Multiple Linear Regression on F0119, T0186, and S1628 (Male)'))
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628)', data = df2).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Female)', 'No salary increase becasue of prof dev (Female)', 'No Free Training Avaliable (Female)'], 
    title=' Multiple Linear Regression on F0119, T0186, and S1628 (Female)'))
    print()



def regressMulti4():
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x)', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Overall)', 'No salary increase becasue of prof dev (Overall)', 'No Free Training Avaliable (Overall)',
    'Younger than 40 (Overall)'], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, and AGE_T_x  (Overall)'))
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x)', data = df1).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Overall)', 'No salary increase becasue of prof dev (Male)', 'No Free Training Avaliable (Male)',
    'Younger than 40 (Male)'], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, and AGE_T_x  (Male)'))
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x)', data = df2).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Female)', 'No salary increase becasue of prof dev (Female)', 'No Free Training Avaliable (Female)',
    'Younger than 40 (Overall)'], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, and AGE_T_x  (Female)'))
    print()



def regressMulti5():
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x) + C(S0287)', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Overall)', 'No salary increase becasue of prof dev (Overall)', 'No Free Training Avaliable (Overall)',
    'Younger than 40 (Overall)', '20 percent or more free lunch (Overall)'], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, AGE_T_x, S0287 (Overall)'))
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x) + C(S0287)', data = df1).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Male)', 'No salary increase becasue of prof dev (Male)', 'No Free Training Avaliable (Male)',
    'Younger than 40 (Male)', '20 percent or more free lunch (Male)'], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, AGE_T_x, S0287 (Male)'))
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x) + C(S0287)', data = df2).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Female)', 'No salary increase becasue of prof dev (Female)', 'No Free Training Avaliable (Female)',
    'Younger than 40 (Female)', '20 percent or more free lunch (Female)'], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, AGE_T_x, S0287 (Female)'))
    print()
    

def regressMulti6():
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x) + C(S0287) + C(T0178)', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Overall)', 'No salary increase becasue of prof dev (Overall)', 'No Free Training Avaliable (Overall)',
    'Younger than 40 (Overall)', '20 percent or more free lunch (Overall)','Not useful prof dev (Overall)' ], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, AGE_T_x, S0287, T0178 (Overall)'))
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x) + C(S0287) + C(T0178)', data = df1).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Male)', 'No salary increase becasue of prof dev (Male)', 'No Free Training Avaliable (Male)',
    'Younger than 40 (Male)', '20 percent or more free lunch (Male)','Not useful prof dev (Male)' ], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, AGE_T_x, S0287, T0178 (Male)'))
    print()
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x) + C(S0287) + C(T0178)', data = df2).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Female)', 'No salary increase becasue of prof dev (Female)', 'No Free Training Avaliable (Female)',
    'Younger than 40 (Female)', '20 percent or more free lunch (Female)','Not useful prof dev (Female)' ], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, AGE_T_x, S0287, T0178 (Female)'))
    print()


def regressMulti7():
    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x) + C(S0287) + C(T0178) + C(T0080) + C(EARNALL) + C(T0329) + C(T0330) + C(T0332) + C(T0333) +C(T0336) +C(T0336) +C(T0155) +C(T0157) +C(T0182) +C(T0184) +C(T0159) +C(T0165) +C(T0174)', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Overall)', 'No salary increase becasue of prof dev (Overall)', 'No Free Training Avaliable (Overall)',
    'Younger than 40 (Overall)', '20 percent or more free lunch (Overall)','Not useful prof dev (Overall)', 'No Masters Degree (Overall)',
    'Teachers that make less than 39,999k (Overall)', 'Student Alcohol Abuse Problem', 'Student Drug Abuse Problem', 'Disrespect Towards Teachers', 'Problem with dropouts',
    'Problem With Poverty', 'No Mentorship or Couching', 'No Prof Dev Workshops', 'No Tuition reimbersement', 'No reimbersement daily expenses',
    'No Prof Dev Main Assign', 'No Prof Dev Methods', 'No Prof Dev Student Discipline'], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, AGE_T_x, S0287, T0178 (Overall)'))
    print()

    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x) + C(S0287) + C(T0178) + C(T0080) + C(EARNALL) + C(T0329) + C(T0330) + C(T0332) + C(T0333) +C(T0336) +C(T0336) +C(T0155) +C(T0157) +C(T0182) +C(T0184) +C(T0159) +C(T0165) +C(T0174)', data = df1).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Male)', 'No salary increase becasue of prof dev (Male)', 'No Free Training Avaliable (Male)',
    'Younger than 40 (Male)', '20 percent or more free lunch (Male)','Not useful prof dev (Female)', 'No Masters Degree (Male)',
    'Teachers that make less than 39,999k (Male)', 'Student Alcohol Abuse Problem', 'Student Drug Abuse Problem', 'Disrespect Towards Teachers', 'Problem with dropouts',
    'Problem With Poverty', 'No Mentorship or Couching', 'No Prof Dev Workshops', 'No Tuition reimbersement', 'No reimbersement daily expenses',
    'No Prof Dev Main Assign', 'No Prof Dev Methods', 'No Prof Dev Student Discipline'], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, AGE_T_x, S0287, T0178 (Male)'))
    print()

    model = smf.logit('NEW_STATUS ~ C(F0119) + C(T0186) + C(S1628) + C(AGE_T_x) + C(S0287) + C(T0178) + C(T0080) + C(EARNALL) + C(T0329) + C(T0330) + C(T0332) + C(T0333) +C(T0336) +C(T0336) +C(T0155) +C(T0157) +C(T0182) +C(T0184) +C(T0159) +C(T0165) +C(T0174)', data = df2).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Not Pleased Prof Dev (Female)', 'No salary increase becasue of prof dev (Female)', 'No Free Training Avaliable (Female)',
    'Younger than 40 (Female)', '20 percent or more free lunch (Female)','Not useful prof dev (Female)', 'No Masters Degree (Female)',
    'Teachers that make less than 39,999k (Female)','Student Alcohol Abuse Problem', 'Student Drug Abuse Problem', 'Disrespect Towards Teachers', 'Problem with dropouts',
    'Problem With Poverty', 'No Mentorship or Couching', 'No Prof Dev Workshops', 'No Tuition reimbersement', 'No reimbersement daily expenses',
    'No Prof Dev Main Assign', 'No Prof Dev Methods', 'No Prof Dev Student Discipline'], 
    title=' Multiple Linear Regression on F0119, T0186, S1628, AGE_T_x, S0287, T0178 (Female)'))
    print()


def logiGraph(): # Need to try it with a continuous variable for independent
    # Overall Data on No mentorship or couch 
    # Train Test Split Method
    # plt.scatter(df.T0157, df.NEW_STATUS, marker='+', color='red')

    # Setting Up the Training Data
    X_train, X_test, y_train, y_test = train_test_split(df[['T0157']],df.NEW_STATUS, test_size=0.1)

    model = LogisticRegression()
    model.fit(X_train , y_train)

    # Predicting Data
    prediction = model.predict(X_test) # Predicting the model 
    rank = model.score(X_test,y_test) # Shows the accuracy of the model 
    probability = model.predict_proba(X_test) # Probability of individuals in the model Left Side 0 (Not Leave) | Right Side 1  (Leave) = Which you read as a percentage
    print(rank) # Value of 1.0 should be perfect. Currently it is 0.5691
    print(probability)
    sns.regplot(x='T0157', y='NEW_STATUS', data=df, logistic=True)
    plt.title("Logistic Regression (Sigma Loc Function S -Curve) Leaver Vs. No mentorship ")
    plt.show(block=False)
    plt.savefig('profiling/no_mentorship_logi_graph.png')
    plt.pause(2)
    plt.close()

def logiGraph2(): # Contiuous Variable 

    # Graph Variable
    plt.scatter(df.TOTEXPER_x, df.NEW_STATUS, marker='+', color='red')

    # Setting Up the Training Data
    X_train, X_test, y_train, y_test = train_test_split(df[['TOTEXPER_x']],df.NEW_STATUS, test_size=0.1)

    model = LogisticRegression()
    model.fit(X_train , y_train)

    # Predicting Data
    prediction = model.predict(X_test) # Predicting the model 
    rank = model.score(X_test,y_test) # Shows the accuracy of the model 
    probability = model.predict_proba(X_test) # Probability of individuals in the model Left Side 0 (Not Leave) | Right Side 1  (Leave) = Which you read as a percentage
    print(rank) # Value of 1.0 should be perfect. Currently it is 0.5691
    # print(probability)
    sns.regplot(x='TOTEXPER_x', y='NEW_STATUS', data=df, logistic=True)
    plt.title("Logistic Regression (Sigma Loc Function S -Curve) Leaver Vs. Experiance")
    plt.show(block=False)
    plt.savefig('profiling/total_experiance_logi_graph.png')
    plt.pause(2)
    plt.close()




# regressMulti2()
# regressMulti3()
# regressMulti4()
# regressMulti5()
# regressMulti6()
# regressMulti7()
logiGraph()
logiGraph2()
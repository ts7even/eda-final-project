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
from sklearn.metrics import confusion_matrix
matplotlib.use('Qt5Agg')

df = pd.read_csv('source/data_cleaning/cleaned_data.csv')
df1 = pd.read_csv('source/data_cleaning/cleaned_data_male.csv')
df2 = pd.read_csv('source/data_cleaning/cleaned_data_female.csv')


def playground():
    data = ['AGE', 'S0287', 'T0080', 'SALARY', 'T0329', 'T0333', 'T0159', 'T0165', 'EXPER']
    for d in data:
        sns.countplot(x=d, data=df, palette='Set3', hue='LEAVER')
        plt.legend(['Stayer','Leaver'])
        plt.show()



def regressMulti5():
    model = smf.logit('LEAVER ~ S0287 + EXPER + T0277 + T0278 + AGE', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', '4 percent or less free lunch (S0287)', 'Total Years of Experiance (EXPER)', 'Total Hours of Preperation (T0277)',
    'How many students where tardy (T0278)', 'Younger than 40 (AGE)'], 
    title=' Multiple Logistic Regression With High Correlation Test'))
    print()

def regressMulti6():
    model = smf.logit('LEAVER ~ T0080 + SALARY + T0329 + T0330', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept', 'Masters Degree (T0080)', 'Teachers that make less than 39,999k (EARNALL)', 'Student Alcohol Abuse Problem (T0329)',
    'Student Drug Abuse Problem (T0330)'], 
    title=' Multiple Logistic Regression With High Correlation Test'))
    print()

def regressMulti7():
    model = smf.logit('LEAVER ~ AGE + S0287 + T0080 + SALARY + T0329 + T0333 + T0159 + T0165 + EXPER + T0278 + T0277 + T0330', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept',
    'Younger than 40 (AGE)', '4 percent or less free lunch (S0287)','Masters Degree (T0080)',
    'Teachers that make less than 39,999k (SALARY)', 'Student Alcohol Abuse Problem (T0329)', 'Problem with dropouts (T0333)',
    'Prof Dev Main Assign (T0159)', 'Prof Dev Methods of Teaching (T0165)', 'Total Years of Experiance (EXPER)', ' Total students in class (T0278)',
    'Hours spent in preperation (T0277)', 'Problem with drug abuse (T0330)'], 
    title=' Multiple Logistic Regression (Overall)'))
    print()



# This one is good to go. 
def regressMulti8():
    model = smf.logit('LEAVER ~ AGE + S0287 + T0080 + SALARY + T0329 + T0333 + T0159 + T0165 + EXPER', data = df).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept',
    'Younger than 40 (AGE)', '4 percent or less free lunch (S0287)','Masters Degree (T0080)',
    'Teachers that make less than 39,999k (SALARY)', 'Student Alcohol Abuse Problem (T0329)', 'Problem with dropouts (T0333)',
    'Prof Dev Main Assign (T0159)', 'Prof Dev Methods of Teaching (T0165)', 'Total Years of Experiance (EXPER)'], 
    title=' Multiple Logistic Regression (Overall)'))
    print()

    model = smf.logit('LEAVER ~ AGE + S0287 + T0080 + SALARY + T0329 + T0333 + T0159 + T0165 + EXPER  ', data = df1).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept',
    'Younger than 40 (AGE)', '4 percent or less free lunch (S0287)','Masters Degree (T0080)',
    'Teachers that make less than 39,999k (SALARY)', 'Student Alcohol Abuse Problem (T0329)', 'Problem with dropouts (T0333)',
    'Prof Dev Main Assign (T0159)', 'Prof Dev Methods of Teaching (T0165)', 'Total Years of Experiance (EXPER)'], 
    title=' Multiple Logistic Regression (Male)'))
    print()

    model = smf.logit('LEAVER ~ AGE + S0287 + T0080 + SALARY + T0329 + T0333 + T0159 + T0165 + EXPER ', data = df2).fit()
    print(model.summary(yname="Status Leaver",
    xname=['Intercept',
    'Younger than 40 (AGE)', '4 percent or less free lunch (S0287)','Masters Degree (T0080)',
    'Teachers that make less than 39,999k (SALARY)', 'Student Alcohol Abuse Problem (T0329)', 'Problem with dropouts (T0333)',
    'Prof Dev Main Assign (T0159)', 'Prof Dev Methods of Teaching (T0165)', 'Total Years of Experiance (EXPER)'], 
    title=' Multiple Logistic Regression (Female)'))
    print()


def logiGraph(): # Need to try it with a continuous variable for independent
    # Overall Data on No mentorship or couch 
    # Train Test Split Method
    # plt.scatter(df.T0157, df.LEAVER, marker='+', color='red')

    # Setting Up the Training Data
    X_train, X_test, y_train, y_test = train_test_split(df[['T0159']],df.LEAVER, test_size=0.1)

    model = LogisticRegression()
    model.fit(X_train , y_train)

    # Predicting Data
    prediction = model.predict(X_test) # Predicting the model 
    rank = model.score(X_test,y_test) # Shows the accuracy of the model 
    probability = model.predict_proba(X_test) # Probability of individuals in the model Left Side 0 (Not Leave) | Right Side 1  (Leave) = Which you read as a percentage
    print(f'Score of No Mentorship: {rank}') # Value of 1.0 should be perfect. Currently it is 0.5691
    sns.regplot(x='T0159', y='LEAVER', data=df, logistic=True)
    plt.title("Logistic Regression (Logit Function Sigmoid -Curve) Leaver Vs. Prof Dev Main Assign ")
    plt.ylabel('Teacher Left')
    plt.xlabel('No Mentorship or Coaching')
    plt.close()

def logiGraph2(): # Contiuous Variable 

    # Graph Variable
    plt.scatter(df.EXPER, df.LEAVER, marker='+', color='red')

    # Setting Up the Training Data
    X_train, X_test, y_train, y_test = train_test_split(df[['EXPER']],df.LEAVER, test_size=0.1)

    model = LogisticRegression()
    model.fit(X_train , y_train)

    # Predicting Data
    prediction = model.predict(X_test) # Predicting the model 
    rank = model.score(X_test,y_test) # Shows the accuracy of the model 
    probability = model.predict_proba(X_test) # Probability of individuals in the model Left Side 0 (Not Leave) | Right Side 1  (Leave) = Which you read as a percentage
    print(f'Score of Total Experiance: {rank}') # Value of 1.0 should be perfect. Currently it is 0.5691
    # print(probability)
    sns.regplot(x='EXPER', y='LEAVER', data=df, logistic=True)
    plt.title("Logistic Regression (Logit Function Sigmoid -Curve) Leaver Vs. Experiance")
    plt.ylabel('Teacher Left')
    plt.xlabel('Total Years of Experiance')
    plt.show(block=False)
    plt.savefig('profiling/total_experiance_logi_graph.png')
    plt.pause(2)
    plt.close()


def logiGraph3():

    X_train, X_test, y_train, y_test = train_test_split(df[['T0329']],df.LEAVER, test_size=0.1)
    model = LogisticRegression()
    model.fit(X_test,y_test)
    rank = model.score(X_test,y_test)
    prediction = model.predict(X_test)
    print('Accuracy of logistic regression on Alcohol Abuse {:.2f}'.format(rank))

    # Confusion Matrix: You read it diagonally x1 + y2 = Correctly Predicted; X2
    conf_matrix = confusion_matrix(y_test,prediction)
    print(conf_matrix)









# playground()
# regressMulti5()
# regressMulti6()
# regressMulti7()
# regressMulti8()
# logiGraph()
# logiGraph2()
logiGraph3()
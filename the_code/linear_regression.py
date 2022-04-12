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

df = pd.read_csv('source\data_cleaning\cleaned_data.csv')
df1 = pd.read_csv('source\data_cleaning\cleaned_data_male.csv')
df2 = pd.read_csv('source\data_cleaning\cleaned_data_female.csv')



# Data for Graph
# Only do graphs for variables that correlate to our project. 

matplotlib.use('Qt5Agg')
sns.set_theme(style='ticks', color_codes=True)



# On statsmodels, It goes by numerical order of smallest to largest. 
def opportunites():
    print('I was pleased with the opportunities for professional DEVELOPMENT. (Overall)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('F0119')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(F0119)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Not Pleased Prof Dev (Overall)'],  
    title='Linear Regression on the Opportunities for professional development Variable against Leaver (Overall)'))
    print()
    print('I was pleased with the opportunities for professional DEVELOPMENT (Male).\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('F0119')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(F0119)', data = df1).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Not Pleased Prof Dev (Male)'],  
    title='Linear Regression on the Opportunities for professional development Variable against Leaver (Male)'))
    print()
    print('I was pleased with the opportunities for professional DEVELOPMENT (Female).\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('F0119')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(F0119)', data = df2).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Not Pleased Prof Dev (Female)'],  
    title='Linear Regression on the Opportunities for professional development Variable against Leaver (Female)'))
    print()




def graph():
    sns.heatmap(x="F0119", y="NEW_STATUS", data=df)
    plt.title("Scatter plot of leaver (dependent) and Opportunities of Professional Development")
    plt.ylabel('Left Teaching')
    plt.savefig('profiling/graph_scatter.png')
    plt.show(block=False)
    plt.pause(2)
    plt.close()



def IncreseSalaryProfDevelopment(): 
    print('Did you receive an increase in salary or other pay increases as a result of participating in professional development activities? (Overall)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0186')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0186)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'No Salary Increase bc of prof dev (Overall)'],  
    title='Linear Regression on the pay rewards increase of completing professional development variable against Leaver (Overall)'))
    print()
    print('Did you receive an increase in salary or other pay increases as a result of participating in professional development activities? (Male)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0186')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0186)', data = df1).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'No Salary Increase bc of prof dev (Male)'],  
    title='Linear Regression on the pay rewards increase of completing professional development variable against Leaver (Male)'))
    print()
    print('Did you receive an increase in salary or other pay increases as a result of participating in professional development activities? (Female)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0186')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0186)', data = df2).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'No Salary Increase bc of prof dev (Female)'],  
    title='Linear Regression on the pay rewards increase of completing professional development variable against Leaver (Female)'))
    print()



def regressFreeTraining():
    print('If free training available in this district? regardless of funding source, to prepare staff members to teach in fields with current or anticipated shortages? (Overall)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('S1628')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(S1628)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Yes Free Training is Avaliable', 'No Free Training Avaliable (Overall)'], # 0 is 
    title='Linear Regression on the Free Traininng Variable against Leaver (Overall)'))
    print()
    print('If free training available in this district? regardless of funding source, to prepare staff members to teach in fields with current or anticipated shortages? (Male)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('S1628')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(S1628)', data = df1).fit()
    print(model.summary(yname="Status Leaver", xname=['Yes Free Training is Avaliable', 'No Free Training Avaliable (Male)'], # 0 is 
    title='Linear Regression on the Free Traininng Variable against Leaver (Male)'))
    print()
    print('If free training available in this district? regardless of funding source, to prepare staff members to teach in fields with current or anticipated shortages? (Female)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('S1628')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(S1628)', data = df2).fit()
    print(model.summary(yname="Status Leaver", xname=['Yes Free Training is Avaliable', 'No Free Training Avaliable (Female)'], # 0 is 
    title='Linear Regression on the Free Traininng Variable against Leaver (Female)'))
    print()




def usefullDevelopment():
    print('Thinking about ALL of the professional development you have participated in over the past 12 months, how useful was it? (Overall)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0178')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0178)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Not useful prof dev (Overall)'], 
    title='Linear Regression on the Free Traininng Variable against Leaver (Overall)'))
    print()
    print('Thinking about ALL of the professional development you have participated in over the past 12 months, how useful was it? (Male)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0178')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0178)', data = df1).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Not useful prof dev (Male)'], 
    title='Linear Regression on the Free Traininng Variable against Leaver (Male)'))
    print()
    print('Thinking about ALL of the professional development you have participated in over the past 12 months, how useful was it? (Female)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0178')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0178)', data = df2).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Not useful prof dev (Female)'], 
    title='Linear Regression on the Free Traininng Variable against Leaver (Female)'))
    print()



def age(): 
    print('What are the ages of the teachers? (Overall) \n ')
    print("Preliminary Stats")
    pre_stats = df.groupby('AGE_T_x')["NEW_STATUS"].describe()
    print(pre_stats)
    print('\n \n')
    model = smf.ols('NEW_STATUS ~ C(AGE_T_x)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Younger than 40 (Overall)'], 
    title='Linear Regression on the Age Variable against Leaver (Overall)'))
    print()
    print('What are the ages of the teachers? (Male) \n ')
    print("Preliminary Stats")
    pre_stats = df.groupby('AGE_T_x')["NEW_STATUS"].describe()
    print(pre_stats)
    print('\n \n')
    model = smf.ols('NEW_STATUS ~ C(AGE_T_x)', data = df1).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Younger than 40 (Male)'], 
    title='Linear Regression on the Age Variable against Leaver (Male)'))
    print()
    print('What are the ages of the teachers? (Female) \n ')
    print("Preliminary Stats")
    pre_stats = df.groupby('AGE_T_x')["NEW_STATUS"].describe()
    print(pre_stats)
    print('\n \n')
    model = smf.ols('NEW_STATUS ~ C(AGE_T_x)', data = df2).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Younger than 40 (Female)'], 
    title='Linear Regression on the Age Variable against Leaver (Female)'))
    print()



def freeLunch(): 
    print('Around the first of October, how many applicants at this school were APPROVED for free or reduced-price lunches? (Overall)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('S0287')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(S0287)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', '20 percent or more free lunch (Overall)'], 
    title='Linear Regression on the Free Lunch Variable against Leaver (Overall)'))
    print()
    print('Around the first of October, how many applicants at this school were APPROVED for free or reduced-price lunches? (Male) \n')
    print("Preliminary Stats")
    pre_stats = df.groupby('S0287')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(S0287)', data = df1).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', '20 percent or more free lunch (Male)'], 
    title='Linear Regression on the Free Lunch Variable against Leaver (Male)'))
    print()
    print('Around the first of October, how many applicants at this school were APPROVED for free or reduced-price lunches? (Female)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('S0287')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(S0287)', data = df2).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', '20 percent or more free lunch (Female)'], 
    title='Linear Regression on the Free Lunch Variable against Leaver (Female)'))
    print()

# New Variables to add 

def mastersDegree():
    print('Does the teacher have a masters degree? (Overall)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0080')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0080)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'No Masters Degree (Overall)'], 
    title='Linear Regression on the Masters Degree Variable against Leaver (Overall)'))
    print()
    print('Does the teacher have a masters degree? (Male) \n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0080')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0080)', data = df1).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'No Masters Degree (Male)'], 
    title='Linear Regression on the Masters Degree Variable against Leaver (Male)'))
    print()
    print('Does the teacher have a masters degree? (Female)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('T0080')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(T0080)', data = df2).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'No Masters Degree (Female)'], 
    title='Linear Regression on the Masters Degree Variable against Leaver (Female)'))
    print()


def currentColab():
    print('There are many opportunities to collaborate with other teachers in this school? (Overall)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('F0603')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(F0603)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Disagree with coloboration with teachers (Overall)'], 
    title='Linear Regression on the Current Year Colaboration with teachers variable against Leaver (Overall)'))
    print()
    print('There are many opportunities to collaborate with other teachers in this school? (Male)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('F0603')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(F0603)', data = df1).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Disagree with coloboration with teachers (Male)'], 
    title='Linear Regression on the Current Year Colaboration with teachers variable against Leaver (Male)'))
    print()
    print('There are many opportunities to collaborate with other teachers in this school? (Female)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('F0603')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(F0603)', data = df2).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Disagree with coloboration with teachers (Female)'], 
    title='Linear Regression on the Current Year Colaboration with teachers variable against Leaver (Female)'))
    print()    


# def formerColab():
#     print('There are many opportunities to collaborate with other teachers in this school? (Overall)\n')
#     print("Preliminary Stats")
#     pre_stats = df.groupby('F0115')["NEW_STATUS"].describe()
#     print(pre_stats)
#     print()
#     model = smf.ols('NEW_STATUS ~ C(F0115)', data = df).fit()
#     print(model.summary(yname="Status Leaver", xname=['Intercept', 'Disagree with coloboration with teachers (Overall)'], 
#     title='Linear Regression on the former year Colaboration with teachers variable against Leaver (Overall)'))
#     print()
#     print('There are many opportunities to collaborate with other teachers in this school? (Male)\n')
#     print("Preliminary Stats")
#     pre_stats = df.groupby('F0115')["NEW_STATUS"].describe()
#     print(pre_stats)
#     print()
#     model = smf.ols('NEW_STATUS ~ C(F0115)', data = df1).fit()
#     print(model.summary(yname="Status Leaver", xname=['Intercept', 'Disagree with coloboration with teachers (Male)'], 
#     title='Linear Regression on the former year Colaboration with teachers variable against Leaver (Male)'))
#     print()
#     print('There are many opportunities to collaborate with other teachers in this school? (Female)\n')
#     print("Preliminary Stats")
#     pre_stats = df.groupby('F0115')["NEW_STATUS"].describe()
#     print(pre_stats)
#     print()
#     model = smf.ols('NEW_STATUS ~ C(F0115)', data = df2).fit()
#     print(model.summary(yname="Status Leaver", xname=['Intercept', 'Disagree with coloboration with teachers (Female)'], 
#     title='Linear Regression on the former year Colaboration with teachers variable against Leaver (Female)'))
#     print()

# def encouragement():
#     print('The school administrators behavior towards the staff was supportive and encouraging? (Overall)\n')
#     print("Preliminary Stats")
#     pre_stats = df.groupby('F0121')["NEW_STATUS"].describe()
#     print(pre_stats)
#     print()
#     model = smf.ols('NEW_STATUS ~ C(F0121)', data = df).fit()
#     print(model.summary(yname="Status Leaver", xname=['Intercept', 'Disagree with administrator encouragement (Overall)'], 
#     title='Linear Regression for administration encouragement for teachers variable against Leaver (Overall)'))
#     print()
#     print('The school administrators behavior towards the staff was supportive and encouraging? (Male)\n')
#     print("Preliminary Stats")
#     pre_stats = df.groupby('F0121')["NEW_STATUS"].describe()
#     print(pre_stats)
#     print()
#     model = smf.ols('NEW_STATUS ~ C(F0121)', data = df1).fit()
#     print(model.summary(yname="Status Leaver", xname=['Intercept', 'Disagree with administrator encouragement (Male)'], 
#     title='Linear Regression for administration encouragement for teachers variable against Leaver (Male)'))
#     print()
#     print('The school administrators behavior towards the staff was supportive and encouraging? (Female)\n')
#     print("Preliminary Stats")
#     pre_stats = df.groupby('F0121')["NEW_STATUS"].describe()
#     print(pre_stats)
#     print()
#     model = smf.ols('NEW_STATUS ~ C(F0121)', data = df2).fit()
#     print(model.summary(yname="Status Leaver", xname=['Intercept', 'Disagree with administrator encouragement (Female)'], 
#     title='Linear Regression for administration encouragement for teachers variable against Leaver (Female)'))
#     print()


def teacherEarnings():
    print('Base Salary Pay? (Overall)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('EARNALL')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(EARNALL)', data = df).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Teachers that make less than 39,999k (Overall)'], 
    title='Linear Regression for administration encouragement for teachers variable against Leaver (Overall)'))
    print()
    print('Base Salary Pay? (Male)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('EARNALL')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(EARNALL)', data = df1).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Teachers that make less than 39,999k (Male)'], 
    title='Linear Regression for administration encouragement for teachers variable against Leaver (Male)'))
    print()
    print('Base Salary Pay? (Female)\n')
    print("Preliminary Stats")
    pre_stats = df.groupby('EARNALL')["NEW_STATUS"].describe()
    print(pre_stats)
    print()
    model = smf.ols('NEW_STATUS ~ C(EARNALL)', data = df2).fit()
    print(model.summary(yname="Status Leaver", xname=['Intercept', 'Teachers that make less than 39,999k (Female)'], 
    title='Linear Regression for administration encouragement for teachers variable against Leaver (Female)'))
    print()


opportunites()
IncreseSalaryProfDevelopment()
regressFreeTraining()
usefullDevelopment()
age()
freeLunch()
mastersDegree()
currentColab()
# formerColab()
# encouragement()
teacherEarnings()
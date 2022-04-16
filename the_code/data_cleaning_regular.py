# Where Dr. Feng said to code Yes as 1; No as 0; regardless. 
# Git Branch Presentation
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

matplotlib.use('Qt5Agg')


df = pd.read_csv('source/merge/data-merge2.csv', low_memory=False)


# Reassigning Varables to {Status L:1, M:0, S:0}
df['NEW_STATUS'] = df.STATUS.map({'L':1, 'M':0, 'S':0}) #Coded for Leaver and mover 




df['S0287'] = df['S0287'].replace([1,2,3,4, -8, -9], [1, 1, 0, 0, np.nan, np.nan]) 
df['AGE_T_x'] = df['AGE_T_x'].replace([1,2,3,4], [1,1,0,0]) # Weighing teachers that are 39 years and younger since the need prof dev the most. And 40 and older are more likley at the end of their carrers. 
df['T0080'] = df['T0080'].replace([1,2,-8], [1,0,np.nan]) 
df['EARNALL'] = df['EARNALL'].replace([1,2,3,4], [1,1,0,0]) 
df['T0329'] = df['T0329'].replace([1,2,3,4], [1,1,0,0]) #Does have studeent use of aclcohol problem
df['T0333'] = df['T0333'].replace([1,2,3,4], [1,1,0,0]) # Problem with dropouts
df['T0336'] = df['T0336'].replace([1,2,3,4], [1,1,0,0]) # Problem with poverty
df['T0155'] = df['T0155'].replace([1,2], [1,0]) # Mentoring and/or peer observation and coaching, as part of a formal arrangement that is recognized or supported by the school or district
df['T0159'] = df['T0159'].replace([1,2], [1,0]) # participated in any professional development activities that focused on in-depth study of the content in your MAIN teaching assignment field?
df['T0165'] = df['T0165'].replace([1,2], [1,0])
# Testing varaibles that might have a correlation
# participated in any professional development activities that focused on methods of teaching?
# df['T0330'] = df['T0330'].replace([1,2,3,4], [1,1,0,0]) # does have a dug abouse problem at the school
# df['T0332'] = df['T0332'].replace([1,2,3,4], [1,1,0,0]) # Problem disrepect towards teachers
# df['T0157'] = df['T0157'].replace([1,2], [1,0]) # attending workshops, conferences or training
# df['T0182'] = df['T0182'].replace([1,2], [1,0]) # full or partial reimbursement of college tuition - which is type of support
# df['T0184'] = df['T0184'].replace([1,2], [1,0]) # Reimbersement for travel and or daily expenses
# df['T0174'] = df['T0174'].replace([1,2], [1,0]) # participated in any professional development activities that focused on student discipline and management in the classroom?
# df['NEWTCH'] = df['NEWTEACH'].replace([1,2], [1,0]) # Weighted for teachers has taught 3 years or less
# Ones we are targeting
# df['F0119'] = df['F0119'].replace([1,2,3,4,5,-8], [0,0,np.nan,1,1,np.nan])  
# df['T0178'] = df['T0178'].replace([1,2,3,4,5], [0,0,np.nan,1, 1]) 
# df['T0186'] = df['T0186'].replace([1,2], [1,0])  
# df['S1628'] = df['S1628'].replace([1,2,-9], [0,1,np.nan])  



#df = df[['NEW_STATUS', 'F0119', 'T0186', 'S1628', 'AGE_T_x', 'S0287', 'T0178',  'T0080', 'EARNALL', 'T0356',
#'T0329', 'T0330', 'T0332', 'T0333', 'T0336', 'T0155','T0157', 'T0182', 'T0184', 'T0159', 'T0165', 'T0174', 'TOTEXPER_x', 
#'T0277' , 'T0278' ]]

df = df[['NEW_STATUS','AGE_T_x', 'S0287', 'T0080', 'EARNALL', 'T0329', 'T0333', 'T0159', 'T0165', 'TOTEXPER_x', 'T0356', 'T0278', 'T0277', 'T0330']]
 

df2 = df[(df['T0356']==1)] # Male
df3 = df[(df['T0356']==2)] # Female

# Sending to new dataframe 
def dataClean():
    df.to_csv('source/data_cleaning/cleaned_data.csv')
    df2.to_csv('source/data_cleaning/cleaned_data_male.csv')
    df3.to_csv('source/data_cleaning/cleaned_data_female.csv')

df = df.rename(columns={'NEW_STATUS': 'LEAVER','AGE_T_x': 'AGE','TOTEXPER_x': 'EXPER', 'EARNALL': 'SALARY'})
df2 = df.rename(columns={'NEW_STATUS': 'LEAVER','AGE_T_x': 'AGE','TOTEXPER_x': 'EXPER', 'EARNALL': 'SALARY'})
df3 = df.rename(columns={'NEW_STATUS': 'LEAVER','AGE_T_x': 'AGE','TOTEXPER_x': 'EXPER', 'EARNALL': 'SALARY'})
sns.set(rc={"figure.figsize":(10, 8)}) #width=3, #height=4

def correlationMatrix():
    
    cor = df.corr().drop(columns=['T0356']).drop('T0356')
    cor.to_csv('source/matrix/overall_correlation_matrix.csv')
    cor2 = df2.corr().drop(columns=['T0356']).drop('T0356')
    cor2.to_csv('source/matrix/male_correlation_matrix.csv')
    cor3 = df3.corr().drop(columns=['T0356']).drop('T0356')
    cor3.to_csv('source/matrix/female_correlation_matrix.csv')
    print('\n\n')
    print('Correlation Matrix of overall data')
    print(cor)
    print('\n\n')
    print('Correlation Matrix of male data')
    print(cor2)
    print('\n\n')
    print('Correlation Matrix of female data')
    print(cor3)
    print()

    sns.heatmap(cor, annot=True, annot_kws={"size":10})
    plt.title("Heatmap of Correlation of Leaver (Overall)")
    plt.yticks(rotation=0)
    plt.show(block=False)
    plt.savefig('profiling/heatmap_correlation_overall_regular.png')
    plt.pause(2)
    plt.close()
   
    sns.heatmap(cor2, annot=True, annot_kws={"size":10})
    plt.title("Heatmap of Correlation of Leaver (Male)")
    plt.yticks(rotation=0)
    plt.show(block=False)
    plt.savefig('profiling/heatmap_correlation_male_regular.png')
    plt.pause(2)
    plt.close()
    
    sns.heatmap(cor3, annot=True, annot_kws={"size":10})
    plt.title("Heatmap of Correlation of Leaver (Female)")
    plt.yticks(rotation=0)
    plt.show(block=False)
    plt.savefig('profiling/heatmap_correlation_female_regular.png')
    plt.pause(2)
    plt.close()







def profiler():
    profile = ProfileReport(df, title='Graphing Targeted Variables', minimal=True)
    profile.to_file('profiling/project-profiling.html')

dataClean()
correlationMatrix() 
# profiler()
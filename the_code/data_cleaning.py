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

# Dataframe that was the final merge from Project two. (Concat t2, t3), (Merged S3a, S4a) (Merged Concat, Merged S3a, S4A)
df = pd.read_csv('source\merge\data-merge2.csv', low_memory=False)


# Reassigning Varables to {Status L:1, M:0, S:0}
df['NEW_STATUS'] = df.STATUS.map({'L':1, 'M':0, 'S':0}) #Coded for Leaver

# Ones we used F0119, T0186, S1628, AGE_T_x, S0287, T0178
df['S0287'] = df['S0287'].replace([1,2,3,4, -8, -9], [0, 0, 1, 1, np.nan, np.nan]) #Free Lunch Coded for 20% or more 

# Recoding Dummy Varibles to include 1 for our model 0 against our model
df['S1628'] = df['S1628'].replace([1,2,-9], [0,1,np.nan]) # Recoded for 0 = yes 1 = no to see if no free training avaible


# Ones we are targeting
df['F0119'] = df['F0119'].replace([1,2,3,4,5,-8], [1,1,np.nan,0,0,np.nan]) # To show that people disagreed with the opportunities with professional devlopment. 
df['T0178'] = df['T0178'].replace([1,2,3,4,5], [1,1,np.nan,0, 0]) # Not useful from all professional development you have done
df['T0186'] = df['T0186'].replace([1,2], [0,1]) # Has not had salary increase becasue of professional development.  


# Hard Recode for Younger Teachers 
df['AGE_T_x'] = df['AGE_T_x'].replace([1,2,3,4], [1,1,0,0]) # Weighing teachers that are 39 years and younger since the need prof dev the most. And 40 and older are more likley at the end of their carrers. 

# Professor wanted us to add more variables; These are our wild cards
# df['NEWTCH'] = df['NEWTEACH'].replace([1,2], [1,0]) # Weighted for teachers has taught 3 years or less
df['T0080'] = df['T0080'].replace([1,2,-8], [0,1,np.nan]) # No to teach having masters degree
df['F0603'] = df['F0603'].replace([1,2,3,4,5], [1,1,np.nan,0,0]) # Disagreed with there is many opportuniites to colab with teachers for the current year
df['F0115'] = df['F0115'].replace([1,2,3,4,5], [1,1,np.nan,0,0]) # Disagreed with there is many opportuniites to colab with teachers for the former year
df['F0121'] = df['F0121'].replace([1,2,3,4,5], [1,1,np.nan,0,0]) # Disagree with adminitrator encouagement
df['EARNALL'] = df['EARNALL'].replace([1,2,3,4], [1,1,np.nan,0]) # For teachers that make less than (39,999)

# Testing varaibles that might have a correlation
df['T0329'] = df['T0329'].replace([1,2,3,4], [1,1,0,0]) #Does have studeent use of aclcohol problem
df['T0330'] = df['T0330'].replace([1,2,3,4], [1,1,0,0]) # does have a dug abouse problem at the school
df['T0332'] = df['T0332'].replace([1,2,3,4], [1,1,0,0]) # Problem disrepect towards teachers
df['T0333'] = df['T0333'].replace([1,2,3,4], [1,1,0,0]) # Problem with dropouts
df['T0336'] = df['T0336'].replace([1,2,3,4], [1,1,0,0]) # Problem with poverty
df['F0752'] = df['F0752'].replace([1,2,3,4,5,-8], [0,0,1,1,1,np.nan]) # Disatisfied with working conditions
df['F0746'] = df['F0746'].replace([1,2,3,4,5,-8], [1,1,np.nan,0,0,np.nan]) # Do not have shared beliefs with collegues
df['F0747'] = df['F0747'].replace([1,2,3,4,5,-8], [1,1,np.nan,0,0,np.nan]) # There was not a great deal of cooperative effort among the staf
df['F0754'] = df['F0754'].replace([1,2,3,4,5,-8], [0,0,1,1,1,np.nan]) # Disasisfied with changes in job discription and details
df['F0757'] = df['F0757'].replace([1,2,3,4,5,-8], [0,1,1,1,1,np.nan]) # Importance of being laid off involuntary or voluntary
# df[''] = df[''].replace([], [])
# df[''] = df[''].replace([], [])
# df[''] = df[''].replace([], [])
# df[''] = df[''].replace([], [])


 

df = df[['NEW_STATUS', 'F0119', 'T0186', 'S1628', 'AGE_T_x', 'S0287', 'T0178',  'T0080', 'EARNALL', 'T0356',
'T0329', 'T0330', 'T0332', 'T0333', 'T0336', 'F0752', 'F0746', 'F0747']]
# Not in index  'TOTEXPER', 'NEWTCH', 

df2 = df[(df['T0356']==1)] # Male
df3 = df[(df['T0356']==2)] # Female

# Sending to new dataframe 
def dataClean():
    df.to_csv('source\data_cleaning\cleaned_data.csv')
    df2.to_csv('source\data_cleaning\cleaned_data_male.csv')
    df3.to_csv('source\data_cleaning\cleaned_data_female.csv')



def correlationMatrix():
    cor = df.corr()
    cor2 = df2.corr()
    cor3 = df3.corr()
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
    sns.heatmap(cor)
    plt.show()
    sns.heatmap(cor2)
    plt.show()
    sns.heatmap(cor3)
    plt.show()




def profiler():
    profile = ProfileReport(df, title='Graphing Targeted Variables', minimal=True)
    profile.to_file('profiling/project-profiling.html')

dataClean()
correlationMatrix() 
# profiler()
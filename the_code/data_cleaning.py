import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport



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


 

df = df[['NEW_STATUS', 'AGE_T_x','S1628', 'F0119', 'T0178', 'T0186', 'S0287','T0080','F0603','F0115','F0121', 'EARNALL', 'T0356']]
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




def profiler():
    profile = ProfileReport(df, title='Graphing Targeted Variables', minimal=True)
    profile.to_file('profiling/project-profiling.html')

dataClean()
correlationMatrix() 
# profiler()
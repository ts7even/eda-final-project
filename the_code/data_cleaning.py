import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport



# Dataframe that was the final merge from Project two. (Concat t2, t3), (Merged S3a, S4a) (Merged Concat, Merged S3a, S4A)
df = pd.read_csv('source/merge/data-merge2.csv', low_memory=False)

# Reassigning Varables to {Status L:1, M:0, S:0} or {Status L:0, M:1, S:1}
df['NEW_STATUS'] = df.STATUS.map({'L':1, 'M':0, 'S':0})
df['S0287'] = df['S0287'].replace([1,2,3,4, -8, -9], [0, 0, 1, 1, np.nan, np.nan]) #Free Lunch



# Recoding Dummy Varibles to include 1 for our model 0 against our model
df['S1628'] = df['S1628'].replace([1,2,-9], [0,1,np.nan]) # Recoded for 0 = yes 1 = no 
df['T0313'] = df['T0313'].replace([1,2,3,4], [1,1,0,0]) # Recoded for 1 = Agree 2 = Disagree
# Ones we are targeting
df['F0119'] = df['F0119'].replace([1,2,3,4,5,-8], [0,0,np.nan,1,1,np.nan])
df['F0120'] = df['F0120'].replace([1,2,3,4,5,-8], [1,1,np.nan,0,0,np.nan])
df['T0165'] = df['T0165'].replace([1,2], [1,0])
df['T0178'] = df['T0178'].replace([1,2,3,4,5], [0,np.nan, np.nan, np.nan, 1])
df['T0179'] = df['T0179'].replace([1,2], [1,0])
df['T0186'] = df['T0186'].replace([1,2], [0,1])



# Hard Recode 
df['AGE_T_x'] = df['AGE_T_x'].replace([1,2,3,4], [0,0,1,1])



# Ones we used F0119, T0186, S1628, AGE_T_x, S0287, T0178
# Shortening Data to only include what we are using 
df = df[['NEW_STATUS','AGE_T_x', 'F0119', 'T0186', 'S1628', 'S0287', 'T0178', 'T0356']]


df1 = df[(df['T0356']==1)] # male
df2 = df[(df['T0356']==2)] # female

# Sending to new dataframe 
def sendCleanData():
    df.to_csv('source/data_cleaning/cleaned_data.csv') # Overall 
    df1.to_csv('source/data_cleaning/cleaned_data_male.csv') # Male
    df2.to_csv('source/data_cleaning/cleaned_data_female.csv') # Female



def profiler():
    profile = ProfileReport(df1, title='Graphing Targeted Variables', minimal=True)
    profile.to_file('profiling/project-profiling.html')



# sendCleanData()
# profiler()

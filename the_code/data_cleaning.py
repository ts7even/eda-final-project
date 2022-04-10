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
df['T0178'] = df['T0178'].replace([1,2,3,4,5], [1,np.nan, np.nan, np.nan, 0])
df['T0179'] = df['T0179'].replace([1,2], [1,0])
df['T0186'] = df['T0186'].replace([1,2], [0,1])

# Hard Recode 
df['AGE_T_x'] = df['AGE_T_x'].replace([1,2,3,4], [0,0,1,1])







# Shortening Data to only include what we are using 
df = df[['NEW_STATUS', 'ATTACK', 'ASSIGN_x', 'AGE_T_x', 'TOTEXPER_x',
'SCHLEVEL_x', 'S1628', 'S0287', 'T0313', 'T0314', 'T0318', 'T0244', 'F0119',
'F0120','F0115','T0165', 'T0178', 'T0179', 'T0186']]


# Sending to new dataframe 
df.to_csv('source/data_cleaning/cleaned_data.csv')



print("Test to see if data has been cleaned")
sad1 = df['T0186'].describe()
print(sad1)


def profiler():
    profile = ProfileReport(df1, title='Graphing Targeted Variables', minimal=True)
    profile.to_file('profiling/project-profiling.html')


# profiler()
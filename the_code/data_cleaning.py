import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport



# Dataframe that was the final merge from Project two. (Concat t2, t3), (Merged S3a, S4a) (Merged Concat, Merged S3a, S4A)
df = pd.read_csv('source/merge/data-merge2.csv', low_memory=False)

# Reassigning Varables to {Status L:1, M:0, S:0} or {Status L:0, M:1, S:1}
df['NEW_STATUS'] = df.STATUS.map({'L':1, 'M':0, 'S':0})
df['S0287'] = df['S0287'].replace([-8, -9], np.nan)



# Recoding Dummy Varibles to include 1 for our model 0 against our model
df['ATTACK'] = df['ATTACK'].replace(2, 1) # Made attack in last 12 months and attack not in the last 12 months the same thing. 
df['S1628'] = df['S1628'].replace([1,2,-9], [0,1,np.nan]) # Recoded for 0 = yes 1 = no 
df['T0313'] = df['T0313'].replace([1,2,3,4], [1,1,0,0]) # Recoded for 1 = Agree 2 = Disagree




# Shortening Data to only include what we are using 
df = df[['NEW_STATUS', 'ATTACK', 'ASSIGN', 'AGE_T', 'TOTEXPER', 'SCHLEVEL_x', 'S1628', 'S0287', 'T0313', 'T0314', 'T0318', 'T0244']]


# Sending to new dataframe 
df.to_csv('source/data_cleaning/cleaned_data.csv')



print("Test to see if data has been cleaned")
sad1 = df['T0313'].describe()
print(sad1)


def profiler():
    profile = ProfileReport(df1, title='Graphing Targeted Variables', minimal=True)
    profile.to_file('profiling/project-profiling.html')


# profiler()










# df['F0119'] = df['F0119'].replace(-8, np.nan) Not in merge 2 data 
# df['F0115'] = df['F0115'].replace(-8, np.nan) Not in merge 2 data
# df['F0744'] = df['F0744'].replace(-8, np.nan) Not in merge 2 data
# Not in data: 'F0607', 'F0603', 'F0115','F0121', 'F0744'
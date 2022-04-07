import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport



# Dataframe that was the final merge from Project two. (Concat t2, t3), (Merged S3a, S4a) (Merged Concat, Merged S3a, S4A)
df = pd.read_csv('source\merge\data-merge2.csv', low_memory=False)

# Reassigning Varables to {Status L:1, M:0, S:0}
df['NEW_STATUS'] = df.STATUS.map({'L':0, 'M':1, 'S':1})

# NaN for variables with -8
df['SO287'] = df['S0287'].replace(['-8', '-9'], np.nan)
df['S1628'] = df['S1628'].replace(-9, np.nan)
# df['F0119'] = df['F0119'].replace(-8, np.nan) Not in merge 2 data 
# df['F0115'] = df['F0115'].replace(-8, np.nan) Not in merge 2 data
# df['F0744'] = df['F0744'].replace(-8, np.nan) Not in merge 2 data

# Not in data: 'F0607', 'F0603', 'F0115','F0121', 'F0744'

# Shortening the data to only include variables that we need: TOTEXPER is Continuous
df1 = df[['NEW_STATUS', 'ATTACK', 'ASSIGN', 'AGE_T', 'TOTEXPER', 'SCHLEVEL_x',  'S1628', 'S0287', 'T0313', 'T0314', 'T0318', 'T0244']]




# Sending to new dataframe 
def dataClean():
    df1.to_csv('source\data_cleaning\cleaned_data.csv')




def profiler():
    profile = ProfileReport(df1, title='Graphing Targeted Variables', minimal=True)
    profile.to_file('profiling/project-profiling.html')

dataClean()
profiler()
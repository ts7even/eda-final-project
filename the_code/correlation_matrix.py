import pandas as pd
import numpy as np


# Dataframe that was the final merge from Project two. (Concat t2, t3), (Merged S3a, S4a) (Merged Concat, Merged S3a, S4A)
df = pd.read_csv('source/merge/data-merge2.csv', low_memory=False)

df['NEW_STATUS'] = df.STATUS.map({'L':1, 'M':0, 'S':0})
# Ones we used F0119, T0186, S1628, AGE_T_x, S0287, T0178
# Reassigning Varables to {Status L:1, M:0, S:0} or {Status L:0, M:1, S:1}
df['S0287'] = df['S0287'].replace([1,2,3,4, -8, -9], [0, 0, 1, 1, np.nan, np.nan]) #Free Lunch
# Recoding Dummy Varibles to include 1 for our model 0 against our model
df['S1628'] = df['S1628'].replace([1,2,-9], [0,1,np.nan]) # Recoded for 0 = yes 1 = no 
# Ones we are targeting
df['F0119'] = df['F0119'].replace([1,2,3,4,5,-8], [0,0,np.nan,1,1,np.nan])
df['T0178'] = df['T0178'].replace([1,2,3,4,5], [0,np.nan, np.nan, np.nan, 1])
df['T0186'] = df['T0186'].replace([1,2], [0,1])


# Hard Recode 
df['AGE_T_x'] = df['AGE_T_x'].replace([1,2,3,4], [0,0,1,1])


# Shortening Data to only include what we are using 
df = df[['NEW_STATUS', 'AGE_T_x','S1628', 'F0119', 'T0178', 'T0186']]
def correlationMatrix():
    df1 = df
    cor = df1.corr()
    print('Correlation Matrix of cleaned Data \n')
    print(cor)

# correlationMatrix()    
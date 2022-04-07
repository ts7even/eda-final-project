import pandas as pd
import numpy as np 



# Dataframes and filtered variables 
df1 = pd.read_csv('source/datasets/SASS_99_00_T2_v1_0.csv', low_memory=False) # Current Teachers
df1_filter = df1[['STATUS', 'CNTLNUM']]
df2 = pd.read_csv('source/datasets/SASS_99_00_T3_v1_0.csv', low_memory=False) # Former Teachers 
df2_filter = df2[['STATUS', 'CNTLNUM']]



# Data Concatination to CSV. 
def teacherConcat():
    data_concat = pd.concat([df1_filter, df2_filter])
    data_concat.to_csv('source/concat/data-concat.csv')
    print('Data concatination worked')


# All questions answered for Step 1. 
def currentFormer():
    df3 = pd.read_csv('source/concat/data-concat.csv', low_memory=False)
    df3_observations = df3.shape[0]
    df3_variables = df3.shape[1]
    print(f'The amount of observations that conactinted is {df3_observations}')
    print(f'The amount of variables that concatinated is {df3_variables}')

# teacherConcat()
# currentFormer()
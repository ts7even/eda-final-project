import pandas as pd
import numpy as np 



# Dataframes and filtered variables 
df1 = pd.read_csv('source/datasets/SASS_99_00_T2_v1_0.csv', low_memory=False) # Current Teachers
df2 = pd.read_csv('source/datasets/SASS_99_00_T3_v1_0.csv', low_memory=False) # Former Teachers 




# Data Concatination to CSV. 
def teacherConcat():
    data_concat = pd.concat([df1, df2])
    data_concat.to_csv('source/concat/data-concat.csv')
    print('Data concatination worked')

teacherConcat()

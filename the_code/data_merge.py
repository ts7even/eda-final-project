import pandas as pd
import numpy as np 



# Dataframe
df1 = pd.read_csv('source/datasets/SASS_99_00_S3a_v1_0.csv') # Public School Control Numbers is (SCHCNTL - School Control Number)
df2 = pd.read_csv('source/datasets/SASS_99_00_S4a_v1_0.csv') # Public School Teachers (CNTLNUM- Disctric) and (SCHCNTL - School Control Number)



# Initial Merge
def dataMerge():
    data_merge = pd.merge(df1, df2, on="SCHCNTL")
    data_merge.to_csv('source/merge/data-merge.csv')
    # print('Data merged worked for public teacher and public school datasets.')

# Merging concatinated data with public teacher and schools datasets. 

def dataMerge2():
    df3 = pd.read_csv('source/concat/data-concat.csv')
    df4 = pd.read_csv('source/merge/data-merge.csv')
    data_merge2 = pd.merge(df3, df4, on='CNTLNUM')
    data_merge2.to_csv('source/merge/data-merge2.csv')


dataMerge()
dataMerge2()
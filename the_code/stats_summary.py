import pandas as pd
import numpy as np


df = pd.read_csv('source/data_cleaning/cleaned_data.csv')


# To See if Data Carried over and observations are the same.
test_observation1 = df['LEAVER'].describe()
test_observation2 = df['AGE'].describe()
test_observation3 = df['S0287'].describe()
test_observation4 = df['T0080'].describe()
test_observation5 = df['SALARY'].describe()
test_observation6 = df['T0329'].describe()
test_observation7 = df['T0333'].describe()
test_observation8 = df['T0159'].describe()
test_observation9 = df['T0165'].describe()
test_observation10 = df['EXPER'].describe()
print(f'Stats summary for Leaver:  {test_observation1}')
print('\n\n')
print(f'Stats summary for AGE:  {test_observation2}')
print('\n\n')
print(f'Stats summary for S0287:  {test_observation3}')
print('\n\n')
print(f'Stats summary for T0080:  {test_observation4}')
print('\n\n')
print(f'Stats summary for SALARY:  {test_observation5}')
print('\n\n')
print(f'Stats summary for T0329:  {test_observation6}')
print('\n\n')
print(f'Stats summary for T0333:  {test_observation7}')
print('\n\n')
print(f'Stats summary for T0159:  {test_observation8}')
print('\n\n')
print(f'Stats summary for T0165:  {test_observation9}')
print('\n\n')
print(f'Stats summary for EXPER: {test_observation10}')
print('\n\n')


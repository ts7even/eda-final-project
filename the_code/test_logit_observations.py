import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn as sk 
from sklearn.linear_model import LinearRegression
import scipy as sp
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
matplotlib.use('Qt5Agg')

df = pd.read_csv('source/data_cleaning/cleaned_data.csv')


# To See if Data Carried over and observations are the same.
test_observation1 = df['NEW_STATUS'].count()
test_observation2 = df['F0119'].count()
test_observation3 = df['T0186'].count()
test_observation4 = df['S1628'].count()
test_observation5 = df['AGE_T_x'].count()
test_observation6 = df['S0287'].count()
test_observation7 = df['T0178'].count()
test_observation8 = df['T0080'].count()
test_observation9 = df['EARNALL'].count()
test_observation10 = df['T0329'].count()
test_observation11 = df['T0330'].count()
test_observation12= df['T0332'].count()
test_observation13 = df['T0333'].count()
test_observation14 = df['T0336'].count()
test_observation15 = df['T0155'].count()
test_observation16 = df['T0157'].count()
test_observation17 = df['T0182'].count()
test_observation18 = df['T0184'].count()
test_observation19 = df['T0159'].count()
test_observation20 = df['T0165'].count()
test_observation21 = df['T0174'].count()
test_observation22 = df['TOTEXPER_x'].count()
print(f'Observations for NEW_STATUS:  {test_observation1}')
print(f'Observations for F0119: {test_observation2}')
print(f'Observations for T0186: {test_observation3}')
print(f'Observations for S1628: {test_observation4}')
print(f'Observations for AGE_T_x: {test_observation5}')
print(f'Observations for S0287: {test_observation6}')
print(f'Observations for T0178: {test_observation7}')
print(f'Observations for T0080: {test_observation8}')
print(f'Observations for EARNALL: {test_observation9}')
print(f'Observations for T0329: {test_observation10}')
print(f'Observations for T0330: {test_observation11}')
print(f'Observations for T0332: {test_observation12}')
print(f'Observations for T0333: {test_observation13}')
print(f'Observations for T0336: {test_observation14}')
print(f'Observations for T0155: {test_observation15}')
print(f'Observations for T0157: {test_observation16}')
print(f'Observations for T0182: {test_observation17}')
print(f'Observations for T0184: {test_observation18}')
print(f'Observations for T0159: {test_observation19}')
print(f'Observations for T0165: {test_observation20}')
print(f'Observations for T0174: {test_observation21}')
print(f'Observations for TOTEXPER_x: {test_observation22}')



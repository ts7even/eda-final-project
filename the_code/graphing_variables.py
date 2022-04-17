import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn as sk 
import scipy as sp
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Qt5Agg')

df = pd.read_csv('source/data_cleaning/cleaned_data.csv')
df1 = pd.read_csv('source/data_cleaning/cleaned_data_male.csv')
df2 = pd.read_csv('source/data_cleaning/cleaned_data_female.csv')

vari = ['AGE', 'S0287', 'T0080', 'SALARY', 'T0329', 'T0333', 'T0159', 'T0165', 'EXPER']
def graphVariables():
    df['AGE'] = df['AGE'].replace([0,1], ['Older than 40','Younger than 40'])
    sns.countplot('AGE', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Teacher Age')
    plt.show()

    df['S0287'] = df['S0287'].replace([0,1], ['More than 4 percent', 'Less than 4 percent'])
    sns.countplot('S0287', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Eligible for Free Lunch')
    plt.show()

    df['T0080'] = df['T0080'].replace([0,1], ['No Masters Degree', 'Yes Masters Degree'])
    sns.countplot('T0080', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Does the teacher have a masters degree?')
    plt.show()

    df['SALARY'] = df['SALARY'].replace([0,1], ['More than 40k', 'Less than 39k'])
    sns.countplot('SALARY', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Annual teacher earnings')
    plt.show()

    df['T0329'] = df['T0329'].replace([0,1], ['Minor to no problem', 'Serious Problem'])
    sns.countplot('T0329', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Student alcohol use problem')
    plt.show()

    df['T0333'] = df['T0333'].replace([0,1], ['Minor to no problem', 'Serious Problem'])
    sns.countplot('T0333', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Student drop out problem')
    plt.show()

    df['T0159'] = df['T0159'].replace([0,1], ['No', 'Yes'])
    sns.countplot('T0159', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Main Teaching Assingment Professional Development')
    plt.show()

    df['T0165'] = df['T0165'].replace([0,1], ['No', 'Yes'])
    sns.countplot('T0165', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Teaching Methods Professional Development')
    plt.show()

    # df[''] = df[''].replace([0,1], ['', ''])
graphVariables()
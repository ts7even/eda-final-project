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
df4 = pd.read_csv('source/data_cleaning/cleaned_data_experiance.csv')

vari = ['AGE', 'S0287', 'T0080', 'SALARY', 'T0329', 'T0333', 'T0159', 'T0165', 'EXPER']
def graphVariables():
    df['AGE'] = df['AGE'].replace([0,1], ['Older than 40','Younger than 40'])
    sns.countplot('AGE', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Teacher Age')
    plt.show(block=False)
    plt.savefig('profiling/presentation_graphs/presentation_AGE.png') 
    plt.pause(2)
    plt.close()

    df['YOUNG'] = df['YOUNG'].replace([0,1], ['Between 30 and 49 years old','Younger than 30'])
    sns.countplot('YOUNG', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Teacher Age')
    plt.show(block=False)
    plt.savefig('profiling/presentation_graphs/presentation_YOUNG.png') 
    plt.pause(2)
    plt.close()

    df['S0287'] = df['S0287'].replace([0,1], ['More than 4 percent', 'Less than 4 percent'])
    sns.countplot('S0287', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Eligible for Free Lunch')
    plt.show(block=False)
    plt.savefig('profiling/presentation_graphs/presentation_free_lunch.png') 
    plt.pause(2)
    plt.close()

    df['T0080'] = df['T0080'].replace([0,1], ['No Masters Degree', 'Yes Masters Degree'])
    sns.countplot('T0080', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Does the teacher have a masters degree?')
    plt.show(block=False)
    plt.savefig('profiling/presentation_graphs/presentation_masters_degree.png') 
    plt.pause(2)
    plt.close()

    df['SALARY'] = df['SALARY'].replace([0,1], ['More than 40k', 'Less than 39k'])
    sns.countplot('SALARY', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Annual teacher earnings')
    plt.show(block=False)
    plt.savefig('profiling/presentation_graphs/presentation_salary.png') 
    plt.pause(2)
    plt.close()

    df['T0329'] = df['T0329'].replace([0,1], ['Minor to no problem', 'Serious Problem'])
    sns.countplot('T0329', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Student alcohol use problem')
    plt.show(block=False)
    plt.savefig('profiling/presentation_graphs/presentation_alcohol_abuse.png') 
    plt.pause(2)
    plt.close()

    df['T0333'] = df['T0333'].replace([0,1], ['Minor to no problem', 'Serious Problem'])
    sns.countplot('T0333', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Student drop out problem')
    plt.show(block=False)
    plt.savefig('profiling/presentation_graphs/drop_out.png') 
    plt.pause(2)
    plt.close()

    df['T0159'] = df['T0159'].replace([0,1], ['No', 'Yes'])
    sns.countplot('T0159', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Main Teaching Assingment Professional Development')
    plt.show(block=False)
    plt.savefig('profiling/presentation_graphs/prof_dev_main_assign.png') 
    plt.pause(2)
    plt.close()

    df['T0165'] = df['T0165'].replace([0,1], ['No', 'Yes'])
    sns.countplot('T0165', data=df, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Teaching Methods Professional Development')
    plt.show(block=False)
    plt.savefig('profiling/presentation_graphs/prof_dev_teach_methods.png') 
    plt.pause(2)
    plt.close()

    df['T0165'] = df['T0165'].replace([0,1], ['No', 'Yes'])
    sns.countplot('T0165', data=df4, palette='Set3', hue='LEAVER')
    plt.legend(['Stayer','Leaver'])
    plt.title('Teaching Methods Professional Development Less than 1 Year Experiance')
    plt.show(block=False)
    plt.savefig('profiling/presentation_graphs/prof_dev_teach_methods_less_one_year_experiance.png') 
    plt.pause(2)
    plt.close()

    



def graphExperVar():
    plot_ = sns.countplot('EXPER', data=df, palette='Set3', hue='LEAVER')

    for ind, label in enumerate(plot_.get_xticklabels()):
        if ind % 5 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    plt.legend(['Stayer','Leaver'])
    plt.title('Teacher Experiance')
    plt.show(block=False) 
    plt.savefig('profiling/presentation_graphs/teacher_experiance.png') 
    plt.pause(2)
    plt.close()

# graphVariables()
graphExperVar()
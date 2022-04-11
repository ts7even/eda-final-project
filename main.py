from the_code import data_concat
from the_code import data_merge
from the_code import data_cleaning
from the_code import correlation_matrix
from the_code import linear_regression
from the_code import logistical_regression


def master():
    data_concat.teacherConcat()
    data_concat.currentFormer()
    data_merge.dataMerge()
    data_merge.dataMerge2()
    data_cleaning.sendCleanData()
    data_cleaning.profiler()
    print('\n\n\n\n\n\n\n')
    correlation_matrix.correlationMatrix()
    print('\n\n\n\n\n\n\n')
    linear_regression.graph()
    linear_regression.opportunites()
    linear_regression.IncreseSalaryProfDevelopment()
    linear_regression.regressFreeTraining()
    linear_regression.usefullDevelopment()
    linear_regression.age()
    linear_regression.freeLunch()
    linear_regression.regressMulti2()
    linear_regression.regressMulti3()
    linear_regression.regressMulti4()
    linear_regression.regressMulti5()
    linear_regression.regressMulti6()
    print('\n\n\n\n\n\n\n')
    logistical_regression.opportunites()
    logistical_regression.IncreseSalaryProfDevelopment()
    logistical_regression.regressFreeTraining()
    logistical_regression.usefullDevelopment()
    logistical_regression.age()
    logistical_regression.freeLunch()
    logistical_regression.regressMulti2()
    logistical_regression.regressMulti3()
    logistical_regression.regressMulti4()
    logistical_regression.regressMulti5()
    logistical_regression.regressMulti6()



master()
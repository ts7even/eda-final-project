from the_code import data_concat
from the_code import data_merge
from the_code import data_cleaning
from the_code import linear_regression
from the_code import logistical_regression


def master():
    data_concat.teacherConcat()
    data_concat.currentFormer()
    data_merge.dataMerge()
    data_merge.dataMerge2()
    data_cleaning.dataClean()
    data_cleaning.profiler()
# master()
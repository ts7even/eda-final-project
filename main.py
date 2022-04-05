
# Import code modulues
from the_code import step1_data_contcatination
from the_code import step2_data_merge
from the_code import step3_data_cleaning
from the_code import step4_exploratory_linear_regression

# Selecting and running codes
def main_step1():
    step1_data_contcatination.teacherConcat()
    step1_data_contcatination.currentFormer()

def main_step2():
    step2_data_merge.dataMerge()
    step2_data_merge.dataMerge2()

def main_step3():
    step3_data_cleaning.stopteach()
    step3_data_cleaning.teachage()
    step3_data_cleaning.sex()
    step3_data_cleaning.raza()
    step3_data_cleaning.teachex()
    step3_data_cleaning.mainass()
    step3_data_cleaning.padawan()
    step3_data_cleaning.enroll()
    step3_data_cleaning.schlvl()
    step3_data_cleaning.smite()


def main_step4():
    step4_exploratory_linear_regression.summaryStatsdependent()
    step4_exploratory_linear_regression.summaryStatsindependent()
    step4_exploratory_linear_regression.graphingMentorIndependentVariable()
    step4_exploratory_linear_regression.graphingAttackIndependentVariable()



main_step1()
main_step2()
main_step3()
main_step4()

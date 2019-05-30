'''
Kathryn (Katy) Koenig
CAPP 30254
'''
from sys import argv
import warnings
import descriptive_stats as ds
import updated_functions_hw5 as fn

warnings.filterwarnings("ignore")


def go(csvfile='projects_2012_2013.csv', outputfile='results.csv',
       descript_stats="No", model=None):
    '''
    Takes a csv file of data, cleans, runs multiple machine learning models,
    and saves the evaluations of these models in a csv file.

    Inputs:
        csvfile: csv file of data
        outputfile:
        descript_stats: a string denoting if descriptive statistics of the data should be printed
        model: list of ML models to be run or the string 'all'

    Outputs: None
    '''
    if not model:
        models_to_run = ['decision_tree', 'random_forest', 'knn',
                         'logistic_regression', 'ada_boost', 'bagging']
    elif model == 'all':
        models_to_run = ['decision_tree', 'random_forest', 'knn',
                         'logistic_regression', 'ada_boost',
                         'gradient_boost', 'bagging']
    else:
        models_to_run = model

    dataframe = fn.read(csvfile, 'projectid')
    fn.change_date_type(dataframe)
    if descript_stats == "Describe":
        ds.combine_des_stats(dataframe)
    date_lst = fn.temporal_validate(dataframe, 'date_posted', 6, 60)
    to_dummy = ['primary_focus_subject', 'primary_focus_area',
                'secondary_focus_subject', 'secondary_focus_area',
                'resource_type', 'grade_level', 'school_metro',
                'teacher_prefix', 'poverty_level']
    to_discretize = ['school_charter', 'eligible_double_your_impact_match',
                     'school_magnet']
    # I dropped 'datefullyfunded' to prevent leakage and 'school_city',
    # school_state', 'school_county', 'school_district' and the lat/long
    # columns would be problematic to convert to numerics.
    to_drop = ['school_latitude', 'school_longitude', 'datefullyfunded',
               'school_city', 'school_state', 'school_county',
               'school_district']
    thresh_lst = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    # We run the models below and save the results as a pandas dataframe
    results_df = fn.combining_function(date_lst, models_to_run, dataframe,
                                       'date_posted', to_dummy, to_discretize,
                                       thresh_lst, 'funded_by_deadline',
                                       to_drop)
    # Coverts pandas dataframe to csv for manual analysis
    results_df.to_csv(outputfile)
    # Evaluates our results automatically to show most relevant models
    eval_lst = ['accuracy_at_5', 'precision_at_5', 'recall_at_5',
                'f1_score_at_5', 'auc_roc_at_5']
    fn.results_eval(dataframe, results_df, 'date_posted', date_lst, to_dummy,
                    to_discretize, 'funded_by_deadline', to_drop, eval_lst)


if __name__ == "__main__":
    INPUT_CSV = argv[1]
    OUTPUTFILE_NAME = argv[2]
    DESCRIPT_STATS = argv[3]
    if len(argv) == 5:
        MODEL_LST = argv[4]
    else:
        MODEL_LST = None
    go(INPUT_CSV, OUTPUTFILE_NAME, DESCRIPT_STATS, MODEL_LST)

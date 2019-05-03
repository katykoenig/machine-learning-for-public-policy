'''
Kathryn (Katy) Koenig
CAPP 30254
'''
from sys import argv
import warnings
import descriptive_stats as ds
import updated_functions_hw3 as fn

warnings.filterwarnings("ignore")


def go(csvfile='projects_2012_2013.csv', outputfile='results.csv', DESCRIPT_STATS="No", model=None):
    '''
    Takes a csv file of data, cleans, runs multiple machine learning models, and saves the evaluations of these models in a csv file.

    Inputs:
        csvfile: csv file of data
        outputfile: csv output filename
        model: list of ML models to be run

    Outputs: None
    '''
    if not model:
        models_to_run = ['decision_tree', 'random_forest', 'knn',
                        'logistic_regression', 'ada_boost', 'bagging']
    elif model == 'all':
        models_to_run = ['decision_tree', 'random_forest', 'knn',
                         'logistic_regression', 'SVM', 'ada_boost',
                         'gradient_boost', 'bagging']
    else:
        models_to_run = model

    dataframe = fn.read(csvfile, 'projectid')
    fn.change_date_type(dataframe)
    if DESCRIPT_STATS == "Describe":
        sum_stats = ds.summary_stats(dataframe)
        print("Summary Statistics")
        print(sum_stats)
        null_summ = ds.check_null_values(dataframe)
        print("Summary of Null Values")
        print(null_summ)
        corr_df = ds.evaluate_correlations(dataframe)
        print("Correlations")
        print(corr_df)
    processed_df, _ = fn.preprocess(dataframe)
    to_dummy_lst = ['primary_focus_subject', 'primary_focus_area',
                    'secondary_focus_subject', 'secondary_focus_area',
                    'resource_type', 'grade_level', 'school_metro',
                    'teacher_prefix', 'poverty_level']
    processed_df = fn.make_dummy_cat(processed_df, to_dummy_lst)
    # I dropped 'datefullyfunded' to prevent leakage and 'school_city',
    # school_state', 'school_county', 'school_district' and the lat/long
    # columns would be problematic to convert to numerics.
    drop_lst = ['school_latitude', 'school_longitude', 'datefullyfunded',
                'school_city', 'school_state', 'school_county',
                'school_district']
    features_lst = fn.generate_features(processed_df, 'funded_by_deadline',
                                        drop_lst)
    features_lst, processed_df = fn.discretize_dates(processed_df, features_lst)
    master_dict, cleaned_df = fn.discretize_by_unique_val(processed_df,
                                                          ['school_charter',
                                                           'eligible_double_your_impact_match',
                                                           'school_magnet'])
    date_lst = fn.temporal_validate(cleaned_df, 'date_posted', 6)
    fn.combining_function(outputfile, date_lst, models_to_run, cleaned_df,
                          'date_posted', features_lst)
    fn.results_eval(outputfile, cleaned_df, 'date_posted', features_lst,
                    date_lst)


if __name__ == "__main__":
    INPUT_CSV = argv[1]
    OUTPUTFILE_NAME = argv[2]
    DESCRIPT_STATS = argv[3]
    if len(argv) == 5:
        MODEL_LST = argv[4]
    else:
        MODEL_LST = None
    go(INPUT_CSV, OUTPUTFILE_NAME, DESCRIPT_STATS, MODEL_LST)

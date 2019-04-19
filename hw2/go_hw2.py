'''
Kathryn (Katy) Koenig
CAPP 30254
'''
from sys import argv
import functions_hw2 as fn
import warnings
warnings.filterwarnings("ignore")

CSV_FILENAME = 'credit-data.csv'


def go(csvfile=CSV_FILENAME):
    '''
    Combines all functions from functions_hw2.py

    Inputs:
        csvfile: a csv file to be analyzed
s
    Outputs:
        Prints the following:
            - statistical summary generated dataframe
            - correlation generated dataframe
            - model accuracy score for decision tree
        Saves the following as a png file:
        -scatterplots for every variable in the generated dataframe
        -a decision tree
    '''
    dataframe = fn.read(csvfile)
    stats_df = fn.summary_stats(dataframe)
    print("Summary Statistics of Dataset")
    print(stats_df)
    print()
    corr_df = fn.evaluate_correlations(dataframe, 'corr_heatmap.png')
    print("Correlation Among Variables")
    print(corr_df)
    print()
    print("Number of Null Values per Column")
    null_table = fn.check_null_values(dataframe)
    print(null_table)
    print()
    #fn.show_distribution(dataframe)
    #fn.create_scatterplots(dataframe)
    processed_df, kept_col = fn.preprocess(dataframe)
    for col in list(kept_col):
        fn.make_dummy(processed_df, col, 'Freq_Late', 20)
    fn.discretize_variable_by_quintile(processed_df, 'MonthlyIncome')
    features_lst = fn.generate_features(processed_df, 'SeriousDlqin2yrs')
    x_train, x_test, y_train, y_test = fn.split_data(processed_df, features_lst, 'SeriousDlqin2yrs')
    decision_tree = fn.make_decision_tree(x_train, y_train)
    fn.rep_d_tree(decision_tree, features_lst)
    test_acc = fn.evaluate_model(decision_tree, x_test, y_test)
    print()
    print("Accuracy of Model on Test Set: {:.3f}".format(test_acc))


if __name__ == "__main__":
    if len(argv) < 2:
        go()
    elif len(argv) == 2:
        go(argv[1])

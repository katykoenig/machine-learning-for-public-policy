'''
Kathryn (Katy) Koenig
CAPP 30254

Functions for Creating ML Pipeline
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, \
     precision_score, recall_score, accuracy_score as accuracy, precision_recall_curve
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, \
     GradientBoostingClassifier, AdaBoostClassifier
from dateutil.relativedelta import relativedelta


def read(csv_filename, index='projectid'):
    '''
    Reads csv into a pandas dataframe

    Inputs:
        csv_filename: csv filename
        index: column to be specified as index column when data read into pandas dataframe

    Outputs: a pandas dataframe
    '''
    return pd.read_csv(csv_filename, index_col=index)


def check_null_values(dataframe):
    '''
    Counts the number of null values in each column of a dataframe

    Input: a pandas dataframe

    Output: a pandas series with the number of null values in each column
    '''
    return dataframe.isnull().sum(axis=0)


def temporal_validate(dataframe, col, window, val_time):
    '''
    Creates list of for start and end dates for training and testing sets of the data.

    Inputs:
        dataframe: a pandas dataframe
        col: date column
        window (integer): length of time for which we are predicting
        val_time (integer): length of time in which it takes to validate a project

    Outputs:
        dates_lst: a list of dates representing traning and testing start dates for testing set
    '''
    # First we initialize the start date as the earliest date in the df
    train_start_time = dataframe[col].min()
    # We never want any set to have a date that is larger than the latest date
    #in the set so we find the end time below
    end_time = dataframe[col].max()
    # We subtract 2 days in initialized our train)end_time (as well as our test_end_time:
    # 1 day for finding dates under our val_time (e.g. 59 in our data here)
    # 1 day for ensure no overlap between testing and training dates
    train_end_time = train_start_time + relativedelta(months=+window, days=-(val_time+2))
    test_start_time = train_end_time + relativedelta(days=+val_time+2)
    test_end_time = test_start_time + relativedelta(months=+window, days=-val_time-2)
    dates_lst = [(train_start_time, train_end_time, test_start_time, test_end_time)]
    # While loop below ensures that we never iterate past our max date in our column
    while end_time >= test_start_time + relativedelta(months=+window):
        train_end_time = test_end_time
        test_start_time += relativedelta(months=+window)
        test_end_time = test_start_time + relativedelta(months=+window, days=-val_time-2)
        dates_lst.append((train_start_time, train_end_time, test_start_time, test_end_time))
    return dates_lst


def split_and_clean_data(dataframe, date_col, date, dummy_lst, discretize_lst, target_att, drop_lst):
    '''
    Splits data into testing and training datasets and cleans/processes
    each training and testing individually

    Inputs:
        dataframe: a pandas dataframe
        date_col: column name which is relevant for the splitting into test/train sets
        date: tuple for dates on which training and testing sets are split
        dummy_lst: list of column names to be converted to dummy variables
        discretize_lst: list of column names to be discretized
        target_att: outcome variable to be prediced (a column name)
        drop_lst: list of column names to not be considered features

    Output:
        x_train: pandas dataframe with only features columns of training data
        x_test: pandas dataframe with only features columns of testing data
        y_train: pandas series with outcome column of training data
        y_test: pandas series with outcome column of testing data
    '''
    train_start_time = date[0]
    train_end_time = date[1]
    test_start_time = date[2]
    test_end_time = date[3]
    # Divides full dataframe into training dataframe
    training_df = dataframe[(dataframe[date_col] >= train_start_time)
                            & (dataframe[date_col] <= train_end_time)]
    # Cleans (makes dummies, discretizes columns) post-division into training/testing
    processed_training, _, training_dict = preprocess(training_df, dummy_lst, discretize_lst)
    features_lst, processed_training = generate_features(processed_training, target_att, drop_lst)
    x_train = processed_training[features_lst]
    y_train = check_for_funding(training_df, timeframe=60)
    testing_df = dataframe[(dataframe[date_col] >= test_start_time) & \
                 (dataframe[date_col] <= test_end_time)]
    processed_testing, _, testing_dict = preprocess(testing_df, dummy_lst, discretize_lst)
    # We do not keep the first output (updated features list) for the testing
    #as we only want to use features generated w/ the training set
    _, processed_testing = discretize_dates(processed_testing, features_lst)
    x_test = processed_testing[features_lst]
    y_test = check_for_funding(testing_df, timeframe=60)
    return x_train, x_test, y_train, y_test


def change_date_type(dataframe):
    '''
    Converts columns with dates to datetime objects

    Inputs: a pandas dataframe

    Outputs: None
    '''
    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])


def preprocess(dataframe, to_dummy_lst, cols_to_discretize):
    '''
    Preprocesses data through:
        - Fills null values with median values across columns
        - Cleans dataframe to drop very highly correlated variables

    Input:
        dataframe: a pandas dataframe
        to_dummy_lst: a list of column names to be converted to dummy columns
        cols_to_discretize: list of columns to be discretized

    Outputs:
        dataframe: a pandas dataframe
        kept_col: set of columns to be kept in dataframe
        master_dict: dictionary mapping binary encodings
    '''
    corr_df = dataframe.corr()
    drop_lst = []
    kept_col = []
    dataframe.fillna(dataframe.median(), inplace=True)
    # Loop below finds columns that exhibit (almost) perfect collinearity
    # The columns are saved as a set that is returned by this function
    for column in corr_df.columns:
        drop_lst += (corr_df.index[(abs(corr_df[column]) > 0.95) & \
                    (abs(corr_df[column]) < 1.00)].tolist())
        to_drop = set(drop_lst)
        kept_col += list(to_drop)[:1]
    # Loop below drops one of the collinear columns
    for column in to_drop:
        if column not in kept_col:
            dataframe.drop([column], axis=1, inplace=True)
    dataframe = make_dummy_cat(dataframe, to_dummy_lst)
    master_dict, dataframe = discretize_by_unique_val(dataframe, cols_to_discretize)
    return dataframe, set(kept_col), master_dict


def check_for_funding(dataframe, timeframe, col1='datefullyfunded',
                      col2='date_posted', target_col='funded_by_deadline'):
    '''
    Checks if project funded within given time frame and creates a new column
    on the dataframe reflecting this

    Inputs:
        dataframe: a pandas dataframe
        timeframe: an integer representing the number of days allowed to pass
        col1: column name with earlier/first date
        col2: column name with later/second date (to be compared with date
              in col1)
        target_col: desired name for target/outcome column

    Output:
        pandas series with outcome column of testing data
    '''
    dataframe['val_col'] = dataframe[col1] - dataframe[col2]
    dataframe['val_col'] = dataframe['val_col'].dt.days
    make_dummy_cont(dataframe, 'val_col', target_col, timeframe)
    return dataframe[target_col]


def make_dummy_cont(dataframe, column, desired_col_name, cutoff):
    '''
    Creates new column of dummy variables where the value becomes 1 if above a
    given cutoff point and 0 if below cutoff point and drops original column

    Inputs:
        dataframe: a pandas dataframe
        column: name of column to be converted to dummy variable column
        desired_col_name: new column name for dummy variable column
        cutoff: cutoff point for which new column value becomes 1 if above
                and 0 if below

    Outputs: None
    '''
    dataframe[desired_col_name] = np.where(dataframe[column] > cutoff, 1, 0)
    # We drop this column b/c it would exhibit perfect collinearity
    # with the newly created column
    dataframe.drop(column, axis=1, inplace=True)


def discretize_variable_by_quintile(dataframe, col_name):
    '''
    Discretizes and relabels values in a column by breaking it into quintiles

    Inputs:
        dataframe: a pandas dataframe
        col_name: name of column to be discretized into quintiles

    Outputs: a pandas dataframe
    '''
    dataframe[col_name] = pd.qcut(dataframe[col_name], 5,
                                  labels=[1, 2, 3, 4, 5])


def discretize_by_unique_val(dataframe, cols_to_discretize):
    '''
    Discretizes categorical columns in col_lst to integer values

     Inputs:
        dataframe: a pandas dataframe
        col_lst: list of column names to be discretized

    Ouptuts:
        master_dict: a dictionary with the column names, mapping the integer
                     values to their meanings
        dataframe: a pandas dataframe
    '''
    master_dict = {}
    for col in cols_to_discretize:
        discret_dict = {}
        counter = 0
        for i in dataframe[col].unique():
            discret_dict[i] = counter
            counter += 1
        dataframe[col] = dataframe[col].map(discret_dict)
        master_dict[col] = discret_dict
    return master_dict, dataframe


def discretize_dates(dataframe, features_lst):
    '''
    Converts datetime types into integer of month and adds new discretized
    date columns to features list

    Inputs:
        dataframe: a pandas dataframe
        features_lst: a list of columns

    Outputs:
        features_lst: a list of updated columns
        dataframe: updated pandas dataframe
    '''
    types_df = dataframe.dtypes.reset_index()
    datetime_df = types_df[types_df[0] == 'datetime64[ns]']
    to_discretize = list(datetime_df['index'])
    for col in to_discretize:
        new_col = "month_" + col[-6:]
        dataframe[new_col] = dataframe[col].dt.month
        if new_col not in features_lst:
            features_lst.append("month_" + col[-6:])
    return features_lst, dataframe


def make_dummy_cat(dataframe, col_lst):
    '''
    Creates new columns of dummy variables from categorical columns of the data

    Inputs:
        dataframe: a pandas dataframe
        col_lst: list of columns to convert to dummy columns

    Outputs: a pandas dataframe
    '''
    dfs_to_concat = [dataframe]
    for column in col_lst:
        dummy_df = pd.get_dummies(dataframe[column], prefix=column)
        dfs_to_concat.append(dummy_df)
    dataframe = pd.concat(dfs_to_concat, axis=1)
    for column in col_lst:
        dataframe.drop(column, axis=1, inplace=True)
    return dataframe


def generate_features(dataframe, target_att, drop_lst):
    '''
    Generates the list of features/predictors to be used in training model

    Inputs:
        dataframe: a pandas dataframe
        target_att: outcome variable to be prediced (a column name)
        drop_lst: list of columns to not be included in features

    Output:
        features_lst: list of column names of features/predictors
    '''
    features_lst = [i for i in list(dataframe.columns) if i != target_att
                    if "id" not in i if 'date' not in i if i not in drop_lst]
    return discretize_dates(dataframe, features_lst)


# The code below relies on Rayid Ghani's magic loop, found here:
# https://github.com/rayidghani/magicloops

def generate_binary_at_k(y_scores, k):
    '''
    Converts classifier predictions to binary based on desired
    percentage/threshold

    Inputs:
        y_scores: a series of probability prediction made by classifier
        k: a float, denoting the threshold

    Outputs: a pandas series of binary values
    '''
    cutoff_index = int(len(y_scores) * k)
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary


def joint_sort_descending(array_one, array_two):
    '''
    Sorts two arrays in descending order

    Inputs:
        array_one: a numpy array
        array_two: a numpy array

    Outputs: two sorted arrays
    '''
    idx = np.argsort(array_one)[::-1]
    return array_one[idx], array_two[idx]


PARAMS_DICT = {
    'random_forest': {'n_estimators': [10, 100, 1000], 'max_depth': [1, 5, 10],
                      'min_samples_split': [2, 5, 10], 'n_jobs': [-1]},
    'logistic_regression': {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10],
                            'solver': ['lbfgs']},
    'decision_tree': {'criterion': ['gini', 'entropy'], 'max_depth': [1, 5, 10],
                      'min_samples_split': [2, 5, 50]},
    'SVM': {},
    'knn': {'n_neighbors': [5, 10, 25, 50], 'weights': ['uniform', 'distance']},
    'ada_boost': {'algorithm': ['SAMME.R'],
                  'n_estimators': [1, 10, 100, 1000]},
    'gradient_boost': {'n_estimators': [10, 100], 'max_depth': [3, 5, 10]},
    'bagging': {'n_estimators': [10, 100], 'random_state': [0], 'n_jobs': [-1]}}

CLFS = {'random_forest': RandomForestClassifier(),
        'logistic_regression': LogisticRegression(),
        'SVM': svm.SVC(random_state=0, probability=True),
        'decision_tree': DecisionTreeClassifier(),
        'knn': KNeighborsClassifier(),
        'ada_boost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)),
        'gradient_boost': GradientBoostingClassifier(),
        'bagging': BaggingClassifier(DecisionTreeClassifier(max_depth=5))}

RESULTS_COLS = ['model', 'parameters', 'train_start', 'train_end', 'test_start',
                'test_end', 'test_baseline', 'accuracy_at_1', 'precision_at_1',
                'recall_at_1', 'f1_score_at_1', 'auc_roc_at_1', 'accuracy_at_2',
                'precision_at_2', 'recall_at_2', 'f1_score_at_2',
                'auc_roc_at_2', 'accuracy_at_5', 'precision_at_5',
                'recall_at_5', 'f1_score_at_5', 'auc_roc_at_5',
                'accuracy_at_10', 'precision_at_10', 'recall_at_10',
                'f1_score_at_10', 'auc_roc_at_10', 'accuracy_at_20',
                'precision_at_20', 'recall_at_20', 'f1_score_at_20',
                'auc_roc_at_20', 'accuracy_at_30', 'precision_at_30',
                'recall_at_1', 'f1_score_at_30', 'auc_roc_at_30',
                'accuracy_at_50', 'precision_at_50', 'recall_at_50',
                'f1_score_at_50', 'auc_roc_at_50']


def combining_function(date_lst, model_lst, dataframe, col, dummy_lst,
                       discretize_lst, threshold_lst, target_att, drop_lst):
    '''
    Creates models, evaluates models and writes evaluation of models to csv.

    Input:
        date_lst: a list of dates on which to split training and testing data
        model_lst: list of classifier models to run
        dataframe: a pandas dataframe
        col: target column for prediction
        dummy_lst: list of column names to be converted to dummy variables
        discretize_lst: list of column names to be discretized
        threshold_lst: list of threshold values
        target_att: outcome variable to be prediced (a column name)
        drop_lst: list of column names to not be considered features


    Outputs: a pandas dataframe with the results of our models
    '''
    results_df = pd.DataFrame(columns=RESULTS_COLS)
    # Loop through dates and create appropriate splits while also
    # cleaning/processing AFTER split
    check = []
    for date in date_lst:
        x_train, x_test, y_train, y_test = split_and_clean_data(dataframe, col,
                                                                date, dummy_lst,
                                                                discretize_lst,
                                                                target_att,
                                                                drop_lst)
        train_start = date[0]
        train_end = date[1]
        test_start = date[2]
        test_end = date[3]
        # Loop through models and differing parameters
        # while fitting each model with split data
        for model in model_lst:
            print('Running model ' + model + ' for test start date ' + str(test_start))
            clf = CLFS[model]
            params_to_run = PARAMS_DICT[model]
            # Loop through varying paramets for each model
            for param in ParameterGrid(params_to_run):
                row_lst = [model, param, train_start, train_end, test_start,
                           test_end, np.mean(y_test)]
                clf.set_params(**param)
                clf.fit(x_train, y_train)
                predicted_scores = clf.predict_proba(x_test)[:, 1]
                total_lst = []
                # Loop through thresholds,
                # and generating evaluation metrics for each model
                for threshold in threshold_lst:
                    y_scores_sorted, y_true_sorted = joint_sort_descending(
                        np.array(predicted_scores), np.array(y_test))
                    preds_at_k = generate_binary_at_k(y_scores_sorted,
                                                      threshold)
                    acc = accuracy(y_true_sorted, preds_at_k)
                    prec = precision_score(y_true_sorted, preds_at_k)
                    recall = recall_score(y_true_sorted, preds_at_k)
                    f_one = f1_score(y_true_sorted, preds_at_k)
                    auc_roc = roc_auc_score(y_true_sorted, preds_at_k)
                    total_lst += [acc, prec, recall, f_one, auc_roc]
                results_df.loc[len(results_df)] = row_lst + total_lst
    return results_df


#To understand plotting the AUC-ROC curve, this work was informed by the
#following site:
#https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

def results_eval(dataframe, results_df, col, dates, dummy_lst, discretize_lst,
                 target_att, drop_lst, evaluator_lst):
    '''
    Evaluates the results of the models run and creates AUC-ROC and
    precision-recall curves for models deemed best

    Inputs:
        dataframe: a pandas dataframe
        results_df:
        col: target column for prediction
        dates: a list of dates on which to split training and testing data
        dummy_lst: list of column names to be converted to dummy variables
        discretize_lst: list of column names to be discretized
        outcome variable to be prediced (a column name)
        drop_lst: list of column names to not be considered features
        evaluator_lst: list of evaluation metrics

    Outputs: None
    '''
    for date in dates:
        print("BEST MODELS FOR START TEST DATE " + str(date[2]))
        x_train, x_test, y_train, y_test = split_and_clean_data(dataframe,
                                                                col, date,
                                                                dummy_lst,
                                                                discretize_lst,
                                                                target_att,
                                                                drop_lst)
        specified_df = results_df[results_df['test_start'] == date[2]]
        for evaluator in evaluator_lst:
            print("BEST MODEL FOR " + evaluator)
            best_index = specified_df[evaluator].idxmax()
            best_mod = results_df.iloc[best_index, 0:2]
            print(best_mod)
            print(results_df.iloc[best_index, 17:22])
            print()
            create_curves(best_mod[0], best_mod[1], x_train, y_train, x_test, y_test)


def create_curves(model, params, x_train, y_train, x_test, y_test, threshold=.05):
    '''
    Prints area under the curve and creates and saves an ROC and precision-recall curves image

    Inputs:
        model: name of machine learning classifer
        params: params for classifier to run
        x_train: pandas dataframe with only features columns of training data
        x_test: pandas dataframe with only features columns of testing data
        y_train: pandas series with outcome column of training data
        y_test: pandas series with outcome column of testing data

    Outputs: None
    '''
    clf = CLFS[model]
    clf.set_params(**params)
    clf.fit(x_train, y_train)
    predicted_scores = clf.predict_proba(x_test)[:, 1]
    y_scores_sorted, y_true_sorted = joint_sort_descending(
        np.array(predicted_scores), np.array(y_test))
    preds_at_k = generate_binary_at_k(y_scores_sorted, threshold)
    auc = roc_auc_score(y_true_sorted, preds_at_k)
    print(model)
    print('AUC: %.3f' % auc)
    fpr, tpr, _ = roc_curve(y_true_sorted, preds_at_k)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    roc_title = "ROC " + model + " with " + str(params)
    plt.title(roc_title)
    plt.savefig(roc_title + '.png')
    plt.clf()
    plot_precision_recall_n(y_true_sorted, preds_at_k, model, params)


# The code below also comes from Rayid Ghani's magic loop, again found here:
# https://github.com/rayidghani/magicloops

def plot_precision_recall_n(y_true, y_score, model, params):
    '''
    Plots and saves precision-recall curve for a given model

    Inputs:
        y_true: pandas series with outcome column
        y_score: pandas series of predicted outcome
        model: name of machine learning classifer
        params: params for classifier to run

    Outputs: None
    '''
    precision_curve, recall_curve, pr_thresh = precision_recall_curve(y_true,
                                                                      y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresh:
        num_above_thresh = np.count_nonzero([y_score >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    _, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0, 1])
    ax1.set_ylim([0, 1])
    ax2.set_xlim([0, 1])
    p_r_title = "Precision-Recall " + model + " with " + str(params)
    plt.title(p_r_title)
    plt.savefig(p_r_title + '.png')
    plt.clf()

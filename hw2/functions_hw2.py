'''
Kathryn (Katy) Koenig
CAPP 30254

Functions for Creating ML Pipeline
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotnine as p9
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score as accuracy
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image


# Step 1: Read in csv

def read(csv_filename, index='PersonID'):
    '''
    Reads csv into a pandas dataframe

    Inputs:
        csv_filename: csv filename
        index: column to be specified as index column when data read into pandas dataframe

    Outputs: a pandas dataframe
    '''
    return pd.read_csv(csv_filename, index_col=index)


# Step 2: Explore Data

def summary_stats(dataframe):
    '''
    Creates table of summary statitics for each column of dataframe

    Input: a pandas dataframe

    Output: a table
    '''
    stats_df = dataframe.agg(['median', 'mean', 'max', 'min'])
    return stats_df.T


def evaluate_correlations(dataframe, output_filename):
    '''
    Presents information regarding the correlation of each variable/column of a dataframe

    Input: a pandas dataframe

    Outputs:
        corr_df: a dataframe, showing the correlation between each variable
        corr_heatmap: a heatmap, reflecting the correlation between each variable
    '''
    corr_df = dataframe.corr()
    corr_heatmap = sns.heatmap(corr_df, xticklabels=corr_df.columns, yticklabels=corr_df.columns)
    plt.title("Correlation of Variables")
    plt.tight_layout()
    corr_heatmap.figure.savefig(output_filename)
    plt.clf()
    return corr_df


def show_distribution(dataframe):
    '''
    Prints histograms of for each column of the dataframe

    Inputs: a pandas dataframe

    Outputs: histograms of each column of a given dataframe
    '''

    fig, ax = plt.subplots()
    for column in dataframe.columns:
        file_name = str(column) + 'histogram' + '.png'
        dataframe.hist(column=column, grid=False, sharey=True, alpha=0.5, figsize=(20, 10))
        plt.tight_layout()
        plt.savefig(file_name)
        plt.clf()


def create_scatterplots(dataframe, unique_id='PersonID'):
    '''
    Creates and saves scatterplots for each column in a dataframe

    Inputs:
        dataframe: a pandas dataframe
        unique_id: a pandas series representing a unique identifier for each observation

    Outputs: None
    '''
    reset_df = dataframe.reset_index()
    for column in dataframe.columns:
        file_name =  str(column) + 'scatterplot' + '.png'
        plt1 = p9.ggplot(reset_df, p9.aes(x=column, y=unique_id)) + p9.geom_point()
        print('Saving scatterplot: '  + file_name)
        p9.ggsave(filename=file_name, plot=plt1, device='png')


def check_null_values(dataframe):
    '''
    Counts the number of null values in each column of a dataframe

    Input: a pandas dataframe

    Output: a pandas series with the number of null values in each column
    '''
    return dataframe.isnull().sum(axis=0)


# Step 3: Pre-process data

def preprocess(dataframe):
    '''
    Preprocesses data through:
        - Fills null values with median values across columns
        - Cleans dataframe to drop very highly correlated variables

    Input: a pandas dataframe

    Outputs:
        dataframe: a pandas dataframe
        kept_col: set of columns to be kept in dataframe
    '''
    corr_df = dataframe.corr()
    drop_lst = []
    kept_col = []
    dataframe.fillna(dataframe.median(), inplace=True)
    for column in corr_df.columns:
        drop_lst += (corr_df.index[(abs(corr_df[column]) > 0.95) & \
                    (abs(corr_df[column]) < 1.00)].tolist())
        to_drop = set(drop_lst)
        kept_col += list(to_drop)[:1]
    for column in to_drop:
        if column not in kept_col:
            dataframe.drop([column], axis=1, inplace=True)
    return dataframe, set(kept_col)


def discretize_variable_by_quintile(dataframe, col_name):
    '''
    Discretizes and relabels values in a column by breaking it into quintiles

    Inputs:
        dataframe: a pandas dataframe
        col_name: name of column to be discretized into quintiles

    Outputs: a pandas dataframe
    '''
    dataframe[col_name] = pd.qcut(dataframe[col_name], 5, labels=[1, 2, 3, 4, 5])


def make_dummy(dataframe, column, desired_col_name, cutoff):
    '''
    Creates new column of dummy variables where the value becomes 1 if above a
    given cutoff point and 0 if below cutoff point and drops original column

    Inputs:
        dataframe: a pandas dataframe
        column: name of column to be converted to dummy variable column
        desired_col_name: new column name for dummy variable column
        cutoff: cutoff point for which new column value becomes 1 if above and 0 if below.

    Outputs: a pandas dataframe
    '''
    dataframe[desired_col_name] = np.where(dataframe[column] >= cutoff, 1, 0)
    dataframe.drop(column, axis=1, inplace=True)


# Step 4: Generate Features

def generate_features(dataframe, target_att):
    '''
    Generates the list of features/predictors to be used in training model

    Inputs:
        dataframe: a pandas dataframe
        target_att: outcome variable to be prediced (a column name)

    Output: features_lst(list)
    '''
    features_lst = [i for i in list(dataframe.columns) if i != target_att]
    return features_lst


# Step 5: Build Classifier

def split_data(dataframe, features_lst, target_att):
    '''
    Splits data into testing and training datasets

    Inputs:
        dataframe: a pandas dataframe
        features_lst: list of column names of features/predictors
        target_att: outcome variable to be prediced (a column name)
        test_size:

    Output:
        x_train: pandas dataframe with only features columns of training data
        x_test: pandas dataframe with only features columns of testing data
        y_train: pandas series with outcome column of training data
        y_test: pandas seires with outcome column of testing data
    '''
    x = dataframe[features_lst]
    y = dataframe[target_att]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=57)
    return x_train, x_test, y_train, y_test


def make_decision_tree(x_train, y_train):
    '''
    Makes a decision tree based on training data

    Inputs:
        x_train: pandas dataframe with only features columns of training data
        y_train: pandas series with outcome column of training data

    Output: a decision tree object
    '''
    dec_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    dec_tree.fit(x_train, y_train)
    return dec_tree


# Step 6: Evaluate Classifer

def evaluate_model(dec_tree, x_test, y_test):
    '''
    Evaluates predictive model based on accurately predicting outcome variable
    of testing dataset

    Input:
        dec_tree: a decision tree object
        x_test: pandas dataframe with only features columns of testing data
        y_test: pandas dataframe with only outcome column of testing data

    Output: test_acc (float)
    '''
    predicted_scores_test = dec_tree.predict_proba(x_test)[:, 1]
    threshold = 0.5
    calc_threshold = lambda x, y: 0 if x < y else 1
    predicted_test = np.array([calc_threshold(score, threshold) for score in
                               predicted_scores_test])
    test_acc = accuracy(predicted_test, y_test)
    return test_acc


def rep_d_tree(dec_tree, features_lst):
    '''
    Saves a .png representation of the decision tree

    Input: decision tree object

    Outputs: None
    '''
    dot_data = StringIO()
    export_graphviz(dec_tree, feature_names=features_lst, out_file=dot_data,
                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.write_png('decision_tree.png'))

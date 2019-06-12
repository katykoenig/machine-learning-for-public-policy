'''
Kathryn (Katy) Koenig
CAPP 30254

Functions for ML Pipeline
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image, display


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


def summary_stats(dataframe):
    '''
    Creates table of summary statistics for each column of dataframe

    Input: a pandas dataframe

    Output: a table
    '''
    summary = dataframe.describe()
    return summary

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
    plt.show()
    return corr_df


def show_distribution(dataframe):
    '''
    Saves a histogram for each column of the dataframe

    Inputs: a pandas dataframe

    Outputs: None
    '''
    dataframe.hist(grid=False, sharey=True, alpha=0.5, figsize=(20, 10))
    plt.tight_layout()
    plt.show()


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
    change_date_type(dataframe)
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
    return dataframe


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


def make_kmeans_model(dataframe, k):
    '''
    Builds kmeans model and creates column in dataframe with results

    Inputs:
        dataframe: a pandas dataframe
        k: an integer representing the number of desired clusters

    Outputs:
        dataframe: a pandas dataframe
        clust_mod:
        centroids:
    '''
    clust_mod = KMeans(n_clusters=k)
    clust_mod.fit(dataframe)
    labels = clust_mod.predict(dataframe)
    centroids = clust_mod.cluster_centers_
    dataframe['result'] = labels
    return dataframe, clust_mod, labels, centroids


def describe_clusters(dataframe, col1, col2, features):
    '''
    '''
    file_lst = []
    for label, group in dataframe.groupby(col1):
        clf = make_decision_tree(group[features], group[col2])
        print("Important Features for Cluster: " + str(label))
        for i, col in enumerate(features):
            if abs(clf.feature_importances_[i]) > 0.2:
                print(col)
        filename = 'decision_tree_cluster' + str(label) + ".png"
        rep_d_tree(clf, features, filename)
        print(filename)
        display(Image(filename))
        file_lst.append(filename)
        print()

def make_decision_tree(x_train, y_train):
    '''
    Makes a decision tree based on training data

    Inputs:
        x_train: pandas dataframe with only features columns of training data
        y_train: pandas series with outcome column of training data

    Output: a decision tree object
    '''
    dec_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    dec_tree.fit(x_train, y_train)
    return dec_tree


def rep_d_tree(dec_tree, features_lst, filename):
    '''
    Saves a .png representation of the decision tree

    Input: decision tree object

    Outputs: None
    '''
    dot_data = StringIO()
    export_graphviz(dec_tree, feature_names=features_lst, out_file=dot_data,
                    filled=True, rounded=True, special_characters=True,)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.write_png(filename))


def merge_clusters(dataframe, to_merge, new_label):
    '''
    Merges multiple clusters into one single cluster

    Inputs:
        dataframe:
        to_merge:
        new_labels:

    Outputs:
    '''
    copy_df = dataframe.copy()
    for val in to_merge:
        copy_df.loc[copy.result == val, 'result'] = new_label
    return copy_df


def recluster(dataframe, k):
    '''

    Inputs:
        dataframe:
        k:

    Outputs: an updated pandas dataframe
    '''
    copy_df = dataframe.copy()
    copy_df.drop(columns=['result'], inplace=True)
    copy_df, _, _, _ = make_kmeans_model(copy_df, k)
    return copy_df

def split_cluster(dataframe, label, k):
    copy_df = dataframe.copy()
    mask = copy_df.result == label
    next_avail_label = max(copy_df.result) + 1
    reclustered = recluster(dataframe[mask], k)
    for num in range(k):
        reclustered.loc[reclustered['result'] == num, 'result'] = next_avail_label
        next_avail_label += 1
    updated = pd.concat([copy_df[copy_df['result'] != label], reclustered])
    return updated


# def split_cluster(dataframe, label, k):
#     '''
#     Inputs:
#         dataframe:
#         label:
#         k:

#     Outputs: an updated pandas dataframe
#     '''

#     dataframe['result'].unique()


#     mask = dataframe['result'] == label
#     updated_section = recluster(dataframe[mask], k)
#     mask2 = dataframe['result'] != label
#     return dataframe[mask2], updated_section


#     def split_cluster(df, x_cols, cluster_to_split, k):
#     '''
#     splits clusters from cluster_to_split into k clusters
#     '''
#     tmp = df.copy()
#     unused_label = max(tmp.pred_label) + 1
#     cluster = tmp[tmp['pred_label'] == cluster_to_split]
#     kmean = KMeans(n_clusters=k).fit(cluster[x_cols])
#     tmp.loc[tmp['pred_label'] == cluster_to_split, 'pred_label'] = pd.Series(kmean.labels_, index=cluster.index) + unused_label
#     return tmp

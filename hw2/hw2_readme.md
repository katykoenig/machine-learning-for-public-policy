# Machine Learning Pipeline
### CAPP 30254 Homework 2


## Getting Started

This file contains the following python files:

1. functions_hw2.py
  * This contains all the functions for reading, cleaning and processing data as well as the functions for generating features and building and evaluating our model.
2. go_hw2.py
  * This aggregations the functions in functions_hw2.py and allows us to call our functions from the command line.

### Necessary Modules:
* numpy 1.16.2
* pandas 0.24.2
* matplotlib 3.0.3
* plotnine 0.5.1
* seaborn 0.9.0
* sklearn 0.20.3
* pydotplus 2.0.2

If using the data given in the writeup, from the command line, run
```
$ python go_hw2.py
```

If using another csv, from the command line, run
```
$ python go_hw2.py FILENAME.csv
```

This create and save the following data visualizations in the directory in which the command was run:

  1. corr_heatmap.png
  2. Histograms of every column in a given dataset, saved a png files
  3. Scatterplots of every column in given dataset, saved as png files
  4. decision_tree.png

This program will also print the following tables:
  1. Summary Statistics of Dataset
  2. Correlation Among Variables
  3. Number of Null Values per Column


We will explore the printed tables and saved visualizations in detail in the analysis below.

## Exploratory Analysis & Preprocessing of Data

We begin by converting the csv to a pandas dataframe and generating basic summary statistics.

* INSERT SUMMARY PNG

Then, we check for the amount of null values across columns:

Column Name | Number of Null Values 
--- | --- 
SeriousDlqin2yrs | 0
RevolvingUtilizationOfUnsecuredLines | 0
age | 0
zipcode | 0
NumberOfTime30-59DaysPastDueNotWorse | 0
DebtRatio | 0
MonthlyIncome |7974
NumberOfOpenCreditLinesAndLoans | 0
NumberOfTimes90DaysLate | 0
NumberRealEstateLoansOrLines | 0
NumberOfTime60-89DaysPastDueNotWorse | 0
NumberOfDependents | 1037

In our analysis, we fill the null values in each column with the median value of the column.

To get an understanding of the distribution of the variables as well as to see if there are any extreme outliers (perhaps indicating misreporting), we generate both histograms and scatterplots of each variable.

Here is the aggregation of the histograms of each variable:

Below, please find the scatterplots of for the following variables: age, NumberRealEstateLoansOrLines, MonthlyIncome, NumberOfTime30-59DaysPastDueNotWorse 

* INSERT SCATTERPLOTS

As we can see above, the distributions vary significantly across variables. Additionally, while there appears to be a few outliers across columns (liek income and age), we choose to include these observations in our model.

Next, we evaluate correlations through creating a correlation table. Because the table generated is somewhat large, it is easier to understand the correlations between the variables through a heatmap:

* INSERT HEATMAP PNG

As we can see in the generated table and the heatmap above, the following variables are highly correlated: 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate'.

Because of they exhibit (almost) perfect multicollinearity, we drop two of these variables. The code is abstracted to drop all correlated variables (save one) with correlations above 0.95. Because of this, the code chooses which of the three variables mentioned above to keep and which two to drop. Therefore, the column kept will vary between runs.

As we can see in the scatterplots for 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', there exists a very large margin between observations at the low end and high end of each spectrum, e.g. all observations either had fewer than 13 or more than 80 times past 30-59 days late. Because of extreme difference, we used the column of the three that was not dropped to create a dummy variable, 'Freq_Late'. If the observation was late more than 20 times, they were labeled "1" in 'Freq_Late', otherwise, they were label "0". We then dropped the original column to avoid correlaton.

We then transformed the 'MonthlyIncome' column into quartiles in which the lowest income quartile was labeled 1, the highest income quartile was labeled 5, etc.

## Predicting

In generating our list of features, we used all the variables except our target variable, 'SeriousDlqin2yrs' and our unique identifier/index 'PersonID'.

We used the scikit-learn package to split the date between testing and training sets, with 30% of the dataset held for testing, using random_state = 57 to split the data. We then created a decision tree with our split criteron being information gain ('entropy'). We also set the maximum depth to 5 to avoid overfitting. We used scikit-learn's fit method to fit our training data to the decision tree.

* ADD DECISION TREE PNG HERE

In the representation of our decision tree above, in the first line of each node, we can see the variable on which the node was split as well as the current entropy of each note. The "value" of each node shows how many observations in each node fall into our two categories. This also represented in the color of each node.

To evaluate our model, we evaluated the accuracy of our decision tree's prediction for the testing data. Our code prints the accuracy of our model on the command line, which evaluates to 0.85. While this is high accuracy, it is worth noting that of the total 41,016 observations given, only 6,620 observations were labeled a '1' in 'SeriousDlqin2yrs', or approximately 16%. This means that had our model predicted all observations as no, or '0', for 'SeriousDlqin2yrs', our accuracy would have been 84%.


![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

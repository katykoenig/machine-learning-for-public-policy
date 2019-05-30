# Machine Learning Updated Pipeline
### CAPP 30254 Homework 5


## Getting Started

This file contains the following python files:

1. updated_functions_hw5.py
  * This contains all the functions for reading, cleaning and processing data as well as the functions for generating features and building and evaluating our model.
2. descriptive_stats.py
  * This contains the functions for running the initial descriptive statistics.
3. go_hw3.py
  * This aggregations the functions in updated_functions_hw3.py and allows us to call our functions from the command line.

### Necessary Modules:
* numpy 1.16.2
* pandas 0.24.2
* matplotlib 3.0.3
* plotnine 0.5.1
* seaborn 0.9.0
* sklearn 0.20.3
* typed-ast 1.3.5
* dateutil 2.4.1
* csvkit 1.0.4


Running the following from the command line will print descriptive statistics, create and print the following:

   1. Summary statistics table
   2. Correlation table 

   and will save save the following as png files:
   1. Correlation heatmap
   2. Histaograms of the distributitions of columns

```
$ python go_hw5.py 'projects_2012_2013.csv' output_csvname 'Describe'
```
By default this command will run the following classifiers:
  * Decision tree
  * Random forest
  * KNN
  * Logisitic Regression
  * Ada Boost
  * Bagging of Decision Tree Classifier

If you would like to also include gradient boosting and support vector machine classifiers, from the command line, run:
```
$ python go_hw5.py 'projects_2012_2013.csv' 'outputfile.csv' 'Describe' 'all'
```

This program create and save the following data visualizations in the directory in which the command was run:
  1. corr_heatmap.png
  2. Histograms of every column in a given dataset, saved a png file
  3. CSV file ('outputfile.csv') with the accuracy, precision, recall, AUC-ROC score, f1 score of each of the following models (with varying parameters) for each of the test periods
  4. An AUC-ROC curve for each the models with the highest accuracy, precision, recall, AUC-ROC score, f1 score for each testing period, saved as png files
  5. A precision-recall curve for the models with the highest accuracy, precision, recall, AUC-ROC score, f1 score for each testing period, saved as png files

This program will also print the following:
  1. Summary Statistics of Dataset
  2. Correlation Among Variables
  3. Number of Null Values per Column
  4. "Best" models and their parameters and evaluation metrics at the 5% threshold
  5. AUC for "Best" models

This program will also print the following tables:
  1. Summary Statistics of Dataset
  2. Correlation Among Variables
  3. Number of Null Values per Column


  ======
  # Policy Report

 ## Data & Preprocessing Description

 To begin our analysis, we evaluated the null values in each column. The chart below summarizes the null values in the dataset:

Column Name | Number of Null Values 
--- | --- 
schoolid | 0
school_ncesid | 9233
school_latitude | 0
school_longitude | 0
school_city | 0
school_state | 0
school_metro | 15224
school_district | 172
school_county | 0
school_charter | 0
school_magnet | 0
teacher_prefix | 0
primary_focus_subject | 15
primary_focus_area | 15
secondary_focus_subject | 40556
secondary_focus_area | 40556
resource_type | 17
poverty_level | 0
grade_level | 3
total_price_including_optional_support | 0
students_reached | 59
eligible_double_your_impact_match | 0
date_posted | 0
datefullyfunded | 0


Because each school already has a distinct id, “schoolid”, we drop the “school_ncesid” column. In cleaning the data, we make the following columns dummy columns:

  * school_charter
  * eligible_double_your_impact_match
  * school_magnet
  * metro
  * primary_focus_subject
  * primary_focus_area
  * secondary_focus_area
  * secondary_focus_subject
  * grade_level

 Therefore, we do not need to address the issue of null values at this point because the null values will only have zeros in each of the dummy columns for the corresponding original column. We replaced the null values in the “students_reached” column with the median of the column. We ultimately decided to drop the small amount of observations with null values in the “school_district” column as we could not impute the data through searching for schools with the same “schoolid” that had a “school_district,” i.e. if “school_district” value was missing for an observation, all schools with the same “schoolid” as the observation also had no value in the “school_district” column.

  We also provide a quick summary of numeric columns below: 

stat | total_price_including_optional_support | students_reached | not_funded_by_deadline
--- | --- | --- | --- | 
count | 124976 | 124976 | 124976
mean | 654.0118114678019 | 95.44575998462979 | 0.288135
std | 1098.015853637623 | 163.48191151120417 | 0.452896
min | 92 | 1 | 0
25% | 345.81 | 23 | 0
50% | 510.5 | 30 | 0
75% | 752.96 | 100 | 1
max | 164382.84 | 12143 | 1

As we can see in our total dataset, the baseline for projects not being funded by the deadline of 60 days after posting is roughly 29%, meaning that a dummy classifier which classifies all predictions as 1, or not funded by the deadline, would be accurate in 29% of its classifications. We will examine our classifiers in comparison to this number to the section below.

In our analysis, we drop the 'datefullyfunded' column to prevent leakage and 'school_city', school_state', 'school_county', 'school_district', 'school_latitude' and 'school_longitude' columns because even if we converted the categorical columns to numeric, we would need to ensure that cities that are physically the closest were assigned numbers closer to each other, we could end up regressing to the mean. For example, if projects in the cities on the east and west coasts were more likely to be funded in sixty days, some models may take the mean of the respective latitudes and longitudes, resulting in outcomes stating that projects middle America is the most likely to be funded by the deadline, which is a very inaccurate outcome. 

 Below we also provide a heatmap of the correlations between numeric variables in the dataset as well as histograms of these columns as well:

  ![alt text](INSERTTABLEHERE)
  ![alt text](INSERTTABLEHERE)

 ## Feature Selection & Model Selection

We used the following as features when running our models:

 * whether school was in urban, suburban or rural area, each as a dummy column
 * whether a school was a charter school or not
 * whether a school was a magnet school or 
 * whether the prefix list for a teacher was "Mrs.", "Mr." or "Ms.", each as a dummy column
 * Whether the school had the following primary or secondary focus areas: Math & Science, History & Civics, Literacy & Language, Applied Learning, Music & The Arts, Health & Sports, Special Needs, each a dummy column
 * Whether the school had the following primary or secondary focus subject: Mathematics, Civics & Government, Literacy, Other, Social Sciences, Visual Arts, Health & Wellness, Environmental Science, Literature & Writing, Music, History & Geography, Health & Life Science, Special Needs, ESL, Character Education, Early Development, Performing Arts, Nutrition, Gym & Fitness, Applied Sciences, College & Career Prep, Community Service, Sports, Extracurricular, Foreign Languages, Economics, Parent Involvement, each a dummy column
 * Whether the reasoure type listed was supplies, books, technology, other, trips, visitors, each as a dummy column
 * Whether the school's poverty level was listed as highest poverty, high poverty, low poverty, moderate poverty, each as a dummy column
 * Whether the project was for a class in Grades PreK-2, Grades 3-5, Grades 9-12, Grades 6-8, each a dummy column
 * Total price of project
 * Whether the project was eligible for double impact
 * Number of students reached 
 * Month the project was posted


While the large amount of features allowed for some models to obtain high scores of our evaluation methods, the policy implications of utilizing so many features may be problematic as there are so many factors influencing the models. Specifically, if a school was looking to maximize the likelihood of its project to be funded, it may be difficult to discern the factors that could lead to their funding.

 ## Analysis of Outcome

Because we seek to intervene with 5% of the projects at highest risk of not getting fully funded by the deadline of 60 days after posting, we looked specifically at models performing the best at each metric (accuracy, precision, recall, AUC-ROC score, F1 score).

Model | Params | Start Train | End Train | Start Test | End Test | Accuracy | Precision | Recall | F1 | AUC-ROC
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
logistic regression | 'C': 0.01, 'penalty': 'l2', 'solver': 'lbfgs' |2012-01-01 | 2012-06-30 | 2012-07-01 | 2013-12-31 | 0.830886 | 0.814608 | 1 | 0.897833 | 0.670879
decision tree |'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 2| 2012-01-01 | 2012-06-30 | 2012-07-01 | 2012-12-31 | 0.743598 | 0.743466 | 1 | 0.85286 | 0.501004
decision tree | 'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 2| 2012-01-01 | 2012-12-31 | 2013-01-01 | 2013-07-01 | 0.762243 | 0.742092 | 1 | 0.851955 | 0.623669


We include charts of the precision-recall and AUC-ROC curves for the above mentioned models in "Precision-Recall-Curves" and "AUC-ROC-Plots" folders, respectively in this Github repository. We also include the precision-recall curve for our decision tree (with max depth 10) as well as the AUC-ROC curve for our logistic regression model below.

 ![alt text](INSERTTABLEHERE)
 ![alt text](INSERTTABLEHERE)

In the shorter training term, with training data from January 2012 to April 2012 (evaluation from May 2012 to June 2012) and testing data being from July 2012 through October 2012 (evaluation from November 2012 December 2012), our logistic regression model performed the best in all metrics

In the second timeframe with training data from January 2012 through October 2012 (evaluation from November 2012 to December 2012) and testing data from January 2013 through April 2013 (evaluation from May 2013 to June 2013), our ADA boosting model had the highest accuracy, precision, recall, AUC-ROC and F1 scores.

When we look at larger training sets with training data from January 2012 to April 2013 (evaluation May 2013 to June 2013) to predict testing data from July 2013 through October 2013 (evaluation November 2013 to December 2013), the 


gradient boosting model performed the best in the following metrics: accuracy, precision, recall, AUC-ROC score, F1 score while again one variant of our decision tree model produced the highest recall.

The persistently high precision score reflects very few false positives. This is most likely due to the lack of positive labels the models gave. The overall low recall scores reflect the high number of false negatives labeled by the models. This is to be expected as we are evaluating the models at a low threshold of 5 percent. 

Our decision tree outperformed not only random forests but other ensemble models on our larger datasets, suggesting that either our training data is not representative of our testing data or that our data is easily divisible, meaning that our decision tree is relatively stable. 

The shift from the initial success for the logistic regression model in the smallest training timeframe to worsening performance in longer training timeframes suggests that in the long term, our data is not linear. This could explain why our support machine vector model was not successful in its predictions under any evaluation metric. 

It is interesting to note that while decision trees performed the best in both the second and third testing tests, the model parameters changed. 

Of the models tested with these three specified time periods, we ultimately would recommend the decision tree model
that performed that best in the longest and last time period. The variation in time of the models deemed most successful based on our evaluation metrics does suggest that there may underlying trends in the data that can only be captured through training the dataset on the longest period possible. Similarly, if the intervention was to take place in 2019, it is worth noting that the delay from available data to data of policy implementation may result in unsuccessful interventions when our gradient model is deployed. 

If the recommendation we are making is for a short-term immediate intervention program, it may be worth running the models on a more short-term training set of the most recent data. For example, if the intervention program was to be implemented only from July 2019 through December 2019, it may be fruitful to run our models on data from July 2018 to December 2018 instead of all available data, which is more likely to capture long term trends as opposed to seasonal trends in data. 


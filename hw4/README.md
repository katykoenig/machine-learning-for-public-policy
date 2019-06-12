# Machine Learning: Homework 4 Clustering
### CAPP 30254 Homework 5

## Getting Started

This file contains the following files:
1. hw4pipeline.py
	* Python file that contains all functions for reading, cleaning and processing data
2. HW4.ipynb
	* Example usage of the functions in hw4pipeline.py

### Necessary Modules:
* numpy 1.16.2
* pandas 0.24.2
* matplotlib 3.0.3
* seaborn 0.9.0
* sklearn 0.21.2
* pydotplus 2.0.2

# Report & Analysis

## Preprocessing

Because we are not creating temporal splits in our analysis, we conduct preprocessing through filling in any null values with the median column value as well as dropping any variables that are almost perfectly multicollinear.

For our k-means analysis, we create the following features:

* whether the school was a charter school 
* whether the school was a magnet school
* total price of the project (including optional support)
* number of students reached
* whether the project was eligible for double your impact match
* resource type requested (e.g. books, technology, etc.) with each type as a dummy column
* grade level
* whether the school was in a rural, urban or suburban area, which each type as a dummy column
* whether the prefix listed for a teacher was "Mrs.", "Mr." or "Ms.", each as a dummy column
* poverty level, with "low", "moderate", "high" and "highest," each as a dummy column

Note that we do not include every possible feature in our analysis as the multidimensionality created by doing so would make it difficult to analyze via clustering.

## Analysis on Overall Data

When we create three clusters on our total dataset, it is interesting to note the uneven distribution of the sizes of the clusters: Specifically, our first cluster includes 124,472, our second 10 and our third 494, suggesting that the initial centriods chosen may have been quite close to each other. Because we used the default ‘k-means++’ for initialization, the skewedness of the size of clusters may be due to the need of scaling. Moreover, as we can see in the decision tree below, our largest cluster (0), splits on the price of the project twice (and it is the most important feature of this cluster). We have not normalized our data, so while many of our columns are dummies with maximum value of one, while our 'total_price_including_optional_support' has a range of 92.00 to 164382.84. We could perhaps get more even clusters if we scale this column (and other non-dummy columns) to be between 0 and 1 like our other columns. At the same time, we would be trading this off for losing information/details of our data. Also interesting to note for cluster 0 is that even after four splits, the majority of each split is funded by the deadline (visually depicted by the saturation of the orange color in the decision tree). Similarly, our other two clusters exhibit much less success in funding by our arbitrary 60 day deadline, meaning that our clustering did pick up on attributes that are correlated with our outcome variable of being funded with in 60 days.

 ![alt text](https://raw.githubusercontent.com/katykoenig/machine-learning-for-public-policy/master/hw5/correlations.png)


## Analysis on Top Five Percent

We created a simple logistic regression to identify the top five percent of projects that were given highest probability of failure (failure meaning not funded in the 60 day deadline). From the top five percent of at-risk projects, we again create three clusters. 

Again, we have uneven clusters -- although they are now centered around different attributes. Our largest cluster (Cluster 0), with 1813 obversations, splits at the root node on if the number of students reached is aboe or below 657. For all observations in this cluster that are below this number, each node is relatively below, meaning that the majority of projects in Cluster 0 that reach more than 657 students do not receive funding by the 60 day deadline. Our second cluster, centers around pricing, but here, only includes 2 observations. This would suggest that there are two outliers to the data relatively near each other other other attributes (as compared to our other projects) and the largest difference between them in the pricing. Our third clusters' most important feature is resource type (supplies).

Through comparison of our analysis from our clusters on the overall data and our clusters from the top 5 percent, we can see that while both sets of clusters focused on similar defining characteristics, the clusters of the overall data seemed to cluster based off characteristics that divided projects into successful/unsuccessful. We can see this in the coloring of the trees printed of each cluster: in our first analysis our largest cluster is all a hue of orange while in our second analysis, all of our clusters are mixed, implying that the projects in our top 5% may be more difficult to cluster.


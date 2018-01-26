# Setting working directory
setwd("C:/Users/Joabe/Dropbox/ACADEMICO/UDEMY/Machine Learning A-Z/Part 1 - Data Preprocessing")

# Data Preprocessing Template

## Importing the dataset
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3]

## Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools) # Same of selecting in Packages panel
set.seed(123) # seeding means that you will have a pre determined set

# split() - returns a vector of BOOLEAN: [1]  TRUE  TRUE  TRUE  TRUE FALSE  TRUE  TRUE  TRUE FALSE  TRUE
# arg1 - Dependent variable
# arg2 - ratio of training set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


## Feature Scaling
# Error in colMeans(x, na.rm = TRUE) : 'x' must be numeric
# This error occurs because the categorical vars are factors, and factors 
# are not numbers.

# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])


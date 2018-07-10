# K-Means Clustering

# Importing the dataset
dataset = read.csv('Mall_Customers.csv')
X <- dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)
plot(1:10,
     wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Fitting K-Means to the dataset
set.seed(29)
kmeans = kmeans(x = X, centers = 5)

# Visualising the clusters
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')

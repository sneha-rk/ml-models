'''
	Implementation of KNN 
'''
import heapq
import numpy as np


# Calculates Euclidean distance between two points.
def EuclideanDistance(a,b):
	return np.linalg.norm(a-b)

# This function returns a list of the k-Nearest Neighbours
# given k and a test instance.
def KNearestNeighbours(X_train, x, k):
	distances = []
	i = 0
	for x_train in X_train:
		distances.append((EuclideanDistance(x_train, x),i))
		i += 1
	return heapq.nsmallest(k, distances)

# This classifies a set given k.
# It calls the k nearest neighbors for each instance, and takes
# a majority vote.
def KNeighboursClassifier(X_train, y_train, X_test, k):
	y_pred = np.zeros((X_test.shape[0]))

	it = 0
	for x_test in X_test:
		# Return the list of k nearest neighbours.
		k_nearest = KNearestNeighbours(X_train, x_test, k)

		indices = [int(i[1]) for i in k_nearest]
		votes = y_train[indices]
		counts = np.unique(votes, return_counts=True)
		i = np.where(counts[1] == counts[1].max())
		y_pred[it] = (counts[0][i[0][0]])
		it += 1

	return y_pred
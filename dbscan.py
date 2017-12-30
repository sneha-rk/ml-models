'''
    DBSCAN

    Also works for sparse matrices
'''

import numpy as np
from numpy import linalg as la
from sklearn.metrics import pairwise_distances

class MyDBSCAN:
    def __init__(self, X, eps, MinPts):
        # Properties
        self.X = X
        self.eps = eps
        self.MinPts = MinPts

        # Presomputed distances.
        self.pairwise_dists = pairwise_distances(X)

        # Part of algo to return. 0 means not classified yet.
        self.y_pred = np.zeros(X.shape[0])        
        self.cluster_id = 1

        # Finding the clusters.
    def fit_predict(self):
        # For all points, try to expand cluster or mark as noise.
        for i in range(0, self.X.shape[0]):
            # If unclassified.
            if self.y_pred[i] == 0:
                if self.expandCluster(i, self.cluster_id):
                    # All members of this cluster found. Now new cluster.
                    self.cluster_id += 1

        # Print the number of clusters.
        print(self.cluster_id-1)
        
        # Return the result.
        return self.y_pred

    # Expand the cluster using the ith point.
    def expandCluster(self, i, cluster_id):
        # Finding e-neighborhood.
        neighbor_pts = self.regionQuery(i)

        # Not a core point. In fact, mark as noise.
        if len(neighbor_pts) < self.MinPts:
            self.y_pred[i] = -1     # Noise
            return False
        
        # Else, we assign it and its neighbor the cluster_id.
        self.y_pred[i] = self.cluster_id
        for j in neighbor_pts:
            self.y_pred[j] = self.cluster_id

        # For all these neighbors.
        while len(neighbor_pts) > 0:
            # Find their neighbors.
            neighbor_pts_j = self.regionQuery(neighbor_pts[0])

            # If there are enough of them.
            if len(neighbor_pts_j) >= self.MinPts:
                # Add all to cluster.
                for k in neighbor_pts_j:
                    if self.y_pred[k] < 1:
                        if self.y_pred[k] == 0:
                            # And possibly consider for expansion.
                            neighbor_pts = np.append(neighbor_pts, k)
                        self.y_pred[k] = self.cluster_id 
            neighbor_pts = neighbor_pts[1:]

        return True
    
    # Returns all points in epsilon-neighborhood of ith point.
    def regionQuery(self, i):
        return np.where(self.pairwise_dists[i] < self.eps)[0]

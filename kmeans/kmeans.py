from scipy.spatial import distance
from sklearn import base
import numpy as np
import random

class KMeans(base.BaseEstimator):
    def __init__(self, k, distance_f=distance.euclidean):
        self.k = k
        self.distance_f = distance_f
        self.centroids = None
        
    def assign_to_cluster(self, X, centroids):
        clusters  = {}
        for x in X:
            best_centroid_index = self.find_closest(x, centroids)
                       
            if clusters.get(best_centroid_index):
                clusters[best_centroid_index].append(x)
            else:
                clusters[best_centroid_index] = [x]

        return clusters

    def find_closest(self, point, centroids):
        closest_centroid_dist = None
        closest_centroid_i = None

        for i, centroid in enumerate(centroids):
            curr_dist = distance.euclidean(point, centroid)
            if closest_centroid_dist is None or curr_dist < closest_centroid_dist:
                closest_centroid_i = i
                closest_centroid_dist = curr_dist

        return closest_centroid_i

    def reevaluate_centers(self, points_in_cluster):
        new_centroids = []

        for k in sorted(points_in_cluster.keys()):
            new_centroids.append(np.mean(points_in_cluster[k], axis = 0))

        return new_centroids

    def has_converged(self, mu, oldmu):
        return np.array_equal(mu, oldmu)

    def find_centers(self, X):
        # Initialize the centroids to K random centers
        old_centroids = random.sample(X, self.k)
        centroids = random.sample(X, self.k)

        while not self.has_converged(centroids, old_centroids):
            old_centroids = centroids
            clusters = self.assign_to_cluster(X, centroids)
            centroids = self.reevaluate_centers(clusters)

        return centroids

    def fit(self, X, y=None):
        self.centroids = self.find_centers(X)
        return self.centroids

    def predict(self, X, y=None):
        return self.find_closest(X, self.centroids)

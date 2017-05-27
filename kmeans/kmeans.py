from scipy.spatial import distance
import numpy as np
import random

def assign_to_cluster(X, centroids):
    clusters  = {}
    for x in X:
        best_centroid_index = find_closest(x, centroids) 
                   
        if clusters.get(best_centroid_index):
            clusters[best_centroid_index].append(x)
        else:
            clusters[best_centroid_index] = [x]

    return clusters

def find_closest(point, centroids):
    closest_centroid_dist = None
    closest_centroid_i = None

    for i, centroid in enumerate(centroids):
        curr_dist = distance.euclidean(point, centroid)
        if closest_centroid_dist is None or curr_dist < closest_centroid_dist:
            closest_centroid_i = i
            closest_centroid_dist = curr_dist

    return closest_centroid_i
 
def reevaluate_centers(points_in_cluster):
    new_centroids = []

    for k in sorted(points_in_cluster.keys()):
        new_centroids.append(np.mean(points_in_cluster[k], axis = 0))

    return new_centroids
 
def has_converged(mu, oldmu):
    return np.array_equal(mu, oldmu)
 
def find_centers(X, K):
    # Initialize the centroids to K random centers
    old_centroids = random.sample(X, K)
    centroids = random.sample(X, K)

    while not has_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = assign_to_cluster(X, centroids)
        centroids = reevaluate_centers(clusters)
        
    return(centroids, clusters)

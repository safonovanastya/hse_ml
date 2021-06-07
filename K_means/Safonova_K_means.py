import matplotlib.pyplot as plt
import numpy as np
import random
import time
import sys
import pandas as pd

def scatter_clusters(centers, spread, n_points):
    points = []
    n_points_in_clust = int(n_points/len(centers))
    random.uniform(10.5, 25.5)
    for i in range(len(centers)):
        for n in range(n_points_in_clust):
            x = centers[i][0] + random.uniform(spread[0], spread[1])
            y = centers[i][1] + random.uniform(spread[0], spread[1])
            arr = [x,y]
            points.append(arr)
    for n in range(len(points)):
        plt.plot(points[n][0], points[n][1], '.')
    plt.show()
    points = np.array(points)
    return points

points = scatter_clusters([[4,2], [4, 5], [2, 10]], (0.5,-0.5), 100)

points_a = points[:32]
points_b = points[33:66]
points_c = points[66:]

def kmeans_cluster_assignment(k, points, centers_guess, max_iterations, tolerance):
    colors = ['blue', 'green', 'red']
    percent_of_correct = []
    iters = []
    if centers_guess:
        centroids = centers_guess
    else:
        centroids = {}
        for i in range(k):
            centroids[i] = points[i]  
        
    for itr in range(max_iterations):
#        print('iteration ' + str(itr))
        classifications = {}
        for i in range(k):
            classifications[i] = []

        for features in points:
            distances = [np.linalg.norm(features-centroids[centroid]) for centroid in centroids]
            classification = distances.index(min(distances))
            classifications[classification].append(features)

        prev_centroids = dict(centroids)
        
        correct_a = []
        correct_b = []
        correct_c = []
        
        for classification in classifications:
            centroids[classification] = np.average(classifications[classification],axis=0)
            color = colors[classification]
            correct_a.append(len(np.intersect1d(points_a, classifications[classification]))/2)
            correct_b.append(len(np.intersect1d(points_b, classifications[classification]))/2)
            correct_c.append(len(np.intersect1d(points_c, classifications[classification]))/2)
            for f in classifications[classification]:
                plt.subplot(2,2,1)
                plt.scatter(f[0], f[1], marker="x", color=color)

        percent_of_correct.append((max(correct_a) + max(correct_b) + max(correct_c)) * 100 / len(points))
        iters.append(itr)
            
        stop = True

        
        for c in centroids:
            original_centroid = prev_centroids[c]
            current_centroid = centroids[c]
            n = np.sum((current_centroid-original_centroid)/original_centroid*100.0)
            if n > tolerance:
                stop = False

        if itr == 0 or itr == round(0.25*max_iterations) or itr == round(0.5*max_iterations) or itr == round(0.75*max_iterations) or itr == round(max_iterations):
            plt.subplot(2,2,2)
            plt.xlabel('iterations')
            plt.ylabel('percent of correct cluster')
            plt.tight_layout()
            plt.plot(iters, percent_of_correct, '-')
            
            plt.show()

        if stop is True:
            plt.show()
            return classifications
            break

start_time = time.time()
my_model = kmeans_cluster_assignment(3, points, False, 10, 0.001)
print("speed -- %s seconds " % (time.time() - start_time))
print("memory -- %s bytes" % (sys.getsizeof(my_model)))

from scipy.cluster.vq import vq, kmeans, whiten

start_time = time.time()
scipy_model = kmeans(points, 3, 10)
plt.scatter(points[:, 0], points[:, 1])
plt.scatter(scipy_model[0][:, 0], scipy_model[0][:, 1], c='r')

plt.show()
print("speed -- %s seconds " % (time.time() - start_time))
print('memory -- %s bytes' % (sys.getsizeof(scipy_model)))


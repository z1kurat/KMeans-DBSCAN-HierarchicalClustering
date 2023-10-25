from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
from typing import NoReturn

# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random",
                 max_iter: int = 300):
        """
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        """
        self.n_clusters: int = n_clusters
        self.max_iter: int  = max_iter
        self.init: str = init

    def euclidean_distances_squared(self, x: np.array, y: np.array) -> float:
        return np.sum((x - y) ** 2)

    def init_start_centroids(self, X: np.array, num_sample, num_feature) -> bool:
        if self.init == "random":
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            self.centroids = np.array([np.random.uniform(X_min, X_max) for _ in range(self.n_clusters)])
            # self.centroids: np.array = np.random.randn(self.n_clusters, num_feature)
        elif self.init == "sample":
            self.centroids: np.array = X[random.sample(range(num_sample), self.n_clusters),: ]
        elif self.init == "k-means++":
            self.centroids: np.array = np.empty((self.n_clusters, num_feature), dtype=X.dtype)
            self.centroids[0] = X[random.sample(range(num_sample), 1)]

            distances: np.array = np.zeros(num_sample)

            for i in range(1, self.n_clusters):
                distances_sum: float = 0
                for j in range(num_sample):
                    distances[j] = min(self.euclidean_distances_squared(X[j], centroid) for centroid in self.centroids[:i])
                    distances_sum += distances[j]

                probabilities = distances / distances_sum
                self.centroids[i] =  X[np.random.choice(range(num_sample), 1, p=probabilities)]

        return True

    def reinit_start_centroids(self, X: np.array, centroids_id: int, num_sample: int) -> bool:
        distances: np.array = np.zeros(num_sample)
        distances_sum: float = 0

        for j in range(num_sample):
            distances[j] = min(self.euclidean_distances_squared(X[j], centroid) for i, centroid in enumerate(self.centroids) if i != centroids_id)
            distances_sum += distances[j]

        probabilities = distances / distances_sum
        self.centroids[centroids_id] =  X[np.random.choice(range(num_sample), 1, p=probabilities)]

        return True

    def fit(self, X: np.array, y = None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        """
        num_sample, num_feature = X.shape
        self.init_start_centroids(X, num_sample, num_feature)

        cluster_assessment: np.array = np.zeros((num_sample, 2))
        cluster_assessment[:, 0] = -1

        interasions: int = 0
        cluster_changed = True

        while cluster_changed and interasions < self.max_iter:
            cluster_changed = True

            for i in range(num_sample):
                distanse_for_i_points: np.array = np.array([self.euclidean_distances_squared(X[i], centroid) for centroid in self.centroids])
                cluster: int = np.argmin(distanse_for_i_points)

                if cluster_assessment[i, 0] != cluster:
                    cluster_assessment[i, 0] = cluster
                    cluster_changed = True

            for j in range(self.n_clusters):
                points_for_j_cluster: np.array = X[cluster_assessment[:, 0] == j]
                points_cout_for_j_cluster: float = points_for_j_cluster.shape[0]
                if points_cout_for_j_cluster == 0:
                    self.reinit_start_centroids(X, j, num_sample)
                else:
                    self.centroids[j] = np.sum(points_for_j_cluster) / points_cout_for_j_cluster

            interasions += 1

    def predict_for_point(self, point: np.array) -> int:
        distanse_for_points: np.array = np.array([self.euclidean_distances_squared(point, centroid) for centroid in self.centroids])
        cluster: int = np.argmin(distanse_for_points)

        return cluster
    
    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        """
        num_sample = X.shape[0]
        predicts: np.array = np.zeros(num_sample, dtype=int)

        for i in range(num_sample):
            predicts[i] = self.predict_for_point(X[i])

        return predicts
    
# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть 
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean 
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.
        """
        self.eps: float = eps
        self.min_samples: int = min_samples
        self.leaf_size: int = leaf_size
        self.metric: str = metric
        
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).
        """
        num_sample = X.shape[0]

        tree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        labels = np.zeros(num_sample, dtype=int)

        point_to_neighbors = [[]] * num_sample

        for i, point in enumerate(X):
            point_to_neighbors[i] = tree.query_radius([point], r=self.eps)[0].tolist()

        cluster_label = 1
        for i in range(num_sample):
            neighbors = point_to_neighbors[i]
            neighbors_count = len(neighbors)

            if neighbors_count < self.min_samples:
                continue

            if labels[i] == 0:
                self.expand_cluster(labels, i, cluster_label, point_to_neighbors)
                cluster_label += 1

        return labels

    def expand_cluster(self, labels, point_index, cluster_label, point_to_neighbors):
        labels[point_index] = cluster_label
        stack = [point_index]

        while stack:
            current_index = stack.pop()
            neighbors = point_to_neighbors[current_index]

            for neighbor_index in neighbors:
                if labels[neighbor_index] == 0:
                    labels[neighbor_index] = cluster_label
                    stack.append(neighbor_index)

# Task 3

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    def cluster_distance(self, cluster1, cluster2):
        if self.linkage == 'average':
            return np.mean(np.sqrt(np.sum((cluster1[:, np.newaxis] - cluster2)**2, axis=1)))
        elif self.linkage == 'single':
            return np.min(np.sqrt(np.sum((cluster1[:, np.newaxis] - cluster2)**2, axis=1)))
        elif self.linkage == 'complete':
            return np.max(np.sqrt(np.sum((cluster1[:, np.newaxis] - cluster2)**2, axis=1)))


    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).
        """
        clusters = [[i] for i in range(len(X))]

        while len(clusters) > self.n_clusters:
            distances = self.calculate_distances(X, clusters)
            min_dist_index = np.unravel_index(np.argmin(distances), distances.shape)

            clusters[min_dist_index[0]] += clusters[min_dist_index[1]]
            clusters.remove(min_dist_index[1])

        labels = np.zeros(len(X))
        for i, cluster in enumerate(clusters):
            labels[cluster] = i

        return labels

    def calculate_distances(self, X: np.array, clusters: list) -> np.array:
        n_clusters = len(clusters)
        distances = np.zeros((n_clusters, n_clusters))

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                clusterA = [X[k] for k in clusters[i]]
                clusterB = [X[k] for k in clusters[j]]
                dist = self.cluster_distance(clusterA, clusterB)
                distances[i, j] = dist

        return distances

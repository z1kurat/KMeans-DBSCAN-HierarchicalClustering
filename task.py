from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
from scipy.spatial.distance import pdist, squareform
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
    def __init__(self, n_clusters: int, init: str = "random", max_iter: int = 300):
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

    def init_start_centroids(self, X: np.array, num_sample) -> bool:
        if self.init == "random":
            self.centroids = np.array([np.random.uniform(X.min(axis=0), X.max(axis=0)) for _ in range(self.n_clusters)])
        elif self.init == "sample":
            self.centroids: np.array = X[random.sample(range(num_sample), self.n_clusters),: ]
        elif self.init == "k-means++":
            self.centroids = []
            self.centroids.append(X[random.sample(range(num_sample), 1)])

            distances: np.array = np.zeros(num_sample)

            for _ in range(1, self.n_clusters):
                for j in range(num_sample):
                    distances[j] = min(self.euclidean_distances_squared(X[j], centroid) for centroid in self.centroids)

                self.centroids.append(X[np.argmax(distances)])

        return True

    def reinit_start_centroids(self, X: np.array, centroids_id: int, num_sample: int) -> bool:
        distances: np.array = np.zeros(num_sample)

        for j in range(num_sample):
            distances[j] = min(self.euclidean_distances_squared(X[j], centroid) for i, centroid in enumerate(self.centroids) if i != centroids_id)

        self.centroids[centroids_id] = X[np.argmax(distances)]

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
        num_sample = X.shape[0]
        self.init_start_centroids(X, num_sample)

        cluster_assessment: np.array = np.full(num_sample, -1, dtype=int)

        interasions: int = 0
        cluster_changed = True

        while cluster_changed and interasions < self.max_iter:
            cluster_changed = False

            for i in range(num_sample):
                distanse_for_i_points: np.array = np.array([self.euclidean_distances_squared(X[i], centroid) for centroid in self.centroids])
                cluster: int = np.argmin(distanse_for_i_points)

                if cluster_assessment[i] != cluster:
                    cluster_assessment[i] = cluster
                    cluster_changed = True

            for j in range(self.n_clusters):
                points_for_j_cluster: np.array = X[cluster_assessment == j]
                points_count_for_j_cluster: float = points_for_j_cluster.shape[0]
                if points_count_for_j_cluster == 0:
                    self.reinit_start_centroids(X, j, num_sample)
                else:
                    self.centroids[j] = np.mean(points_for_j_cluster, axis=0)

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
        labels = np.full(num_sample, -1, dtype=int)

        point_to_neighbors = tree.query_radius(X, r=self.eps).tolist()

        cluster_label = 0
        for i in range(num_sample):
            neighbors = point_to_neighbors[i]
            neighbors_count = len(neighbors)

            if neighbors_count < self.min_samples:
                continue

            if labels[i] == -1:
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
                if labels[neighbor_index] == -1:
                    labels[neighbor_index] = cluster_label
                    if len(point_to_neighbors[neighbor_index]) >= self.min_samples:
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

    def euclidean_distances_squared(self, x: np.array, y: np.array) -> float:
        return np.sum((x - y) ** 2)

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
        cluster_count = len(clusters)
        distances = squareform(pdist(X))
        np.fill_diagonal(distances, np.inf)

        while cluster_count > self.n_clusters:
            clusterA, clusterB = np.unravel_index(np.argmin(distances), distances.shape)
            distances = self.update_distances(distances, clusters, cluster_count, clusterA, clusterB)

            clusters[clusterA] += clusters[clusterB]
            del clusters[clusterB]
            cluster_count -= 1

        labels = np.zeros(len(X), dtype=int)
        for i, cluster in enumerate(clusters):
            labels[cluster] = i

        return labels

    def update_distances(self, distances: np.array, clusters, cluster_count, clusterA: int, clusterB: int):
        for i in range(cluster_count):
            if clusterA == i or clusterB == i:
                continue

            if self.linkage == 'average':
                clusterA_count = len(clusters[clusterA])
                clusterB_count = len(clusters[clusterB])

                averageA = clusterA_count * distances[i][clusterA]
                averageB = clusterB_count * distances[i][clusterB]
                total_count = clusterA_count + clusterB_count

                distances[i][clusterA] = (averageA + averageB) / (total_count)
                distances[clusterA][i] = distances[i][clusterA]
            elif self.linkage == 'single':
                distances[i][clusterA] = np.minimum(distances[i][clusterA], distances[i][clusterB])
                distances[clusterA][i] = distances[i][clusterA]
            elif self.linkage == 'complete':
                distances[i][clusterA] = np.maximum(distances[i][clusterA], distances[i][clusterB])
                distances[clusterA][i] = distances[i][clusterA]

        distances = np.delete(distances, clusterB, axis=0)
        distances = np.delete(distances, clusterB, axis=1)

        return distances

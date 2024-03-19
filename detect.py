"""
    3D Object detection class with Lidar data
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class Detection:
    def __init__(self):
        self.clusters = []
        self.cluster_centers = []
        self.cluster_labels = []
        self.cluster_sizes = []
        self.cluster_boxes
        self.cluster_boxes = []
        self.cluster_boxes_3d = []
        
    def clustering(self, points, n_clusters):
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        dbscan.fit(points)
        self.cluster_labels = dbscan.labels_
        
    def detect(self):
        pas

        
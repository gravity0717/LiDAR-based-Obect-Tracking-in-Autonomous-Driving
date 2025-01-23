from scipy import io
import numpy as np
import open3d as o3d
from dataset import Dataset
import cv2 
from sklearn.cluster import DBSCAN
    
dataset = Dataset("data/20220331T160645_ProjectData.mat")

dataiter = iter(dataset)

pcd, image, left, right = next(dataiter)

points = np.asarray(pcd.points)

dbscan = DBSCAN(eps=0.5, min_samples=10)

clusters = dbscan.fit_predict(points)

clustered_points = {}

# 각 포인트를 해당 클러스터의 리스트에 추가합니다.
clustered_points = {cluster: points[clusters == cluster] for cluster in np.unique(clusters)}

for cluster, points in clustered_points.items():
    print(f"Cluster {cluster}: {points.shape} points")
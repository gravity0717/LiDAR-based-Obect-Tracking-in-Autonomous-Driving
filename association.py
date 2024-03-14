import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

class Association:
    def __init__(self):
        self.clusters = {}  # 클러스터의 ID를 키로 사용하여 클러스터 정보를 저장
        self.dead_index = []  # 클러스터의 ID를 저장하여 삭제할 클러스터를 추적

    def registration(self, src, tar):
        threshold = 0.5
        T_init = np.array([[1, 0, 0, 0],  
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        reg_p2p = o3d.pipelines.registration.registration_icp(
            src, tar, threshold, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
       
        return reg_p2p.transformation
    
    def associate(self, transformation_matrix, prev_centroids, curr_centroids):
        moved_centroids = np.array([transformation_matrix @ np.hstack((prev_centroid, 1))
                                    for prev_centroid in prev_centroids])
        moved_centroids = moved_centroids[:, :3]
        neigh = NearestNeighbors(n_neighbors=1).fit(moved_centroids)
        
        closest_indices = np.array([neigh.kneighbors(curr_centroid.reshape(1, -1), return_distance=False)
                                    for curr_centroid in curr_centroids]).flatten()
        
        indices_table = list(zip(closest_indices, range(len(curr_centroids))))
        # indices_table = list(zip(range(len(prev_centroids), closest_indices)))
        return indices_table
        
    def update_clusters(self, T, prev_centroids, curr_centroids, geometry):
        if not self.clusters:  # 첫 프레임의 경우
            self.clusters = {id: {'centroid': centroid, 'age': 0} for id, centroid in enumerate(curr_centroids)}
        else:
            unmatched_indices = list(self.clusters.keys())
            matched_indices = []
            untracked_indices = set(range(len(curr_centroids)))
            
            table = self.associate(T, prev_centroids, curr_centroids)
            table = sorted(table, key=lambda x: x[0])
            
            for i, j in table:
                if i in unmatched_indices:
                    matched_indices.append(i)
                    unmatched_indices.remove(i)
                    untracked_indices.remove(j)
                    self.clusters[i]['centroid'] = curr_centroids[j]
                    
            # 새 클러스터에 대한 ID 할당
            next_id = max(self.clusters.keys()) + 1 if self.clusters else 0
            for u in untracked_indices:
                self.clusters[next_id] = {'centroid': curr_centroids[u], 'age': 0}
                next_id += 1
                
            # 'age'가 임계값 이상인 클러스터 처리
            for u in unmatched_indices:
                self.clusters[u]['age'] += 1
            
            # 화면에서 제거해야 할 클러스터 삭제
            to_remove = [id for id, cluster in self.clusters.items() if cluster['age'] >= 10]
            for id in to_remove:
                geometry.remove_text(str(id))
                del self.clusters[id]
            
            # 업데이트 또는 화면에 표시
            for cluster_id, cluster in self.clusters.items():
                if cluster['age'] < 10:
                    geometry.update_text(str(cluster_id), cluster['centroid'], f"ID: {cluster_id}, Age: {cluster['age']}")


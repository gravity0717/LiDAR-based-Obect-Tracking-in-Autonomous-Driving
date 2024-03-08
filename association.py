import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from geometry import Geometry
class Association:    
    def __init__(self):
        self.clusters = {}  # 클러스터의 ID와 Age를 저장

    def update_clusters(self, T, prev_centroids, new_centroids, geometry: Geometry):
        """
        이전과 현재 프레임의 클러스터 중심점을 기반으로 클러스터 정보를 업데이트합니다.
        """
        if not self.clusters:  # 첫 프레임의 경우
            self.clusters = {id: {'centroid': centroid, 'age': 0} for id, centroid in enumerate(new_centroids)}
        else:
            # 연관성 매칭 수행
            indices_table = self.associate(T, prev_centroids, new_centroids)

            # 새 클러스터 정보 준비
            new_clusters = {}
            cluster_keys = list(self.clusters.keys())

            # 연관된 클러스터 처리
            for prev_idx, new_idx in indices_table:
                if prev_idx < len(cluster_keys):
                    cluster_id = cluster_keys[prev_idx]

                    if new_idx < len(new_centroids):
                        new_clusters[cluster_id] = {'centroid': new_centroids[new_idx], 'age': self.clusters[cluster_id]['age'] + 1}
                    else:
                        new_clusters[cluster_id] = {'centroid': self.clusters[cluster_id]['centroid'], 
                                                    'age': self.clusters[cluster_id]['age'] + 1}
                    
                # Delete clusters over age older than 30 
                if new_clusters[cluster_id]['age'] > 30 or new_idx >= len(new_centroids):
                    del new_clusters[cluster_id]    
                    
                    # remove id from the scene
                    geometry.remove_text(str(cluster_id))
                    
            # 새로 발견된 클러스터 처리
            existing_ids = set(new_clusters.keys())
            next_id = max(existing_ids) + 1 if existing_ids else 0
            for i, centroid in enumerate(new_centroids):
                if i not in indices_table[:][1]:
                    new_clusters[next_id] = {'centroid': centroid, 'age': 0}
                    next_id += 1
                    
            self.clusters = new_clusters
            

    def registration(self, src, tar):
        threshold = 0.5
        T_init = np.array([[1, 0, 0, 0],  
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        
        reg_p2p = o3d.pipelines.registration.registration_icp(src, tar, threshold, T_init,
                                                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
       
        return reg_p2p.transformation
    
    def associate(self, tranformation_matrix, prev_centroids, new_centroids): 
        
        """
        Data Association / Registration with Nearest Neighbor method 
        """
        
        # Move
        moved_centroids = np.array([(tranformation_matrix @ np.hstack((prev_centroid,1)))
                                    for prev_centroid in prev_centroids])
        moved_centroids = moved_centroids[:,:3]
        # Nearest Neighbor 
        neigh = NearestNeighbors(n_neighbors=1).fit(moved_centroids)
        # Find Nearest Neighbor 
        associated_index = np.array([neigh.kneighbors(new_centroid.reshape(1,-1))[1].flatten() for new_centroid in new_centroids])
        
        # frame t 에서의 cluster index 와 ICP + NN 을 통해 얻은 cluster index를 같이 제공 
        indices_table = [(prev_cluster, int(new_cluster)) for prev_cluster, new_cluster in 
                         zip(range(len(new_centroids)), associated_index)]
        return indices_table
        
        
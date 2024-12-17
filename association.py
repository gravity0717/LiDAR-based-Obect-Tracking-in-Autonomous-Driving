import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from pprint import pprint 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from Index_matcher import IndexMatcher
from lap import lapjv

class Association:
    def __init__(self):
        # self.kalman = KalmanTracker()
        self.tracklets = {}  # 클러스터의 ID를 키로 사용하여 클러스터 정보를 저장
        self.matched_clusters = {}  # 현재 지금 매칭된 클러스터를 저장 
        self.dead_index = []  # 클러스터의 ID를 저장하여 삭제할 클러스터를 추적
        self.max_age = 1
        
        self.table = None 
        self.prev_id = [] # cluster tracklet 중에서 바로 이전 prev_id 만을 가져오기 위해 저장 
        self.history = {} # 클러스터 이력을 저장
        self.cost_matrix = None
        self.matcher = IndexMatcher()
        
    def calc_cost_matrix(self, prev_centroids, curr_centroids): 
        """
        두 프레임 간의 모든 클러스터 쌍에 대한 비용 행렬을 반환
        이때, 비용은 두 클러스터의 거리로 정의
        """
        cost_matrix = np.zeros((len(prev_centroids), len(curr_centroids)))

        for i, prev_centroid in enumerate(prev_centroids):
            for j, curr_centroid in enumerate(curr_centroids):
                cost_matrix[i, j] = np.linalg.norm(prev_centroid - curr_centroid) # L2 distance
        self.cost_matrix = cost_matrix
        return cost_matrix

    def associate(self, cost_matrix, thresh = 100):
        """
            비용 행렬을 입력으로 받아 최소 비용 할당을 반환
            헝가리안 알고리즘을 사용
        """ 
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lapjv(cost_matrix, extend_cost=True, cost_limit=thresh) # x[i]: i번째 행이 x[i] 열과 매칭됨
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_a, unmatched_b

    def table_id_to_cluster_id(self, id_matcher, table_id):
        """
            헝가리안 알고리즘을 통해 매칭된 클러스터 ID를 반환
        """
        for id, prev_id in id_matcher:
            if id == table_id:
                return prev_id
        return None
        
    def update_clusters(self, curr_centroids):
        if not self.tracklets:  # 첫 프레임의 경우
            self.tracklets = {id: {'centroid': centroid, 'age': 0} for id, centroid in enumerate(curr_centroids)}
            
        else:
            for id in self.tracklets.keys():
                if self.tracklets[id]['age'] >= self.max_age:
                    self.dead_index.append(id)
                    
            for id in self.dead_index:
                if id in list(self.tracklets.keys()):
                    _ = self.tracklets.pop(id)   
       
            prev_centroids = [self.tracklets[id]['centroid'] for id in self.tracklets.keys()]
            self.matcher.set_tracklet_index([id for id in self.tracklets.keys()])
            self.matcher.set_match()
        
            self.calc_cost_matrix(prev_centroids, curr_centroids)
            matched_indices, unmatched_indices, untracked_indices = self.associate(self.cost_matrix)

            for t_i, j in matched_indices:
                i = self.matcher.convert_table2tracklet(t_i) # taeble indx 와 tracklet index 는 다르다 
                self.tracklets[i]['centroid'] = curr_centroids[j]
                self.tracklets[i]['age'] = 0  # 갱신된 클러스터의 age 리셋

                    
            if len(unmatched_indices):    
                for t_u in unmatched_indices:
                    u = self.matcher.convert_table2tracklet(t_u) 
                    self.tracklets[u]['age'] += 1
                    
            # 새 클러스터 추가
            if len(untracked_indices):
                next_id = max(self.tracklets.keys(), default=0) + 1  # 비어 있을 경우 대비
                for u in untracked_indices:
                    self.tracklets[next_id] = {'centroid': curr_centroids[u], 'age': 0}
                  
                    
                    
        # record history along ID 
        for i, cluster in self.tracklets.items():
            if i not in self.history:
                self.history[i] = []
            self.history[i].append(cluster)        




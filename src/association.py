import numpy as np
import open3d as o3d
from lap import lapjv
from scipy.linalg import inv
from src.kalman import KalmanTracker
from src.index_matcher import IndexMatcher

 
class Association:
    def __init__(self):
        # self.kalman = KalmanTracker()
        self.tracklets = {}  # 클러스터의 ID를 키로 사용하여 클러스터 정보를 저장
        self.matched_clusters = {}  # 현재 지금 매칭된 클러스터를 저장 
        self.dead_index = []  # 클러스터의 ID를 저장하여 삭제할 클러스터를 추적
        self.max_age = 30
        
        self.table = None 
        self.prev_id = [] # cluster tracklet 중에서 바로 이전 prev_id 만을 가져오기 위해 저장 
        self.history = {} # 클러스터 이력을 저장
        self.cost_matrix = None
        self.matcher = IndexMatcher()
        self.living_track = {}
        
    # def calc_cost_matrix(self, prev_centroids, curr_centroids): 
    #     """
    #     두 프레임 간의 모든 클러스터 쌍에 대한 비용 행렬을 반환
    #     이때, 비용은 두 클러스터의 거리로 정의
    #     """
    #     cost_matrix = np.zeros((len(prev_centroids), len(curr_centroids)))
    #     for i, prev_centroid in enumerate(prev_centroids):
    #         for j, curr_centroid in enumerate(curr_centroids):
    #             cost_matrix[i, j] = np.linalg.norm(prev_centroid - curr_centroid[:2]) # L2 distance
    #     self.cost_matrix = cost_matrix
    #     return cost_matrix

    def calc_cost_matrix(self, prev_centroids, curr_centroids):
        """
        두 프레임 간의 모든 클러스터 쌍에 대한 비용 행렬을 반환
        이때, 비용은 Mahalanobis distance로 정의
        """
        cost_matrix = np.zeros((len(prev_centroids), len(curr_centroids)))
        
        for i, prev_centroid in enumerate(prev_centroids):
            for j, curr_centroid in enumerate(curr_centroids):

                # Kalman Tracker에서 공분산 행렬을 가져와서 Mahalanobis distance 계산
                prev_tracker = self.tracklets[self.matcher.convert_table2tracklet(i)]['tracker']

                # 공분산 행렬 추출
                prev_cov = prev_tracker.ekf.P[:2, :2]  # 2D만 사용
                
                # 공분산 행렬의 역행렬 계산
                prev_cov_inv = inv(prev_cov)
                
                # Mahalanobis distance 계산
                diff = prev_centroid[:2].flatten() - curr_centroid[:2].flatten()
                cost_matrix[i, j]  = np.sqrt(np.dot(np.dot(diff.T, prev_cov_inv), diff))
                
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
        
    def update_clusters(self, centroids):
        if not self.tracklets:  # 첫 프레임의 경우
            # KalmanTracker를 포함하여 초기화
            self.tracklets = {
                id: {'tracker': KalmanTracker(), 'centroid' : centroid[:2], 'age': 0} 
                for id, centroid in enumerate(centroids)
            }
            for (id, cluster), centroid in zip(self.tracklets.items(), centroids):
                cluster['tracker'].ekf.x[:2] = centroid[:2]  # 초기 위치 설정
                
        else:
            
            # Age 증가 및 오래된 클러스터 삭제
            for id in list(self.tracklets.keys()):
                if self.tracklets[id]['age'] > self.max_age:
                    self.dead_index.append(id)
            
            for id in self.dead_index:
                self.tracklets.pop(id, None)
            
            prev_centroids = []
            for track_data in self.tracklets.values():
                if 'tracker' in track_data and hasattr(track_data['tracker'], 'ekf'):
                    centroid = track_data['tracker'].ekf.x[:2]
                    if len(centroid) == 2:  # Ensure 2D data
                        prev_centroids.append(centroid)
            self.matcher.set_tracklet_index(list(self.tracklets.keys()))
            self.matcher.set_match()

            # 비용 행렬 계산 및 매칭
            self.calc_cost_matrix(prev_centroids, centroids)
            matched_indices, unmatched_indices, untracked_indices = self.associate(self.cost_matrix)

            # 매칭된 클러스터 업데이트
            self.living_track = {}
            for t_i, j in matched_indices:
                track_id = self.matcher.convert_table2tracklet(t_i)

                tracker = self.tracklets[track_id]['tracker']
                
                # Kalman Filter 업데이트
                tracker.predict()  # 상태 예측
                tracker.update(observation=centroids[j][:2])  # 관측값으로 업데이트
                
                # 상태 갱신
                self.tracklets[track_id]['age'] = 0
                self.living_track.update({track_id : {'centroid' : centroids[j][:2], 'tracker': tracker, 'age': 0}})


            # 업데이트되지 않은 클러스터의 age 증가
            for t_u in unmatched_indices:
                track_id = self.matcher.convert_table2tracklet(t_u)
                self.tracklets[track_id]['age'] += 1
                
            # 새로운 클러스터 추가
            next_id = max(self.tracklets.keys(), default=-1) + 1

            for u in untracked_indices:
                new_tracker = KalmanTracker()
                new_tracker.ekf.x[:2] = centroids[u][:2]  # 초기 상태 설정
                self.tracklets[next_id] = {'tracker': new_tracker,'centroid':centroids[u][:2], 'age': 0}
                self.living_track.update({next_id : {'centroid' : centroids[u][:2], 'tracker': tracker, 'age': 0}})

                next_id += 1

        # 이력 기록
        for id, cluster in self.tracklets.items():
            if id not in self.history:
                self.history[id] = []
            self.history[id].append(cluster['tracker'].ekf.x.copy())

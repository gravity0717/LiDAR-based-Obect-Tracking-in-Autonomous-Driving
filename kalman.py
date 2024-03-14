import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

class KalmanTracker:
    def __init__(self):
        self.ekf = ExtendedKalmanFilter(dim_x=5, dim_z=2) # state = [x,y,theta,v,w] v,w: anlge velocity
        self.ekf.x = np.array([0., 0., 0., 1., 0.1])  # 초기 위치, 각도, 선속도, 각속도
        # 초기 상태 및 코변서 설정
        self.ekf.P *= 10.  # 초기 추정 오차 공분산
        self.ekf.R = np.diag([0.1, 0.1])  # 측정 노이즈
        # 예측 및 업데이트 루프
        self.dt = 0.1  # 시간 간격
        self.l = 1.0  # 선속도 스케일 팩터
        ekf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.01, block_size=2, order_by_dim=False)
        self.history = []

    def fx(self, state, dt):
        x, y, theta, v, w = state.flatten()
        vt, wt = v * dt, w * dt
        s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)
        return np.array([
            [x + vt * c],
            [y + vt * s],
            [theta + wt],
            [v],
            [w]])

    def Fx(self, state, dt):
        x, y, theta, v, w = state.flatten()
        vt, wt = v * dt, w * dt
        s, c = np.sin(theta + wt / 2), np.cos(theta + wt / 2)
        return np.array([
            [1, 0, -vt * s, dt * c, -vt * dt * s / 2],
            [0, 1,  vt * c, dt * s,  vt * dt * c / 2],
            [0, 0,       1,      0,               dt],
            [0, 0,       0,      1,                0],
            [0, 0,       0,      0,                1]])

    def hx(self, state):
        x, y, *_ = state.flatten()
        return np.array([[x], [y]])

    def Hx(self, state):
        return np.eye(2, 5)

    def predict(self):
        self.ekf.F = self.Fx(self.ekf.x, self.dt)
        self.ekf.x = self.fx(self.ekf.x, self.dt)
        self.ekf.predict()
        
    def update(self):
        self.ekf.update(self.ekf.x, self.hx, self.Hx, self.ekf.R)
        self.history.append(self.ekf.x)
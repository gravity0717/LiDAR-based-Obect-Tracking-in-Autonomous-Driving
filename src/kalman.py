import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
# from filterpy.common import Q_discrete_white_noise

class KalmanTracker:
    def __init__(self):
        self.ekf = ExtendedKalmanFilter(dim_x=5, dim_z=2) # state = [x,y,theta,v,w] v,w: anlge velocity 
        self.ekf.x = np.array([0., 0., 0., 1., 0.1])  # 초기 위치, 각도, 선속도, 각속도
        self.ekf.P *= 5.  # 초기 추정 오차 공분산
        self.ekf.R = 1e-2 * np.eye(2)  # 측정 노이즈
        # 예측 및 업데이트 루프
        self.dt = 0.1  # 시간 간격
        self.l = 1.0  # 선속도 스케일 팩터
        self.ekf.Q = 1e-4 * np.eye(5) # Q_discrete_white_noise는 state dimension 4 까지만 가능 
        self.history = []

    def pcd2state(pcd:np.ndarray):
        x,y = pcd
        
    def state2pcd(state):
        pcd = self.ekf.x[:2]
    
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

    def Hx(self, state):
        # 측정 함수, 상태에서 x, y만 추출하여 반환
        x, y, *_ = state.flatten()
        return np.array([[x], [y]])  # 2D 측정 벡터로 반환

    def H_j(self, state):
        # 측정 함수의 Jacobian, 2x5 행렬로 반환
        return np.array([[1, 0, 0, 0, 0],  # x에 대한 Jacobian
                        [0, 1, 0, 0, 0]]) # y에 대한 Jacobian

    def predict(self):
        self.ekf.F = self.Fx(self.ekf.x, self.dt)
        self.ekf.x = self.fx(self.ekf.x, self.dt)
        self.ekf.predict()
        
    def update(self, observation):
        observation = self.Hx(observation) # dimension
        self.ekf.update(z=observation, HJacobian=self.H_j, Hx=self.Hx, R=self.ekf.R)
        self.history.append(self.ekf.x)
        
    def mahalanobis_distance(self, observation):
        H = self.H_j(self.ekf.x)  # 측정 함수의 Jacobian
        z_pred = self.Hx(self.ekf.x)  # 예상 측정값
        S = H @ self.ekf.P @ H.T + self.ekf.R  # 오차 공분산
        residual = observation - z_pred.flatten()  # 관측값과 예상값의 차이
        return np.sqrt(residual.T @ np.linalg.inv(S) @ residual)
    
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from filterpy.stats import plot_covariance_ellipse

    # KalmanTracker 인스턴스 생성
    tracker = KalmanTracker()

    # 테스트용 관측값 생성
    np.random.seed(42)  # 랜덤값 고정
    true_state = np.array([0., 0., 0., 1., 0.1])  # 초기 실제 상태
    observations = []

    # 시뮬레이션: 20번의 관측값 생성
    for _ in range(20):
        true_state = tracker.fx(true_state, tracker.dt).flatten()
        observation = true_state[:2] + np.random.normal(0, 0.1, size=2)  # x, y만 관측
        observations.append(observation)

    # 시각화 설정
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("EKF State Estimation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # 관측값 플롯
    observations = np.array(observations)
    ax.scatter(observations[:, 0], observations[:, 1], color='red', label='Observations')

    # EKF 업데이트 및 추정 시각화
    estimated_states = []
    for obs in observations:
        tracker.predict()
        tracker.update(obs)
        estimated_states.append(tracker.ekf.x[:2].flatten())
        # 공분산 타원 그리기
        plot_covariance_ellipse(
            mean=tracker.ekf.x[:2],
            cov=tracker.ekf.P[:2, :2],
            edgecolor='blue',
            alpha=0.3
        )

    # 추정된 상태 플롯
    estimated_states = np.array(estimated_states)
    ax.plot(estimated_states[:, 0], estimated_states[:, 1], color='blue', label='Estimated States')

    # 실제 경로 (True Path, Ground Truth)
    ax.plot(true_state[0], true_state[1], 'k*', label='True State', markersize=10)

    ax.legend()
    plt.grid()
    plt.show()
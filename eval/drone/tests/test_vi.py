import numpy as np
from eval.drone.vi import vi_scale, ate, path_length, so3_exp

def test_ate_zero_when_equal():
    P = np.cumsum(np.random.RandomState(0).randn(20,3), 0)
    assert ate(P, P.copy()) < 1e-9

def test_path_length():
    P = np.array([[0,0,0],[1,0,0],[1,1,0.]])
    assert abs(path_length(P) - 2.0) < 1e-9

def test_vi_scale_recovers_known_scale():
    # A kinematically self-consistent smooth trajectory: position, velocity and
    # acceleration all come from the same analytic p(t), so the visual-inertial
    # constraints are exactly satisfiable and vi_scale must recover s_true.
    rng = np.random.RandomState(1)
    s_true = 3.0; n = 12; T = 3.0; imu_hz = 400.0
    amp = rng.uniform(0.3, 0.7, 3); omega = rng.uniform(0.6, 1.4, 3)
    phase = rng.uniform(0, 2*np.pi, 3)
    def p(t):   t = np.asarray(t)[..., None]; return amp*np.sin(omega*t + phase)
    def acc(t): t = np.asarray(t)[..., None]; return -amp*omega**2*np.sin(omega*t + phase)
    kf_t = np.linspace(0, T, n)
    P_metric = p(kf_t)
    P_vis = P_metric / s_true                    # visual is up-to-scale (1/s)
    R_list = [np.eye(3)]*n                        # body aligned with c0
    imu_t = np.linspace(0, T, int(T*imu_hz)+1)
    g_grav = np.array([0., 0., -9.81])
    imu_acc = acc(imu_t) - g_grav                # specific force = a_true - gravity
    s = vi_scale(P_vis, R_list, kf_t, imu_t, imu_acc)
    assert abs(s - s_true) / s_true < 0.10       # recovers metric scale within 10%

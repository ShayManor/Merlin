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


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def test_vi_scale_survives_a_yawing_body():
    """Scale must not depend on how fast the drone is turning.

    The test above holds the body at identity, so it never exercised the attitude used to
    rotate the IMU samples. Holding the keyframe attitude constant across a whole interval
    mis-rotates every sample after its start, which corrupts gravity removal: at 180 deg/s
    over a 0.4 s interval that alone cost 25% of scale, and it is what made the sim C2
    control read 17.8% median error with a PERFECT trajectory.
    """
    s_true = 1.0
    kf_dt, n, imu_hz = 0.4, 9, 250.0
    kf_t = np.arange(n) * kf_dt
    imu_t = np.arange(0.0, kf_t[-1] + 1e-9, 1.0 / imu_hz)
    amp = np.array([1.2, 0.8, 0.35]); om = np.array([0.9, 1.4, 0.7])

    def p(t):   t = np.asarray(t)[..., None]; return amp * np.sin(om * t)
    def acc(t): t = np.asarray(t)[..., None]; return -amp * om**2 * np.sin(om * t)

    g_grav = np.array([0., 0., -9.81])
    for rate_deg in (0.0, 90.0, 180.0):
        w = np.deg2rad(rate_deg)
        # specific force in the BODY frame, from the true attitude at each sample
        f_body = np.array([_Rz(w * t).T @ (a - g_grav) for t, a in zip(imu_t, acc(imu_t))])
        R_list = [_Rz(w * t) for t in kf_t]
        s = vi_scale(p(kf_t) / s_true, R_list, kf_t, imu_t, f_body)
        assert abs(s - s_true) / s_true < 0.02, f"{rate_deg} deg/s gave s={s}"

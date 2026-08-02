"""Visual-inertial metric-scale recovery (C2) and ATE/drift helpers (C3).

vi_scale mirrors eval/vi_scale.py: rotate body accel into the c0 frame with the visual
keyframe rotations, form single/double integral constraints over consecutive keyframes,
and solve one least-squares for [per-kf velocity, gravity, SCALE]. ate/so3_exp are the
Umeyama-aligned RMSE and SO(3) exp from eval/c3_drift.py.
"""
import numpy as np

def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w/th
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

def path_length(P):
    P = np.asarray(P, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

def ate(est, gt):
    X, Y = np.asarray(est, float), np.asarray(gt, float)
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X-mx, Y-my
    U, S, Vt = np.linalg.svd(Yc.T@Xc); R = U@Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = U@Vt
    al = (R@Xc.T).T + my
    return float(np.sqrt(((al-Y)**2).sum(1).mean()))

def _integrals(imu_t, imu_acc, Rk, t0, t1):
    m = (imu_t >= t0) & (imu_t <= t1)
    if m.sum() < 2:
        return None
    ts = imu_t[m]; acc = (Rk @ imu_acc[m].T).T
    si = np.zeros(3); di = np.zeros(3); v = np.zeros(3)
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]; a = 0.5*(acc[i]+acc[i-1])
        di += v*dt + 0.5*a*dt*dt
        v += a*dt; si = v.copy()
    return si, di

def vi_scale(P_vis, R_list, kf_t, imu_t, imu_acc):
    P = np.asarray(P_vis, float); ts = np.asarray(kf_t, float)
    imu_t = np.asarray(imu_t, float); imu_acc = np.asarray(imu_acc, float)
    n = len(P); rows = []; rhs = []
    for k in range(n-1):
        dt = ts[k+1]-ts[k]
        ii = _integrals(imu_t, imu_acc, R_list[k], ts[k], ts[k+1])
        if ii is None:
            return float('nan')
        si, di = ii
        for d in range(3):
            row = np.zeros(3*n+4); row[3*k+d] = -1; row[3*(k+1)+d] = 1; row[3*n+d] = dt
            rows.append(row); rhs.append(si[d])
        for d in range(3):
            row = np.zeros(3*n+4); row[3*k+d] = -dt; row[3*n+d] = 0.5*dt*dt
            row[3*n+3] = (P[k+1]-P[k])[d]
            rows.append(row); rhs.append(di[d])
    x, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return float(x[-1])

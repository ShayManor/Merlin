"""Coordinate transforms between PX4 local NED and Habitat world (Y-up, right-handed).

hab = [E, U, -N] with NED = (N, E, D): x_hab=E=+y_ned, y_hab=U=-z_ned, z_hab=-N=-x_ned.
Camera in Habitat looks down its local -Z (habitat convention). See plan Task 12 for the
on-cluster render check that validates the forward axis before any scored run.
"""
import numpy as np

# rows map (N,E,D) -> (E, -D, -N)
R_NED2HAB = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [-1.0, 0.0, 0.0]])

def ned_to_hab_pos(p_ned):
    return R_NED2HAB @ np.asarray(p_ned, float)

def hab_to_ned_pos(p_hab):
    return R_NED2HAB.T @ np.asarray(p_hab, float)

def quat_to_R(q):  # q = [x,y,z,w] -> 3x3
    x, y, z, w = q
    n = (x*x+y*y+z*z+w*w) ** 0.5 + 1e-12
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

# FRD(body) is X-forward, Y-right, Z-down. A camera bolted facing forward looks along +X_frd,
# but the Habitat camera looks down local -Z. Fixed body->camera rotation aligns +X_frd -> -Z_cam.
_R_FRD2CAM = np.array([[0.0, 1.0, 0.0],   # cam_x(right) = frd_y(right)
                       [0.0, 0.0, -1.0],  # cam_y(up)    = -frd_z(down)
                       [-1.0, 0.0, 0.0]]) # cam_z(back)  = -frd_x(forward)

# Habitat's camera looks down its local -Z with +Y up; the pinhole back-projection in
# mapper.py uses the optical convention (+Z forward, +Y down). Multiplying by this converts
# a Habitat camera matrix into an optical one. Without it every mapped point is reflected
# through the camera -- placed behind it and upside down.
HAB_CAM_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


def ned_quat_to_hab_R(q_frd2ned):
    R_frd2ned = quat_to_R(q_frd2ned)
    # world rotation of the camera in habitat: hab_world <- ned <- frd <- cam
    R_cam2ned = R_frd2ned @ _R_FRD2CAM.T
    return R_NED2HAB @ R_cam2ned


def ned_traj_to_cam0(rel_neds, quats_frd2ned):
    """GT camera trajectory in the FIRST frame's camera frame.

    Returns (P_c0, R_list) in exactly the convention forward_window produces: positions in c0
    and rotations cam_k -> c0. Feeding this to vi_scale in place of a visual trajectory is the
    C2 control: the true scale is then 1.0 by construction, so the solver must return ~1.0. If
    it does not, the fault is in the IMU path and no pose estimator can fix it.
    """
    P = np.asarray(rel_neds, float)
    R_cam2ned = [quat_to_R(q) @ _R_FRD2CAM.T for q in quats_frd2ned]
    R0 = R_cam2ned[0]
    return (np.array([R0.T @ (p - P[0]) for p in P]), [R0.T @ R for R in R_cam2ned])


def frd_acc_to_cam(acc_frd):
    """Rotate a body-FRD vector (e.g. PX4 accelerometer) into the student's camera/optical
    frame, so the IMU is consistent with the cam_quats rotations vi_scale uses. Without this
    the accel and the visual rotations live in different body frames and scale recovery fails."""
    return _R_FRD2CAM @ np.asarray(acc_frd, float)


def Ry(angle):
    """Rotation about the Habitat up-axis (+Y) by `angle` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class EpisodeFrame:
    """Places the drone's local PX4-NED frame at an interior Habitat start pose.

    PX4 reports pose relative to its takeoff origin. An episode samples an interior start
    point `start_hab` (a navigable floor point raised to drone altitude) and a heading
    `yaw_hab` (rotation about Habitat +Y) so that the drone's +N_ned axis points toward the
    goal. World Habitat pose = start_hab + Ry(yaw_hab) @ (basis-change of the NED pose).
    Without this the PX4 origin maps to the raw scene origin, which is usually empty space.
    """

    def __init__(self, start_hab, yaw_hab):
        # yaw_hab is the desired Habitat azimuth (atan2(dx,-dz)) of the drone's +N_ned axis.
        # ned_to_hab_pos maps +N_ned -> -Z_hab (azimuth 0), and Ry(theta) rotates that to
        # azimuth -theta, so use Ry(-yaw_hab) to land +N_ned on azimuth yaw_hab.
        self.t = np.asarray(start_hab, float)
        self.Rz = Ry(-float(yaw_hab))
        # kept so a persisted episode can rebuild this exact frame offline
        self.start_hab = self.t
        self.yaw_hab = float(yaw_hab)

    def ned_to_hab_pos(self, pos_ned):
        return self.t + self.Rz @ ned_to_hab_pos(pos_ned)

    def hab_to_ned_pos(self, pos_hab):
        return hab_to_ned_pos(self.Rz.T @ (np.asarray(pos_hab, float) - self.t))

    def ned_quat_to_hab_R(self, q_frd2ned):
        return self.Rz @ ned_quat_to_hab_R(q_frd2ned)

#!/usr/bin/env python3
"""Minimal pymavlink OFFBOARD bridge to PX4 SITL (SIH). Reads GT pose + IMU, streams
position setpoints in NED. OFFBOARD needs a continuous setpoint stream, so callers must
send_position_ned() at >= 2 Hz including before arming.
"""
import os
import threading
import time

import numpy as np
from pymavlink import mavutil


class PX4Bridge:
    def __init__(self, url=None):
        # PX4 instance i publishes its offboard MAVLink on 14540+i (see run_px4.sh), so
        # concurrent jobs pick their instance up through MERLIN_PX4_INSTANCE.
        if url is None:
            url = "udpin:127.0.0.1:%d" % (14540 + int(os.environ.get("MERLIN_PX4_INSTANCE", 0)))
        self.m = mavutil.mavlink_connection(url)
        self._last = {"pos_ned": np.zeros(3), "quat_frd2ned": np.array([0, 0, 0, 1.0]),
                      "acc": np.zeros(3), "gyro": np.zeros(3), "t": 0.0}
        self._imu_buf = []   # (t, ax, ay, az) FRD, appended for every HIGHRES_IMU message
        self._lock = threading.Lock()   # guards the mavlink connection (streamer + reader)
        self._mode = None
        self._in_offboard = False

    MODE_OFFBOARD = 6           # PX4 custom_mode main-mode id

    def wait_ready(self, timeout=30):
        self.m.wait_heartbeat(timeout=timeout)
        # HIGHRES_IMU at ~250 Hz (4000 us) for dense integration; pose at 50 Hz.
        for msg, itv in (("LOCAL_POSITION_NED", 20000), ("ATTITUDE_QUATERNION", 20000),
                         ("HIGHRES_IMU", 4000)):
            self.m.mav.command_long_send(
                self.m.target_system, self.m.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                getattr(mavutil.mavlink, "MAVLINK_MSG_ID_" + msg), itv, 0, 0, 0, 0, 0)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._pump():
                return True
        return False

    def configure_sim(self):
        """Stop the simulated battery from ending episodes.

        The SIH battery drains across back-to-back episodes on one PX4 instance. Once it hits
        the low threshold PX4 raises a failsafe and flies RTL, which silently overrides
        OFFBOARD: the drone stops tracking setpoints ~30 s after takeoff and the episode is
        recorded as an ordinary timeout. Measured at 18 of 66 episodes, and because the drain
        is monotonic in episode order it also biases any sweep whose configs run in sequence.
        """
        for name, val in (("SIM_BAT_MIN_PCT", 100.0),   # keep the simulated cell full
                          ("COM_LOW_BAT_ACT", 0.0),     # low battery = warning only
                          ("COM_DISARM_LAND", -1.0)):   # no auto-disarm on land detect
            with self._lock:
                self.m.mav.param_set_send(
                    self.m.target_system, self.m.target_component, name.encode(),
                    float(val), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            time.sleep(0.05)
        self._pump()
        return True

    def in_failsafe(self):
        """True if PX4 has taken control away from OFFBOARD (failsafe / RTL / land).

        Only meaningful once OFFBOARD has actually been observed for this episode, so the
        mode-change lag right after set_mode does not read as a takeover.
        """
        return bool(self._in_offboard and self._mode is not None
                    and self._mode != self.MODE_OFFBOARD)

    def _pump(self):
        got = False
        while True:
            with self._lock:
                msg = self.m.recv_match(blocking=False)
            if msg is None:
                break
            t = msg.get_type()
            if t == "LOCAL_POSITION_NED":
                self._last["pos_ned"] = np.array([msg.x, msg.y, msg.z]); got = True
            elif t == "ATTITUDE_QUATERNION":
                self._last["quat_frd2ned"] = np.array([msg.q2, msg.q3, msg.q4, msg.q1])  # xyzw
            elif t == "HEARTBEAT":
                # PX4 hands control to a failsafe mode (RTL/LAND) without telling the
                # offboard client; the setpoints keep streaming and are simply ignored, so
                # the episode looks like an ordinary timeout unless this is checked.
                # PX4 packs the main mode in bits 16-23; OFFBOARD is 6. Rather than enumerate
                # failsafe modes, treat "we were in OFFBOARD and no longer are" as takeover.
                self._mode = (getattr(msg, "custom_mode", 0) >> 16) & 0xFF
                if self._mode == self.MODE_OFFBOARD:
                    self._in_offboard = True
            elif t == "HIGHRES_IMU":
                self._last["acc"] = np.array([msg.xacc, msg.yacc, msg.zacc])
                self._last["gyro"] = np.array([msg.xgyro, msg.ygyro, msg.zgyro])
                self._last["t"] = msg.time_usec / 1e6
                self._imu_buf.append([msg.time_usec / 1e6, msg.xacc, msg.yacc, msg.zacc])
        return got

    def get_state(self):
        self._pump()
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._last.items()}

    def drain_imu(self):
        """Return and clear the accumulated full-rate IMU samples (M,4) [t, ax, ay, az] FRD."""
        self._pump()
        buf = np.array(self._imu_buf) if self._imu_buf else np.zeros((0, 4))
        self._imu_buf = []
        return buf

    def send_position_ned(self, x, y, z, yaw=0.0):
        with self._lock:
            self.m.mav.set_position_target_local_ned_send(
                0, self.m.target_system, self.m.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000_10_1_111_111_000,   # position + yaw
                x, y, z, 0, 0, 0, 0, 0, 0, yaw, 0)

    def start_streaming(self, x, y, z, yaw=0.0, hz=20.0):
        """Continuously stream the current target setpoint from a background thread so
        OFFBOARD stays alive even when the perception/planning loop ticks slower than the
        ~2 Hz OFFBOARD deadline. Update the target with set_target()."""
        self._stream_target = [float(x), float(y), float(z), float(yaw)]
        self._stream_stop = threading.Event()

        def _run(dt=1.0 / hz):
            while not self._stream_stop.is_set():
                t = self._stream_target
                self.send_position_ned(t[0], t[1], t[2], t[3])
                time.sleep(dt)

        self._stream_thread = threading.Thread(target=_run, daemon=True)
        self._stream_thread.start()

    def set_target(self, x, y, z, yaw=0.0):
        self._stream_target = [float(x), float(y), float(z), float(yaw)]

    def stop_streaming(self):
        if getattr(self, "_stream_stop", None) is not None:
            self._stream_stop.set()
            self._stream_thread.join(timeout=1.0)
            self._stream_stop = None

    def set_speed_limit(self, v):
        """Cap the multicopter velocity so the ablation's speed axis is real. Without this the
        drone's speed is set only by the position controller chasing the planner's carrot."""
        for name in ("MPC_XY_VEL_MAX", "MPC_Z_VEL_MAX_UP", "MPC_Z_VEL_MAX_DN"):
            with self._lock:
                self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                          name.encode(), float(v),
                                          mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            time.sleep(0.05)

    def set_mode_offboard_and_arm(self):
        self._in_offboard = False          # re-arm the takeover check for this episode
        for _ in range(20):
            self.send_position_ned(0, 0, -1.5); time.sleep(0.05)
        self.m.set_mode("OFFBOARD")
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)

    def takeoff(self, alt=1.5):
        t0 = time.time()
        while time.time() - t0 < 8:
            self.send_position_ned(0, 0, -abs(alt)); self._pump(); time.sleep(0.05)

    def close(self):
        self.m.close()

"""nav_loop must FAIL an episode on contact. Fakes stand in for PX4 and the renderer so
this runs locally without pymavlink or habitat-sim."""
import numpy as np
from eval.drone.frames import EpisodeFrame
from eval.drone.nav_loop import run_episode

class FakeBridge:
    def __init__(self):
        self.p = np.zeros(3); self.t = 0.0
    def set_speed_limit(self, v): pass
    def set_mode_offboard_and_arm(self): pass
    def takeoff(self, alt): pass
    def start_streaming(self, *a, **k): pass
    def stop_streaming(self): pass
    def set_target(self, x, y, z, yaw=0.0): self.p = np.array([x, y, z], float)
    def get_state(self):
        self.t += 0.05
        return {"pos_ned": self.p.copy(), "quat_frd2ned": np.array([0, 0, 0, 1.0]),
                "acc": np.zeros(3), "gyro": np.zeros(3), "t": self.t}

class FakeClient:
    def __init__(self, hit_after=3):
        self.n = 0; self.hit_after = hit_after
    def render(self, pos, R):
        return {"rgb": np.zeros((14, 14, 3), np.uint8),
                "depth": np.full((14, 14), 5.0, np.float32),
                "K": np.array([[7.0, 0, 7.0], [0, 7.0, 7.0], [0, 0, 1.0]])}
    def collide(self, pos, radius):
        self.n += 1
        hit = self.n >= self.hit_after
        return {"hit": hit, "dist": 0.05 if hit else 3.0,
                "object_id": 151 if hit else None, "is_novel": hit}

def _cfg():
    # res 14 is a synthetic tiny frame, not a Jetson operating point, so drive the keyframe
    # cadence with an explicit latency instead of the LATENCY_MS lookup
    return {"res": 14, "latency_ms": 147.0, "timeout": 5.0, "alt": 0.75, "voxel": 0.15,
            "step": 0.3, "safety": 0.35, "goal_tol": 0.5, "wp_tol": 0.7, "max_range": 8.0,
            "max_speed": 1.0, "ctrl_dt": 0.0}

def test_collision_terminates_episode_as_failure():
    ep = EpisodeFrame([0.0, 0.75, 0.0], 0.0)
    m = run_episode(FakeBridge(), FakeClient(hit_after=3),
                    lambda out, r=None: out["depth"], ep, [[0, 0, 0], [8.0, 0, 0]], _cfg())
    assert m["ended"] == "collision"
    assert m["success"] == 0.0
    assert m["spl"] == 0.0

def test_collision_events_are_recorded_with_attribution():
    ep = EpisodeFrame([0.0, 0.75, 0.0], 0.0)
    m = run_episode(FakeBridge(), FakeClient(hit_after=2),
                    lambda out, r=None: out["depth"], ep, [[0, 0, 0], [8.0, 0, 0]], _cfg())
    assert len(m["collision_events"]) >= 1
    e = m["collision_events"][0]
    assert e["object_id"] == 151 and e["is_novel"] is True

def test_keyframe_log_is_retained():
    ep = EpisodeFrame([0.0, 0.75, 0.0], 0.0)
    m = run_episode(FakeBridge(), FakeClient(hit_after=999),
                    lambda out, r=None: out["depth"], ep, [[0, 0, 0], [8.0, 0, 0]], _cfg())
    assert len(m["log"]) > 0
    assert {"t", "pos_ned", "quat_frd2ned"} <= set(m["log"][0])

#!/usr/bin/env python3
"""Habitat-sim render/physics server (runs in the conda `habenv`).

Kept in its own process+env so habitat-sim's conda torch/numpy never collide with the
MERLIN training env (the student runs in the client). Protocol: length-prefixed pickle
over a localhost TCP socket. The client sends control dicts; the server returns
observations (RGB + GT depth) plus navmesh collision and geodesic-to-goal.

Continuous kinematic robot on the scene navmesh: a velocity command (forward speed,
yaw rate, dt) is integrated to a desired pose; pathfinder.try_step slides it along the
navmesh (no wall penetration). A collision is logged when the slid end differs from the
desired end -- the standard Habitat PointNav collision model. Geodesic guide via
ShortestPath supplies the goal bearing; the client's local planner can veto.
"""
import argparse
import pickle
import socket
import struct
import sys

import numpy as np

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector


def make_cfg(scene, hfov, width, height, sensor_height, dataset=None):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene
    if dataset:
        sim_cfg.scene_dataset_config_file = dataset
    sim_cfg.enable_physics = False
    specs = []
    for uuid, stype in (("rgba", habitat_sim.SensorType.COLOR),
                        ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [height, width]
        s.position = [0.0, sensor_height, 0.0]
        s.hfov = hfov
        specs.append(s)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


class Server:
    def __init__(self, hfov=90.0, width=504, height=504, sensor_height=0.88):
        self.hfov = hfov; self.width = width; self.height = height
        self.sensor_height = sensor_height
        self.sim = None; self.scene = None
        self.pos = None; self.yaw = 0.0
        self.goal = None; self.collisions = 0; self.path_len = 0.0
        self.dataset = None; self.guide = "geodesic"

    def _ensure_sim(self, scene, dataset=None):
        key = (scene, dataset)
        if self.scene == key and self.sim is not None:
            return
        if self.sim is not None:
            self.sim.close()
        cfg = make_cfg(scene, self.hfov, self.width, self.height, self.sensor_height, dataset)
        self.sim = habitat_sim.Simulator(cfg)
        if not self.sim.pathfinder.is_loaded:
            ns = habitat_sim.NavMeshSettings()
            ns.set_defaults()
            ns.agent_radius = 0.18
            ns.agent_height = 1.0
            self.sim.recompute_navmesh(self.sim.pathfinder, ns)
        self.scene = key

    def _set_agent(self):
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        st.position = np.asarray(self.pos, dtype=np.float32)
        st.rotation = quat_from_angle_axis(self.yaw, np.array([0.0, 1.0, 0.0]))
        agent.set_state(st)

    def _obs(self):
        o = self.sim.get_sensor_observations()
        rgb = np.asarray(o["rgba"])[:, :, :3].copy()
        depth = np.asarray(o["depth"]).astype(np.float32).copy()
        geo, bearing = self._geo_and_bearing()
        return {"rgb": rgb, "depth": depth, "pos": list(map(float, self.pos)),
                "yaw": float(self.yaw), "geo_dist": geo, "goal_bearing": bearing,
                "collisions": self.collisions, "path_len": self.path_len,
                "hfov": self.hfov}

    def _geo_and_bearing(self):
        if self.goal is None:
            return float("nan"), 0.0
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(self.pos, dtype=np.float32)
        sp.requested_end = np.asarray(self.goal, dtype=np.float32)
        ok = self.sim.pathfinder.find_path(sp)
        geo = float(sp.geodesic_distance) if ok else float("inf")
        # geodesic guide -> next navmesh waypoint (routes around obstacles, perception
        # rarely matters). straight guide -> head at the final goal, robot must perceive
        # and avoid furniture itself (perception on the critical path). geo is always the
        # geodesic distance for success/SPL regardless of guide.
        if self.guide == "straight" or not ok or len(sp.points) < 2:
            wp = np.asarray(self.goal, dtype=np.float32)
        else:
            wp = np.asarray(sp.points[1], dtype=np.float32)
        d = wp - np.asarray(self.pos, dtype=np.float32)
        # forward(yaw) = R_y(yaw)*[0,0,-1] = [-sin yaw, 0, -cos yaw], so the world azimuth
        # consistent with our yaw is atan2(-dx, -dz). bearing = wrap(world_az - yaw).
        world_ang = np.arctan2(-d[0], -d[2])
        bearing = float(np.arctan2(np.sin(world_ang - self.yaw), np.cos(world_ang - self.yaw)))
        return geo, bearing

    # ---- commands ----
    def reset(self, scene, start, yaw, goal, dataset=None, guide="geodesic"):
        self._ensure_sim(scene, dataset)
        self.guide = guide
        self.pos = np.asarray(start, dtype=np.float32)
        self.yaw = float(yaw); self.goal = np.asarray(goal, dtype=np.float32)
        self.collisions = 0; self.path_len = 0.0
        self._set_agent()
        return self._obs()

    def step(self, v, yaw_rate, dt):
        self.yaw += yaw_rate * dt
        q = quat_from_angle_axis(self.yaw, np.array([0.0, 1.0, 0.0]))
        fwd = quat_rotate_vector(q, np.array([0.0, 0.0, -1.0]))
        desired = self.pos + fwd * (v * dt)
        end = np.asarray(self.sim.pathfinder.try_step(self.pos, desired), dtype=np.float32)
        moved = float(np.linalg.norm(end - self.pos))
        # collision = the robot could not reach where it aimed (navmesh blocked/slid it).
        # measured as the gap between the intended and achieved point -- this catches
        # glancing/sliding contacts that a displacement-magnitude check misses (sliding
        # preserves magnitude). gap is in metres; >5 cm off-target = a contact event.
        gap = float(np.linalg.norm(desired - end))
        if gap > 0.05:
            self.collisions += 1
        self.path_len += moved
        self.pos = end
        self._set_agent()
        return self._obs()

    def sample_episode(self, scene, min_geo, max_geo, seed, dataset=None):
        self._ensure_sim(scene, dataset)
        pf = self.sim.pathfinder
        pf.seed(int(seed))
        for _ in range(200):
            s = pf.get_random_navigable_point()
            g = pf.get_random_navigable_point()
            if not (pf.is_navigable(s) and pf.is_navigable(g)):
                continue
            sp = habitat_sim.ShortestPath()
            sp.requested_start = s; sp.requested_end = g
            if not pf.find_path(sp):
                continue
            geo = sp.geodesic_distance
            if not np.isfinite(geo) or geo < min_geo or geo > max_geo:
                continue
            eu = float(np.linalg.norm(np.asarray(s) - np.asarray(g)))
            if eu < 1e-3 or geo / eu > 1.8:   # avoid degenerate / overly twisty
                continue
            # face roughly toward goal
            d = np.asarray(g) - np.asarray(s)
            yaw = float(np.arctan2(d[0], -d[2]))
            return {"scene": scene, "start": list(map(float, s)), "yaw": yaw,
                    "goal": list(map(float, g)), "geo": float(geo)}
        return None

    def handle(self, msg):
        c = msg["cmd"]
        if c == "reset":
            return self.reset(msg["scene"], msg["start"], msg["yaw"], msg["goal"],
                              msg.get("dataset"), msg.get("guide", "geodesic"))
        if c == "step":
            return self.step(msg["v"], msg["yaw_rate"], msg["dt"])
        if c == "sample":
            return self.sample_episode(msg["scene"], msg["min_geo"], msg["max_geo"],
                                       msg["seed"], msg.get("dataset"))
        if c == "ping":
            return {"ok": True}
        raise ValueError(c)


def _recv(conn, n):
    buf = b""
    while len(buf) < n:
        b = conn.recv(n - len(buf))
        if not b:
            return None
        buf += b
    return buf


def serve(port):
    srv = Server()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port)); s.listen(1)
    print(f"HABSERVER_READY port={port}", flush=True)
    conn, _ = s.accept()
    while True:
        hdr = _recv(conn, 4)
        if hdr is None:
            break
        n = struct.unpack(">I", hdr)[0]
        msg = pickle.loads(_recv(conn, n))
        if msg.get("cmd") == "close":
            break
        try:
            rep = srv.handle(msg)
        except Exception as e:
            rep = {"error": repr(e)}
        out = pickle.dumps(rep)
        conn.sendall(struct.pack(">I", len(out)) + out)
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()
    serve(args.port)

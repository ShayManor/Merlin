#!/usr/bin/env python3
"""Client transport to the drone Habitat renderer (length-prefixed pickle over a
localhost socket). Trusted loopback channel between two of our own processes; payloads
are numpy RGB/depth arrays, so pickle (not JSON) is used deliberately. Same wire format
as eval/sim/sim_client.py, with render/collide replacing the navmesh step."""
import pickle
import socket
import struct
import time


class SimClient:
    def __init__(self, port, host="127.0.0.1", timeout=45.0):
        for _ in range(120):
            try:
                self.s = socket.create_connection((host, port), timeout=timeout)
                break
            except OSError:
                time.sleep(1.0)
        else:
            raise RuntimeError(f"could not connect to drone renderer on {port}")
        self.s.settimeout(timeout)

    def _rpc(self, msg):
        out = pickle.dumps(msg)
        self.s.sendall(struct.pack(">I", len(out)) + out)
        hdr = self._recv(4)
        n = struct.unpack(">I", hdr)[0]
        rep = pickle.loads(self._recv(n))
        if isinstance(rep, dict) and "error" in rep:
            raise RuntimeError(f"renderer error: {rep['error']}")
        return rep

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            b = self.s.recv(n - len(buf))
            if not b:
                raise RuntimeError("renderer closed connection")
            buf += b
        return buf

    def reset(self, scene, dataset=None, nav_radius=None):
        return self._rpc({"cmd": "reset", "scene": scene, "dataset": dataset,
                          "nav_radius": nav_radius})

    def sample(self, scene, min_geo, max_geo, seed, alt=1.0, dataset=None, stale=False,
               route_clear=0.0):
        return self._rpc({"cmd": "sample", "scene": scene, "min_geo": min_geo,
                          "max_geo": max_geo, "seed": seed, "alt": alt, "dataset": dataset,
                          "stale": stale, "route_clear": route_clear})

    def render(self, pos, R):
        return self._rpc({"cmd": "render", "pos": pos, "R": R})

    def collide(self, pos, radius=0.25, max_dist=5.0):
        return self._rpc({"cmd": "collide", "pos": pos, "radius": radius, "max_dist": max_dist})

    def place_obstacles(self, waypoints, n, seed, alt=0.75, intrude=1.0, min_width=0.0,
                        obs_clear=0.0, block_margin=0.20, jitter=0.35, include=None,
                        yaw=0.0):
        return self._rpc({"cmd": "place_obstacles", "waypoints": waypoints, "n": n,
                          "seed": seed, "alt": alt, "intrude": intrude,
                          "min_width": min_width, "include": include,
                          "obs_clear": obs_clear, "yaw": yaw,
                          "block_margin": block_margin, "jitter": jitter})

    def clear_obstacles(self):
        return self._rpc({"cmd": "clear_obstacles"})

    def route_open(self, start, goal):
        return self._rpc({"cmd": "route_open", "start": start, "goal": goal})

    def topdown(self, mpp=0.05, height=None):
        return self._rpc({"cmd": "topdown", "mpp": mpp, "height": height})

    def ping(self):
        return self._rpc({"cmd": "ping"})

    def close(self):
        try:
            out = pickle.dumps({"cmd": "close"})
            self.s.sendall(struct.pack(">I", len(out)) + out)
            self.s.close()
        except OSError:
            pass

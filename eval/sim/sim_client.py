#!/usr/bin/env python3
"""Client transport to the Habitat render server (length-prefixed pickle over a
localhost socket). Trusted loopback channel between two of our own processes; payloads
are numpy RGB/depth arrays, so pickle (not JSON) is used deliberately."""
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
            raise RuntimeError(f"could not connect to sim server on {port}")
        self.s.settimeout(timeout)

    def _rpc(self, msg):
        out = pickle.dumps(msg)
        self.s.sendall(struct.pack(">I", len(out)) + out)
        hdr = self._recv(4)
        n = struct.unpack(">I", hdr)[0]
        rep = pickle.loads(self._recv(n))
        if isinstance(rep, dict) and "error" in rep:
            raise RuntimeError(f"sim server error: {rep['error']}")
        return rep

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            b = self.s.recv(n - len(buf))
            if not b:
                raise RuntimeError("sim server closed connection")
            buf += b
        return buf

    def sample(self, scene, min_geo, max_geo, seed, dataset=None):
        return self._rpc({"cmd": "sample", "scene": scene, "min_geo": min_geo,
                          "max_geo": max_geo, "seed": seed, "dataset": dataset})

    def reset(self, scene, start, yaw, goal, dataset=None, guide="geodesic", no_slide=False):
        return self._rpc({"cmd": "reset", "scene": scene, "start": start,
                          "yaw": yaw, "goal": goal, "dataset": dataset, "guide": guide,
                          "no_slide": no_slide})

    def step(self, v, yaw_rate, dt):
        return self._rpc({"cmd": "step", "v": v, "yaw_rate": yaw_rate, "dt": dt})

    def close(self):
        try:
            out = pickle.dumps({"cmd": "close"})
            self.s.sendall(struct.pack(">I", len(out)) + out)
            self.s.close()
        except OSError:
            pass

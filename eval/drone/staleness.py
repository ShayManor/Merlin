"""Jetson operating-point keyframe scheduler. The model produces a new depth only every
LATENCY_MS[res] seconds; between keyframes the planner reuses the last map (it goes stale as
the drone moves). Operating points are the measured Nano bf16 numbers from eval/sim/perception.py.
"""
LATENCY_MS = {252: 84.0, 378: 147.0, 518: 189.0}
WATTS = {252: 7.4, 378: 8.5, 518: 8.8}
RES_LIST = [252, 378, 518]

class Staleness:
    def __init__(self, res=378, latency_ms=None):
        # latency_ms overrides the measured operating point, so the ablation can sweep map
        # staleness independently of resolution (the three Jetson points are all fast enough
        # that the held-map distance stays far below the planner's safety margin).
        self.dt = (LATENCY_MS[res] if latency_ms is None else float(latency_ms)) / 1000.0
        self.last = None

    def due(self, sim_t):
        return self.last is None or (sim_t - self.last) >= self.dt

    def mark(self, sim_t):
        self.last = float(sim_t)

    def age(self, sim_t):
        return 0.0 if self.last is None else float(sim_t - self.last)

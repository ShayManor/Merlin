from eval.drone.staleness import Staleness, LATENCY_MS

def test_keyframe_cadence_matches_latency():
    s = Staleness(res=378)                 # 147 ms/frame -> ~6.8 Hz
    assert s.due(0.0)                      # first frame always due
    s.mark(0.0)
    assert not s.due(0.05)                 # 50 ms later, not yet
    assert s.due(0.20)                     # >147 ms later, due again
    s.mark(0.20)
    assert abs(s.age(0.35) - 0.15) < 1e-6

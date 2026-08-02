import numpy as np
from eval.drone.metrics import spl, boot_ci, aggregate

def test_spl():
    assert spl(True, 5.0, 5.0) == 1.0
    assert abs(spl(True, 5.0, 10.0) - 0.5) < 1e-9
    assert spl(False, 5.0, 5.0) == 0.0

def test_boot_ci_bounds_mean():
    m, lo, hi = boot_ci([1.0,2,3,4,5], seed=0)
    assert lo <= m <= hi

def test_aggregate_groups_by_arm():
    rows = [{"arm":"a","success":1.0},{"arm":"a","success":0.0},{"arm":"b","success":1.0}]
    out = aggregate(rows, ["success"])
    assert out["a"]["n"] == 2 and out["b"]["n"] == 1
    assert 0.0 <= out["a"]["success"]["mean"] <= 1.0

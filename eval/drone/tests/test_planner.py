import numpy as np
from eval.drone.mapper import VoxelMap
from eval.drone.planner import Planner3D

def test_straight_line_when_clear():
    vmap = VoxelMap(voxel=0.2)
    pl = Planner3D(step=0.4, safety=0.5, require_known=False)
    cur = np.array([0.,0.,0.]); goal = np.array([4.,0.,0.])
    tgt, blocked = pl.next_setpoint(vmap, cur, goal)
    assert not blocked
    assert tgt[0] > cur[0]                     # moves toward goal
    assert np.linalg.norm(tgt - cur) <= 0.4 + 1e-6

def test_detours_around_wall():
    vmap = VoxelMap(voxel=0.2, min_hits=1)
    # a wall of voxels at x=2 spanning y in [-1,1], z in [-1,1]
    for y in np.arange(-1,1,0.2):
        for z in np.arange(-1,1,0.2):
            vmap._hits[vmap._key([2.0,y,z])] = 5
    pl = Planner3D(step=0.4, safety=0.5, require_known=False)
    # start within one step+safety of the wall so the straight path is unsafe and a detour is forced
    cur = np.array([1.4,0.,0.]); goal = np.array([4.,0.,0.])
    tgt, blocked = pl.next_setpoint(vmap, cur, goal)
    # must not head straight into the wall: chosen target should have off-axis component
    assert abs(tgt[1]) > 1e-3 or abs(tgt[2]) > 1e-3 or blocked

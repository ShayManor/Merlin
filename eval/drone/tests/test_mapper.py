import numpy as np
from eval.drone.mapper import VoxelMap, backproject

def _K(w=64, h=48, hfov_deg=90.0):
    f = (w/2) / np.tan(np.deg2rad(hfov_deg)/2)
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1.0]])

def test_backproject_center_ray():
    K = _K()
    d = np.full((48,64), 4.0, np.float32)
    pts = backproject(d, K, max_range=8.0)          # camera frame, +Z forward (optical)
    # center pixel at depth 4 -> point ~ (0,0,4)
    c = pts[int(np.argmin(np.linalg.norm(pts[:, :2], axis=1)))]  # point nearest optical axis = center ray
    assert abs(np.linalg.norm(c) - 4.0) < 0.2

def test_integrate_and_query():
    K = _K()
    d = np.full((48,64), 4.0, np.float32)
    m = VoxelMap(voxel=0.2, min_hits=1)   # single synthetic frame: 1 hit per axis-aligned wall voxel
    # camera at habitat origin looking down -Z_hab: optical +Z maps to world -Z
    R = np.array([[1,0,0],[0,1,0],[0,0,-1.0]])       # flip optical Z to world -Z
    m.integrate(d, K, R, np.zeros(3), max_range=8.0)
    assert m.occupied_points().shape[0] > 100
    # a wall ~4 m in front (world z=-4) should be occupied, empty space at z=-1 should not
    assert m.is_occupied(np.array([0.0, 0.0, -4.0]))
    assert not m.is_occupied(np.array([0.0, 0.0, -1.0]))
    assert m.nearest_obstacle_dist(np.array([0.0,0.0,-2.0])) < 2.5

def test_any_occupied_within_matches_brute_force():
    # the planner's clearance test runs on this; it must agree exactly with a distance scan
    rng = np.random.default_rng(0)
    m = VoxelMap(voxel=0.15, min_hits=1)
    for p in rng.uniform(-2, 2, (500, 3)):
        m._hits[m._key(p)] = 3
    pts = m.occupied_points()
    q = rng.uniform(-3, 3, (200, 3))
    for radius in (0.2, 0.35, 0.9):
        fast = m.any_occupied_within(q, radius)
        brute = np.array([np.linalg.norm(pts - p, axis=1).min() <= radius for p in q])
        assert (fast == brute).all()

def test_one_vote_per_frame_not_per_point():
    # A single depth map lands thousands of points on one wall voxel. Counting them as
    # separate evidence made any voxel one frame touched permanently occupied and left
    # free-space carving unable to outvote it -- the map became the union of every frame's
    # depth noise. One frame must contribute at most one count.
    K = _K()
    d = np.full((48, 64), 4.0, np.float32)
    m = VoxelMap(voxel=0.2, min_hits=1)
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1.0]])
    m.integrate(d, K, R, np.zeros(3), max_range=8.0)
    assert max(m._hits.values()) == 1

def test_carving_retracts_a_phantom_seen_once():
    # A phantom surface reported by one frame and seen through by later frames must be
    # retracted: that is the whole discriminator between real geometry and depth noise.
    K = _K()
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1.0]])
    m = VoxelMap(voxel=0.2, min_hits=1)
    m.integrate(np.full((48, 64), 2.0, np.float32), K, R, np.zeros(3), max_range=8.0)
    assert m.is_occupied(np.array([0.0, 0.0, -2.0]))
    for _ in range(2):
        m.integrate(np.full((48, 64), 4.0, np.float32), K, R, np.zeros(3), max_range=8.0)
    assert not m.is_occupied(np.array([0.0, 0.0, -2.0]))
    assert m.is_occupied(np.array([0.0, 0.0, -4.0]))

def test_any_occupied_within_empty_map():
    m = VoxelMap(voxel=0.15)
    assert not m.any_occupied_within(np.zeros((4, 3)), 0.5).any()

import json
import os

from eval.drone import pair_runs as pr


def _ep(d, fname, scene, seed, arm, success, cls="cloth_04", ended="goal"):
    os.makedirs(f"{d}/episode_logs", exist_ok=True)
    json.dump({"scene": scene, "seed": seed, "arm": arm, "success": success,
               "ended": ended,
               "obstacles": ([{"handle": f"/data/{cls}.object_config.json"}] if cls else [])},
              open(f"{d}/episode_logs/{fname}", "w"))


def test_pairs_on_seed_from_body_not_filename(tmp_path):
    """Two runs on disjoint seeds write IDENTICAL filenames, so filename-keyed pairing
    would silently match unrelated episodes."""
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _ep(a, "apt_0_1016_student_n1.json", "apt_0", 5016, "student", 1)
    _ep(b, "apt_0_1016_student_n1.json", "apt_0", 6016, "student", 0)
    assert pr.pair(pr.load(a)[0], pr.load(b)[0]) is None   # no shared (scene, seed)

    _ep(b, "apt_0_1020_student_n1.json", "apt_0", 5016, "student", 0)
    p = pr.pair(pr.load(a)[0], pr.load(b)[0])
    assert p["n"] == 1 and p["delta"] == 1.0 and p["won"] == 1


def test_voided_episodes_are_dropped_and_counted(tmp_path):
    d = str(tmp_path / "v")
    _ep(d, "e0.json", "apt_0", 1, "student", 1)
    _ep(d, "e1.json", "apt_0", 2, "student", 0, ended="px4_frozen")
    _ep(d, "e2.json", "apt_0", 3, "student", 0, ended="px4_failsafe")
    rows, voided = pr.load(d)
    assert len(rows) == 1 and voided == 2


def test_only_the_requested_arm_is_loaded(tmp_path):
    d = str(tmp_path / "arms")
    _ep(d, "s.json", "apt_0", 1, "student", 1)
    _ep(d, "g.json", "apt_0", 1, "gt", 1)
    assert len(pr.load(d, "student")[0]) == 1
    assert len(pr.load(d, "gt")[0]) == 1


def test_balanced_mean_is_mean_of_class_means(tmp_path):
    """A class that placed more often must not dominate the headline number."""
    d = str(tmp_path / "bal")
    for i in range(9):                      # 9 episodes, all successful
        _ep(d, f"a{i}.json", "apt_0", i, "student", 1, cls="cloth_04")
    _ep(d, "b0.json", "apt_1", 100, "student", 0, cls="umbrella")
    per, bal = pr.summarise(pr.load(d)[0])
    assert per["cloth_04"] == (1.0, 9) and per["umbrella"] == (0.0, 1)
    assert abs(bal - 0.5) < 1e-9            # pooled would be 0.9


def test_obstacle_class_from_handle():
    assert pr.obstacle_class({"obstacles": [{"handle": "/d/cloth_04.object_config.json"}]}) == "cloth_04"
    assert pr.obstacle_class({"obstacles": []}) == ""

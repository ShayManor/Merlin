"""Write apt_<n>_ceil scene instances: the apt_<n> furniture in the Stage_v3_sc0_staging shell.

The frl_apartment_stage that apt_0..5 use has no ceiling (~15% of flight pixels render with no
surface: black RGB, GT depth 0), while the v3 staging stages are the same apartment shell with a
ceiling. Flying the held-out apt layouts in the v3 shell removes that render artifact; the
near-field training scenes already share the shell, so nothing new leaks into training.
usage: make_ceiling_scenes.py DATASET_DIR [n ...]   (DATASET_DIR holds configs/scenes)"""
import json
import os
import sys

scenes = os.path.join(sys.argv[1], "configs", "scenes")
stage = json.load(open(os.path.join(scenes, "v3_sc0_staging_00.scene_instance.json")))["stage_instance"]
for n in sys.argv[2:] or ["0", "3", "5"]:
    d = json.load(open(os.path.join(scenes, f"apt_{n}.scene_instance.json")))
    d["stage_instance"] = stage
    json.dump(d, open(os.path.join(scenes, f"apt_{n}_ceil.scene_instance.json"), "w"), indent=1)
    print(f"apt_{n}_ceil")

#!/usr/bin/env python3
"""Push a distilled MERLIN student checkpoint to a public HF model repo.

HF-native deployment bridge: the A40 produces the checkpoint, this uploads it,
and the Jetson pulls it (no pod<->Jetson direct route). Reads HF_TOKEN from env.

Usage:
  HF_TOKEN=... python tools/upload_hf.py --ckpt /workspace/ckpt/student_baseline.pt \
      --repo ShayManor/merlin-mapanything-student --path-in-repo student_baseline.pt
"""
import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--path-in-repo", default=None)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    assert token, "set HF_TOKEN"
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    name = args.path_in_repo or os.path.basename(args.ckpt)
    api.upload_file(path_or_fileobj=args.ckpt, path_in_repo=name,
                    repo_id=args.repo, repo_type="model")
    print(f"[uploaded] {args.repo}/{name}", flush=True)


if __name__ == "__main__":
    main()

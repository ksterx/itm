"""Bulk-extract MediaPipe Face Landmarker features for AMI meetings.

For each meeting, picks the two most-active speakers (matching the
audio pipeline in ``itm.data.dataset._pick_two_most_active``), maps
each speaker letter to its Closeup camera (A→1, B→2, C→3, D→4 per
AMI corpusResources/meetings.xml), and runs the feature extractor on
the matching ``Closeup{N}.avi``.

Output: ``data/processed/visual/<meeting>/<speaker>.npy`` per speaker
(shape ``(T_video, 56)``, 25 fps, blendshapes + Euler + mouth open).

Usage::

    uv run python scripts/build_visual_dataset.py \\
        --meetings ES2002a ES2002b \\
        --device cpu

    uv run python scripts/build_visual_dataset.py --all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Re-use the smoke extractor's core
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_visual_features import extract_features  # noqa: E402

from itm.data.ami import load_meeting  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOT_ROOT = REPO_ROOT / "data" / "raw" / "ami" / "annotations" / "unpacked"
VIDEO_ROOT = REPO_ROOT / "data" / "raw" / "ami"
OUT_ROOT = REPO_ROOT / "data" / "processed" / "visual"

DEFAULT_ALL = [
    "ES2002a", "ES2002b", "ES2002c",
    "ES2003a", "ES2003b", "ES2003c", "ES2003d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1000a", "IS1000b",
    "IS1001a", "IS1001b", "IS1001c", "IS1001d",
]

# AMI convention from corpusResources/meetings.xml
SPEAKER_TO_CAMERA = {"A": 1, "B": 2, "C": 3, "D": 4}


def pick_two_most_active(meeting_id: str) -> tuple[str, str]:
    m = load_meeting(ANNOT_ROOT, meeting_id)
    talk = {spk: sum(s.duration for s in segs) for spk, segs in m.segments_by_speaker.items()}
    ranked = sorted(talk.items(), key=lambda kv: -kv[1])
    return ranked[0][0], ranked[1][0]


def video_path(meeting_id: str, camera: int) -> Path:
    return VIDEO_ROOT / meeting_id / "video" / f"{meeting_id}.Closeup{camera}.avi"


def out_path(meeting_id: str, speaker: str) -> Path:
    return OUT_ROOT / meeting_id / f"{speaker}.npy"


def process_meeting(meeting_id: str, *, skip_existing: bool = True) -> None:
    spk1, spk2 = pick_two_most_active(meeting_id)
    print(f"\n=== {meeting_id}: speakers ({spk1}, {spk2}) ===")
    for spk in (spk1, spk2):
        out = out_path(meeting_id, spk)
        if skip_existing and out.exists():
            arr = np.load(out, mmap_mode="r")
            print(f"  [skip] {spk} → {out.name} {arr.shape}")
            continue
        cam = SPEAKER_TO_CAMERA[spk]
        vid = video_path(meeting_id, cam)
        if not vid.exists():
            print(f"  [miss] {spk}: video not downloaded ({vid.name})")
            continue
        feats = extract_features(vid)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, feats)
        print(f"  [done] {spk} → {out} {feats.shape}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meetings", nargs="+", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--no-skip", action="store_true", help="Re-extract even if .npy exists")
    args = p.parse_args()

    meetings = DEFAULT_ALL if args.all else (args.meetings or ["ES2002a"])

    t0 = time.time()
    for mid in meetings:
        process_meeting(mid, skip_existing=not args.no_skip)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

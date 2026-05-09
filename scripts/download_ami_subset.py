"""Download a small AMI Corpus subset for ITM prototyping.

By default, fetches:
- 5 scenario meetings (ES2002 a-c, IS1000 a-b)
- Per-speaker headset audio (4 channels per meeting, needed for VAP-style training)
- Closeup videos (4 per meeting, for visual fusion in Phase 3)
- Manual annotations zip

Total ~3 GB. Skips files that already exist.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "ami"

BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
ANNOT_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations"

# Default subset — small but covers different speaker groups
DEFAULT_MEETINGS = [
    "ES2002a",
    "ES2002b",
    "ES2002c",  # Edinburgh, scenario, group 1
    "IS1000a",
    "IS1000b",  # Idiap, scenario, group 1
]

# What to download per meeting
HEADSET_CHANNELS = [0, 1, 2, 3]
CLOSEUP_CAMERAS = [1, 2, 3, 4]


def download(url: str, dest: Path) -> bool:
    """Download url to dest. Returns True if downloaded, False if skipped."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [get ] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            total = int(r.headers.get("Content-Length", 0))
            chunk = 1024 * 1024
            received = 0
            with open(tmp, "wb") as f:
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    received += len(buf)
                    if total:
                        pct = 100 * received / total
                        sys.stdout.write(
                            f"\r        {received / 1e6:6.1f} / {total / 1e6:6.1f} MB ({pct:.0f}%)"
                        )
                        sys.stdout.flush()
            sys.stdout.write("\n")
        tmp.rename(dest)
        return True
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"  [fail] {e}")
        return False


def download_meeting(meeting_id: str, audio_only: bool = False) -> None:
    print(f"\n=== {meeting_id} ===")
    # Per-speaker headset audio
    for ch in HEADSET_CHANNELS:
        url = f"{BASE}/{meeting_id}/audio/{meeting_id}.Headset-{ch}.wav"
        dest = DATA_DIR / meeting_id / "audio" / f"{meeting_id}.Headset-{ch}.wav"
        download(url, dest)
    if audio_only:
        return
    # Closeup videos
    for cam in CLOSEUP_CAMERAS:
        url = f"{BASE}/{meeting_id}/video/{meeting_id}.Closeup{cam}.avi"
        dest = DATA_DIR / meeting_id / "video" / f"{meeting_id}.Closeup{cam}.avi"
        download(url, dest)


def download_annotations() -> None:
    print("\n=== Annotations ===")
    url = f"{ANNOT_BASE}/ami_public_manual_1.6.2.zip"
    dest = DATA_DIR / "annotations" / "ami_public_manual_1.6.2.zip"
    download(url, dest)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--meetings", nargs="+", default=DEFAULT_MEETINGS)
    parser.add_argument(
        "--audio-only", action="store_true", help="Skip videos (fast prototype, ~640MB total)"
    )
    parser.add_argument(
        "--annotations-only", action="store_true", help="Only download annotations zip"
    )
    args = parser.parse_args()

    if args.annotations_only:
        download_annotations()
        return

    print(f"Downloading {len(args.meetings)} AMI meetings to {DATA_DIR}")
    print(f"Audio-only mode: {args.audio_only}")
    download_annotations()
    for m in args.meetings:
        download_meeting(m, audio_only=args.audio_only)

    print("\n=== Summary ===")
    total_bytes = sum(p.stat().st_size for p in DATA_DIR.rglob("*") if p.is_file())
    print(f"Total downloaded: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

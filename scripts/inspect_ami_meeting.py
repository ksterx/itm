"""Quick stats on a parsed AMI meeting and the ITM events extracted from it.

Usage:
    python scripts/inspect_ami_meeting.py [MEETING_ID]

Defaults to ES2002a if no argument is given.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from itm.data import EventType, extract_event_onsets, load_meeting

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOT_ROOT = REPO_ROOT / "data" / "raw" / "ami" / "annotations" / "unpacked"


def main() -> None:
    meeting_id = sys.argv[1] if len(sys.argv) > 1 else "ES2002a"
    print(f"=== AMI meeting: {meeting_id} ===\n")

    if not ANNOT_ROOT.is_dir():
        print(f"AMI annotations not found at {ANNOT_ROOT}.")
        print("Run: python scripts/download_ami_subset.py --annotations-only")
        sys.exit(1)

    meeting = load_meeting(ANNOT_ROOT, meeting_id)
    print(f"Speakers: {meeting.speakers}")

    # Per-speaker stats
    print("\n--- Per-speaker counts ---")
    print(f"{'spk':<4}{'words':>8}{'segs':>8}{'das':>8}")
    for spk in meeting.speakers:
        n_w = len(meeting.words_by_speaker[spk])
        n_s = len(meeting.segments_by_speaker[spk])
        n_d = len(meeting.dialog_acts_by_speaker[spk])
        print(f"{spk:<4}{n_w:>8}{n_s:>8}{n_d:>8}")

    # Dialog act type distribution
    print("\n--- Dialog act type distribution ---")
    da_counts: Counter[str] = Counter()
    for das in meeting.dialog_acts_by_speaker.values():
        for da in das:
            da_counts[da.da_type] += 1
    for da_type, n in da_counts.most_common():
        print(f"  {da_type:<10}{n:>6}")

    # Meeting duration
    all_segs = meeting.all_segments()
    if all_segs:
        duration = max(s.end for s in all_segs)
        print(f"\n--- Duration ---\n  {duration:.1f} seconds ({duration / 60:.1f} min)")
    else:
        duration = 0.0

    # Extract ITM events
    onsets = extract_event_onsets(meeting)
    print("\n--- ITM events ---")
    event_counts: Counter[EventType] = Counter(o.type for o in onsets)
    for ev_type in EventType:
        n = event_counts.get(ev_type, 0)
        rate = n / (duration / 60) if duration > 0 else 0
        print(f"  {ev_type.value:<14}{n:>6}  ({rate:.2f} per minute)")

    # Show first few events of each type as sanity check
    print("\n--- First 5 events of each type ---")
    for ev_type in EventType:
        print(f"\n  {ev_type.value}:")
        events = [o for o in onsets if o.type == ev_type][:5]
        for o in events:
            print(f"    t={o.time:7.2f}s  speaker={o.speaker}")
        if not events:
            print("    (none)")


if __name__ == "__main__":
    main()

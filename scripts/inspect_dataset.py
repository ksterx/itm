"""Sanity-check the AMIDataset on real downloaded AMI meetings.

Loads each meeting once, prints chunk count + tensor shapes + event totals.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from torch.utils.data import DataLoader

from itm.data import AMIDataset, EventType, ami_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOT_ROOT = REPO_ROOT / "data" / "raw" / "ami" / "annotations" / "unpacked"
AUDIO_ROOT = REPO_ROOT / "data" / "raw" / "ami"


def main() -> None:
    available = []
    for mid in ["ES2002a", "ES2002b", "ES2002c", "IS1000a", "IS1000b"]:
        if (AUDIO_ROOT / mid / "audio").is_dir():
            available.append(mid)
    if not available:
        raise SystemExit(f"No AMI audio under {AUDIO_ROOT}; download first.")

    print(f"Building dataset from: {available}")
    ds = AMIDataset(
        ANNOT_ROOT,
        AUDIO_ROOT,
        meeting_ids=available,
        chunk_sec=20.0,
        hop_sec=10.0,
        frame_rate_hz=20,
        horizon_bins=40,
    )
    print(f"Dataset size: {len(ds)} chunks")

    # Inspect first item
    item = ds[0]
    print("\n--- First chunk ---")
    print(f"meeting:   {item['meeting']}")
    print(f"speakers:  {item['speakers']}")
    print(f"start_sec: {item['start_sec']:.1f}")
    print(f"audio:     {tuple(item['audio'].shape)}  dtype={item['audio'].dtype}")
    for ev in EventType:
        h = item["hazard"][ev]
        m = item["mask"][ev]
        print(
            f"  {ev.value:<14} hazard={tuple(h.shape)} sum={int(h.sum())}  "
            f"mask={tuple(m.shape)} sum={float(m.sum()):.0f}"
        )

    # Aggregate event counts across full dataset
    print("\n--- Aggregate (full dataset) ---")
    totals: Counter[EventType] = Counter()
    for i in range(len(ds)):
        it = ds[i]
        for ev in EventType:
            totals[ev] += int(it["hazard"][ev].sum())
    for ev in EventType:
        print(f"  {ev.value:<14}: {totals[ev]:>6} positive hazard frames")

    # Try a tiny DataLoader pass
    print("\n--- DataLoader smoke test (batch_size=2) ---")
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ami_collate)
    batch = next(iter(loader))
    print(f"audio batch:  {tuple(batch['audio'].shape)}")
    for ev in EventType:
        print(f"  {ev.value:<14} hazard batch: {tuple(batch['hazard'][ev].shape)}")


if __name__ == "__main__":
    main()

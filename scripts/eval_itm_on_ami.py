"""Evaluate a trained ITMModel on AMI hold/shift, comparable with eval_maai_on_ami.py.

Loads a checkpoint produced by ``scripts/train_itm.py`` and computes:

1. **Frame-level VAD agreement** — does the model's VAD head argmax match
   the ground-truth speaker?
2. **Hold/shift accuracy** — at each mutual-silence boundary (≥ 200 ms),
   compare predicted turn-shift hazard against ground truth label.

Hold/shift mapping for ITM:
* GT: from segments around the silence (same protocol as eval_maai_on_ami)
* Pred: max(h_turn_shift[t, :K_lookahead]) at the silence midpoint
  - if max-hazard > threshold → predict SHIFT
  - else                       → predict HOLD

Threshold is calibrated on a held-out portion (default: pick the
threshold that maximises balanced accuracy on the eval set; for fair
single-meeting eval we report multiple thresholds).

Usage::

    python scripts/eval_itm_on_ami.py \
        --checkpoint checkpoints/itm_phase2b_v1_best.pt \
        --meetings IS1000b \
        --threshold 0.05
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from itm.data import AMIDataset, EventType, ami_collate
from itm.models import build_itm_model

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOT_ROOT = REPO_ROOT / "data" / "raw" / "ami" / "annotations" / "unpacked"
AUDIO_ROOT = REPO_ROOT / "data" / "raw" / "ami"

DEFAULT_THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
LOOKAHEAD_BINS = 20  # consider hazard in the next 20 bins (= 400 ms @ 50 Hz)


def find_mutual_silences_from_vad(
    vad: np.ndarray,
    frame_rate: int,
    min_duration_sec: float = 0.2,
) -> list[tuple[int, int]]:
    """Frame ranges where both speakers are silent (≥ min_duration_sec)."""
    silent = ~(vad[:, 0] | vad[:, 1])
    min_frames = int(min_duration_sec * frame_rate)
    ranges: list[tuple[int, int]] = []
    i = 0
    n = len(silent)
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        if (j - i) >= min_frames:
            ranges.append((i, j))
        i = j
    return ranges


def classify_hold_shift_gt(vad: np.ndarray, silence_start: int, silence_end: int) -> str | None:
    """Return 'hold' / 'shift' based on speaker before/after silence; None if undefined."""
    prev_ch: int | None = None
    for i in range(silence_start - 1, -1, -1):
        if vad[i, 0] and not vad[i, 1]:
            prev_ch = 0
            break
        if vad[i, 1] and not vad[i, 0]:
            prev_ch = 1
            break

    next_ch: int | None = None
    n = len(vad)
    for i in range(silence_end, n):
        if vad[i, 0] and not vad[i, 1]:
            next_ch = 0
            break
        if vad[i, 1] and not vad[i, 0]:
            next_ch = 1
            break

    if prev_ch is None or next_ch is None:
        return None
    return "hold" if prev_ch == next_ch else "shift"


@torch.no_grad()
def run_meeting_inference(
    model: torch.nn.Module,
    meeting_id: str,
    *,
    chunk_sec: float,
    target_frame_rate: int,
    horizon_bins: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a meeting in non-overlapping chunks.

    Returns ``(turn_shift_hazard, vad)`` arrays, both at ``target_frame_rate``,
    spanning the full meeting duration.
    """
    ds = AMIDataset(
        ANNOT_ROOT,
        AUDIO_ROOT,
        meeting_ids=[meeting_id],
        chunk_sec=chunk_sec,
        hop_sec=chunk_sec,  # non-overlapping
        frame_rate_hz=target_frame_rate,
        horizon_bins=horizon_bins,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ami_collate)

    hazard_chunks: list[np.ndarray] = []
    vad_chunks: list[np.ndarray] = []

    model.eval()
    for batch in loader:
        audio = batch["audio"].to(device)
        out = model(audio, return_vad=True)
        h_turn = out.hazard_logits[EventType.TURN_SHIFT].sigmoid().cpu().numpy()  # (1, T_enc, K)
        v = out.vad_logits.sigmoid().cpu().numpy()  # (1, T_enc, 2)
        # Take batch index 0
        hazard_chunks.append(h_turn[0])
        vad_chunks.append(v[0])

    hazards = np.concatenate(hazard_chunks, axis=0)  # (T_total, K)
    vads = np.concatenate(vad_chunks, axis=0)  # (T_total, 2)

    # Resample to target_frame_rate (model's encoder is ~50 Hz; if matched, identity)
    return hazards, vads


def evaluate(
    model: torch.nn.Module,
    meeting_id: str,
    *,
    threshold: float,
    chunk_sec: float,
    target_frame_rate: int,
    horizon_bins: int,
    device: str,
) -> dict:
    print(f"\n=== {meeting_id}  threshold={threshold} ===")

    # Get model output
    hazards, vads_pred = run_meeting_inference(
        model,
        meeting_id,
        chunk_sec=chunk_sec,
        target_frame_rate=target_frame_rate,
        horizon_bins=horizon_bins,
        device=device,
    )

    # Get ground-truth VAD by reusing AMIDataset's logic via a one-off load
    from itm.data.ami import load_meeting

    meeting = load_meeting(ANNOT_ROOT, meeting_id)
    talk = {s: sum(seg.duration for seg in segs) for s, segs in meeting.segments_by_speaker.items()}
    spk1, spk2 = sorted(talk.items(), key=lambda kv: -kv[1])[:2]
    spk1, spk2 = spk1[0], spk2[0]

    duration_sec = max(s.end for s in meeting.all_segments())
    n_frames = int(duration_sec * target_frame_rate)
    gt_vad = np.zeros((n_frames, 2), dtype=bool)
    dt = 1.0 / target_frame_rate
    for ch, spk in enumerate((spk1, spk2)):
        for seg in meeting.segments_by_speaker.get(spk, []):
            i_start = max(0, int(seg.start / dt))
            i_end = min(n_frames, int(seg.end / dt) + 1)
            gt_vad[i_start:i_end, ch] = True

    # Align lengths
    n = min(len(hazards), len(vads_pred), len(gt_vad))
    hazards = hazards[:n]
    vads_pred = vads_pred[:n]
    gt_vad = gt_vad[:n]

    # ---------- Frame VAD accuracy ----------
    single_mask = gt_vad[:, 0] ^ gt_vad[:, 1]
    if single_mask.sum() > 0:
        gt_speaker = gt_vad[:, 1].astype(int)
        pred_speaker = (vads_pred[:, 1] > vads_pred[:, 0]).astype(int)
        frame_acc = float((gt_speaker[single_mask] == pred_speaker[single_mask]).mean())
    else:
        frame_acc = float("nan")

    # ---------- Hold/shift via hazard ----------
    silences = find_mutual_silences_from_vad(gt_vad, target_frame_rate)
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for i_s, i_e in silences:
        gt_label = classify_hold_shift_gt(gt_vad, i_s, i_e)
        if gt_label is None:
            continue
        mid = (i_s + i_e) // 2
        if mid >= n:
            continue
        # Predicted: max hazard in lookahead window
        max_h = float(hazards[mid, :LOOKAHEAD_BINS].max())
        pred_label = "shift" if max_h > threshold else "hold"
        counts[gt_label] += 1
        if pred_label == gt_label:
            correct[gt_label] += 1
        confusion[(gt_label, pred_label)] += 1

    return {
        "meeting": meeting_id,
        "duration_sec": duration_sec,
        "frame_acc": frame_acc,
        "hold_correct": correct["hold"],
        "hold_total": counts["hold"],
        "shift_correct": correct["shift"],
        "shift_total": counts["shift"],
        "confusion": dict(confusion),
        "threshold": threshold,
    }


def print_aggregate(results: list[dict]) -> None:
    if not results:
        print("\nNo results.")
        return
    print("\n" + "=" * 80)
    print(f"AGGREGATE  (threshold={results[0]['threshold']})")
    print("=" * 80)
    print(f"{'meeting':<12}{'frame_acc':>11}{'hold':>14}{'shift':>14}{'overall':>16}")
    h_c = h_t = s_c = s_t = 0
    fa_w = dur = 0.0
    for r in results:
        ho_c, ho_t = r["hold_correct"], r["hold_total"]
        sh_c, sh_t = r["shift_correct"], r["shift_total"]
        oc, ot = ho_c + sh_c, ho_t + sh_t
        oa = oc / ot if ot else float("nan")
        print(
            f"{r['meeting']:<12}{r['frame_acc']:>11.3f}"
            f"{f'{ho_c}/{ho_t}':>14}{f'{sh_c}/{sh_t}':>14}"
            f"{f'{oc}/{ot} ({oa:.3f})':>16}"
        )
        h_c += ho_c
        h_t += ho_t
        s_c += sh_c
        s_t += sh_t
        fa_w += r["frame_acc"] * r["duration_sec"]
        dur += r["duration_sec"]
    oc, ot = h_c + s_c, h_t + s_t
    oa = oc / ot if ot else float("nan")
    print("-" * 80)
    print(
        f"{'POOLED':<12}{(fa_w / dur):>11.3f}"
        f"{f'{h_c}/{h_t}':>14}{f'{s_c}/{s_t}':>14}"
        f"{f'{oc}/{ot} ({oa:.3f})':>16}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--meetings", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--threshold-sweep", action="store_true", help="Run all DEFAULT_THRESHOLDS and report each"
    )
    parser.add_argument("--chunk-sec", type=float, default=20.0)
    parser.add_argument("--target-frame-rate", type=int, default=50)
    parser.add_argument("--horizon-bins", type=int, default=40)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    print(f"  epoch={ckpt['epoch']}, val_loss={ckpt.get('val_loss', '-')}")
    train_args = ckpt.get("args", {})
    model = build_itm_model(
        lang="en",
        frame_rate=train_args.get("frame_rate", 20),
        context_len_sec=20,
        horizon_bins=args.horizon_bins,
        device=args.device,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"  ⚠ missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    model.to(args.device)
    model.eval()

    if not ANNOT_ROOT.is_dir():
        sys.exit(f"AMI annotations not found at {ANNOT_ROOT}")

    thresholds = DEFAULT_THRESHOLDS if args.threshold_sweep else [args.threshold]

    for thr in thresholds:
        results = []
        for mid in args.meetings:
            r = evaluate(
                model,
                mid,
                threshold=thr,
                chunk_sec=args.chunk_sec,
                target_frame_rate=args.target_frame_rate,
                horizon_bins=args.horizon_bins,
                device=args.device,
            )
            results.append(r)
        print_aggregate(results)


if __name__ == "__main__":
    main()

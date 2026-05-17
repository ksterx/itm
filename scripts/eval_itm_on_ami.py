"""Evaluate a trained ITMModel on AMI hold/shift, comparable with eval_maai_on_ami.py.

Loads a checkpoint produced by ``scripts/train_itm.py`` and computes:

1. **Frame-level VAD agreement** — does the model's VAD head argmax match
   the ground-truth speaker?
2. **Hold/shift accuracy** — at each mutual-silence boundary (≥ 200 ms),
   compare predicted turn-shift hazard against ground truth label.
3. **AUC / PR-AUC** — threshold-free metrics over the (gt, max_hazard) pairs.

Hold/shift mapping for ITM:
* GT: from segments around the silence (same protocol as eval_maai_on_ami)
* Pred: max(h_turn_shift[t, :K_lookahead]) at the silence midpoint
  - if max-hazard > threshold → predict SHIFT
  - else                       → predict HOLD

The threshold sweep and AUC reuse the same per-silence (gt, score) pairs,
so inference runs only once per meeting.

Usage::

    python scripts/eval_itm_on_ami.py \\
        --checkpoint checkpoints/itm_phase2b_v3_best.pt \\
        --meetings IS1000b \\
        --threshold-sweep --auc
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

# Wide range covers both survival-hazard scores (typically 0.01–0.3 for ITM v1–v3)
# and shift-head probabilities (typically 0.3–0.5 in v4+).
DEFAULT_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.38, 0.4, 0.42, 0.45, 0.5]
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
    visual_root: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Returns (hazard, vad, shift_prob_or_None)."""
    ds_kwargs: dict = {}
    if visual_root is not None:
        ds_kwargs["visual_root"] = visual_root
    ds = AMIDataset(
        ANNOT_ROOT,
        AUDIO_ROOT,
        meeting_ids=[meeting_id],
        chunk_sec=chunk_sec,
        hop_sec=chunk_sec,
        frame_rate_hz=target_frame_rate,
        horizon_bins=horizon_bins,
        **ds_kwargs,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ami_collate)

    hazard_chunks: list[np.ndarray] = []
    vad_chunks: list[np.ndarray] = []
    shift_chunks: list[np.ndarray] = []

    model.eval()
    has_shift = False
    for batch in loader:
        audio = batch["audio"].to(device)
        visual = batch.get("visual")
        visual_mask = batch.get("visual_mask")
        if visual is not None:
            visual = visual.to(device)
            visual_mask = visual_mask.to(device)
        out = model(audio, return_vad=True, visual=visual, visual_mask=visual_mask)
        h_turn = out.hazard_logits[EventType.TURN_SHIFT].sigmoid().cpu().numpy()
        v = out.vad_logits.sigmoid().cpu().numpy()
        hazard_chunks.append(h_turn[0])
        vad_chunks.append(v[0])
        if out.shift_logits is not None:
            has_shift = True
            shift_chunks.append(out.shift_logits.sigmoid().cpu().numpy()[0])

    hazards = np.concatenate(hazard_chunks, axis=0)
    vads = np.concatenate(vad_chunks, axis=0)
    shift_probs = np.concatenate(shift_chunks, axis=0) if has_shift else None
    return hazards, vads, shift_probs


def gt_vad_for_meeting(meeting_id: str, target_frame_rate: int) -> tuple[np.ndarray, float]:
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
    return gt_vad, duration_sec


def collect_silence_scores(
    hazards: np.ndarray,
    gt_vad: np.ndarray,
    target_frame_rate: int,
    shift_probs: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """For each well-defined mutual silence: return (gt_label, score).

    Score is mean(shift_prob) over silence frames if a shift head is present
    (v4); otherwise max(turn_hazard) over the lookahead window at the
    silence midpoint (v1–v3).
    """
    n = min(len(hazards), len(gt_vad))
    hazards = hazards[:n]
    gt_vad = gt_vad[:n]
    if shift_probs is not None:
        shift_probs = shift_probs[:n]
    silences = find_mutual_silences_from_vad(gt_vad, target_frame_rate)
    pairs: list[tuple[str, float]] = []
    for i_s, i_e in silences:
        gt_label = classify_hold_shift_gt(gt_vad, i_s, i_e)
        if gt_label is None:
            continue
        mid = (i_s + i_e) // 2
        if mid >= n:
            continue
        if shift_probs is not None:
            score = float(shift_probs[i_s : max(i_s + 1, i_e)].mean())
        else:
            score = float(hazards[mid, :LOOKAHEAD_BINS].max())
        pairs.append((gt_label, score))
    return pairs


def score_threshold(pairs: list[tuple[str, float]], threshold: float) -> dict:
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for gt_label, score in pairs:
        pred_label = "shift" if score > threshold else "hold"
        counts[gt_label] += 1
        if pred_label == gt_label:
            correct[gt_label] += 1
        confusion[(gt_label, pred_label)] += 1
    return {
        "hold_correct": correct["hold"],
        "hold_total": counts["hold"],
        "shift_correct": correct["shift"],
        "shift_total": counts["shift"],
        "confusion": dict(confusion),
        "threshold": threshold,
    }


def compute_auc(pairs: list[tuple[str, float]]) -> dict:
    """ROC-AUC and PR-AUC for shift detection (positive class = 'shift').

    Implemented without sklearn to keep deps minimal.
    """
    if not pairs:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "n_pos": 0, "n_neg": 0}
    y = np.array([1 if g == "shift" else 0 for g, _ in pairs], dtype=np.int64)
    s = np.array([sc for _, sc in pairs], dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "n_pos": n_pos, "n_neg": n_neg}

    # ROC-AUC via Mann-Whitney U statistic
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    # Average rank for ties
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_ranks_pos = ranks[y == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    roc_auc = float(u / (n_pos * n_neg))

    # PR-AUC by trapezoidal integration over sorted thresholds
    desc = np.argsort(-s, kind="mergesort")
    y_sorted = y[desc]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # Prepend (recall=0, precision=1) for proper integration
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    # AP = sum of (recall[i] - recall[i-1]) * precision[i]
    pr_auc = float(np.sum(np.diff(recall) * precision[1:]))

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "n_pos": n_pos, "n_neg": n_neg}


def evaluate_meeting(
    model: torch.nn.Module,
    meeting_id: str,
    *,
    chunk_sec: float,
    target_frame_rate: int,
    horizon_bins: int,
    device: str,
    visual_root: str | None = None,
) -> dict:
    """Run inference once, return scores + frame VAD accuracy."""
    print(f"\n--- inference: {meeting_id} ---")
    hazards, vads_pred, shift_probs = run_meeting_inference(
        model,
        meeting_id,
        chunk_sec=chunk_sec,
        target_frame_rate=target_frame_rate,
        horizon_bins=horizon_bins,
        device=device,
        visual_root=visual_root,
    )
    gt_vad, duration_sec = gt_vad_for_meeting(meeting_id, target_frame_rate)
    n = min(len(hazards), len(vads_pred), len(gt_vad))
    hazards = hazards[:n]
    vads_pred = vads_pred[:n]
    gt_vad = gt_vad[:n]
    if shift_probs is not None:
        shift_probs = shift_probs[:n]

    single_mask = gt_vad[:, 0] ^ gt_vad[:, 1]
    if single_mask.sum() > 0:
        gt_speaker = gt_vad[:, 1].astype(int)
        pred_speaker = (vads_pred[:, 1] > vads_pred[:, 0]).astype(int)
        frame_acc = float((gt_speaker[single_mask] == pred_speaker[single_mask]).mean())
    else:
        frame_acc = float("nan")

    pairs = collect_silence_scores(hazards, gt_vad, target_frame_rate, shift_probs=shift_probs)
    score_source = "shift_head.mean(silence)" if shift_probs is not None else "max(hazard[:K])"
    print(f"  score source: {score_source}")
    return {
        "meeting": meeting_id,
        "duration_sec": duration_sec,
        "frame_acc": frame_acc,
        "pairs": pairs,
        "score_source": score_source,
    }


def print_threshold_table(meeting_results: list[dict], threshold: float) -> None:
    print(f"\nthreshold = {threshold}")
    print(f"{'meeting':<12}{'frame_acc':>11}{'hold':>14}{'shift':>14}{'overall':>16}")
    h_c = h_t = s_c = s_t = 0
    fa_w = dur = 0.0
    for r in meeting_results:
        scored = score_threshold(r["pairs"], threshold)
        ho_c, ho_t = scored["hold_correct"], scored["hold_total"]
        sh_c, sh_t = scored["shift_correct"], scored["shift_total"]
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
    print(
        f"{'POOLED':<12}{(fa_w / dur):>11.3f}"
        f"{f'{h_c}/{h_t}':>14}{f'{s_c}/{s_t}':>14}"
        f"{f'{oc}/{ot} ({oa:.3f})':>16}"
    )


def print_auc_table(meeting_results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("THRESHOLD-FREE METRICS (positive class = 'shift')")
    print("=" * 80)
    print(f"{'meeting':<12}{'n_pos':>8}{'n_neg':>8}{'ROC-AUC':>12}{'PR-AUC':>10}")
    pooled: list[tuple[str, float]] = []
    for r in meeting_results:
        pooled.extend(r["pairs"])
        m = compute_auc(r["pairs"])
        print(
            f"{r['meeting']:<12}{m['n_pos']:>8}{m['n_neg']:>8}"
            f"{m['roc_auc']:>12.3f}{m['pr_auc']:>10.3f}"
        )
    p = compute_auc(pooled)
    print(f"{'POOLED':<12}{p['n_pos']:>8}{p['n_neg']:>8}{p['roc_auc']:>12.3f}{p['pr_auc']:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--meetings", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--threshold-sweep", action="store_true", help="Run all DEFAULT_THRESHOLDS and report each"
    )
    parser.add_argument(
        "--auc", action="store_true", help="Report ROC-AUC and PR-AUC (threshold-free)"
    )
    parser.add_argument("--chunk-sec", type=float, default=20.0)
    parser.add_argument("--target-frame-rate", type=int, default=50)
    parser.add_argument("--horizon-bins", type=int, default=40)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--visual-root",
        default=None,
        help="Per-meeting visual feature dir (auto-enabled if checkpoint has visual_encoder)",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    print(f"  epoch={ckpt['epoch']}, val_loss={ckpt.get('val_loss', '-')}")
    train_args = ckpt.get("args", {})
    has_shift_head = any(k.startswith("shift_head.") for k in ckpt["model_state"])
    has_visual = any(k.startswith("visual_encoder.") for k in ckpt["model_state"])
    visual_root = args.visual_root
    if has_shift_head:
        print("  checkpoint contains shift_head — enabling")
    if has_visual:
        if visual_root is None:
            visual_root = "data/processed/visual"
        print(f"  checkpoint contains visual_encoder — enabling, visual_root={visual_root}")
    model = build_itm_model(
        lang="en",
        frame_rate=train_args.get("frame_rate", 20),
        context_len_sec=20,
        horizon_bins=args.horizon_bins,
        device=args.device,
        enable_shift_head=has_shift_head,
        enable_visual=has_visual,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"  ⚠ missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    model.to(args.device)
    model.eval()

    if not ANNOT_ROOT.is_dir():
        sys.exit(f"AMI annotations not found at {ANNOT_ROOT}")

    meeting_results = [
        evaluate_meeting(
            model,
            mid,
            chunk_sec=args.chunk_sec,
            target_frame_rate=args.target_frame_rate,
            horizon_bins=args.horizon_bins,
            device=args.device,
            visual_root=visual_root if has_visual else None,
        )
        for mid in args.meetings
    ]

    thresholds = DEFAULT_THRESHOLDS if args.threshold_sweep else [args.threshold]
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP")
    print("=" * 80)
    for thr in thresholds:
        print_threshold_table(meeting_results, thr)

    if args.auc:
        print_auc_table(meeting_results)


if __name__ == "__main__":
    main()

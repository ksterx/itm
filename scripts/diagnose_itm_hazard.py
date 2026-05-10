"""Diagnose where the v3 hazard signal goes wrong.

Try different score aggregations / inversions / lookahead windows and report
their ROC-AUC for shift detection. If any of them gives AUC > 0.6, that is
the next thing to try in eval. If none do, the model truly lacks a useful
shift signal — v4 must change the training objective or add a dedicated head.

Usage::

    python scripts/diagnose_itm_hazard.py \\
        --checkpoint checkpoints/itm_phase2b_v3_best.pt \\
        --meeting IS1000b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_itm_on_ami import (  # noqa: E402
    ANNOT_ROOT,
    AUDIO_ROOT,
    classify_hold_shift_gt,
    compute_auc,
    find_mutual_silences_from_vad,
    gt_vad_for_meeting,
)

from itm.data import AMIDataset, EventType, ami_collate  # noqa: E402
from itm.models import build_itm_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def collect_with_score_fn(
    hazards_turn: np.ndarray,
    hazards_bc: np.ndarray | None,
    hazards_overlap: np.ndarray | None,
    gt_vad: np.ndarray,
    target_frame_rate: int,
    score_fn,
) -> list[tuple[str, float]]:
    """Apply a score function (per-silence) and return (gt_label, score) pairs."""
    n = min(
        len(hazards_turn),
        len(gt_vad),
    )
    silences = find_mutual_silences_from_vad(gt_vad[:n], target_frame_rate)
    pairs: list[tuple[str, float]] = []
    for i_s, i_e in silences:
        gt_label = classify_hold_shift_gt(gt_vad, i_s, i_e)
        if gt_label is None:
            continue
        mid = (i_s + i_e) // 2
        if mid >= n:
            continue
        score = score_fn(
            hazards_turn[mid],
            hazards_bc[mid] if hazards_bc is not None else None,
            hazards_overlap[mid] if hazards_overlap is not None else None,
        )
        pairs.append((gt_label, float(score)))
    return pairs


@torch.no_grad()
def run_full_inference(model, meeting_id, *, target_frame_rate=50, horizon_bins=40, device="cpu"):
    """Run inference and return all 3 hazard channels + vad."""
    ds = AMIDataset(
        ANNOT_ROOT,
        AUDIO_ROOT,
        meeting_ids=[meeting_id],
        chunk_sec=20.0,
        hop_sec=20.0,
        frame_rate_hz=target_frame_rate,
        horizon_bins=horizon_bins,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ami_collate)
    h_turn, h_bc, h_ov, vads = [], [], [], []
    model.eval()
    for batch in loader:
        audio = batch["audio"].to(device)
        out = model(audio, return_vad=True)
        h_turn.append(out.hazard_logits[EventType.TURN_SHIFT].sigmoid().cpu().numpy()[0])
        h_bc.append(out.hazard_logits[EventType.BACKCHANNEL].sigmoid().cpu().numpy()[0])
        h_ov.append(out.hazard_logits[EventType.OVERLAP].sigmoid().cpu().numpy()[0])
        vads.append(out.vad_logits.sigmoid().cpu().numpy()[0])
    return (
        np.concatenate(h_turn, axis=0),
        np.concatenate(h_bc, axis=0),
        np.concatenate(h_ov, axis=0),
        np.concatenate(vads, axis=0),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--meeting", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print(f"Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = ckpt.get("args", {})
    model = build_itm_model(
        lang="en",
        frame_rate=train_args.get("frame_rate", 20),
        context_len_sec=20,
        horizon_bins=40,
        device=args.device,
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(args.device).eval()

    print(f"Inference on {args.meeting}...")
    h_turn, h_bc, h_ov, _vads = run_full_inference(model, args.meeting, device=args.device)
    gt_vad, _ = gt_vad_for_meeting(args.meeting, 50)

    n = min(len(h_turn), len(gt_vad))
    h_turn, h_bc, h_ov, gt_vad = h_turn[:n], h_bc[:n], h_ov[:n], gt_vad[:n]

    print(f"  hazard turn shape: {h_turn.shape}, range [{h_turn.min():.3f}, {h_turn.max():.3f}]")
    print(f"  hazard bc   range [{h_bc.min():.3f}, {h_bc.max():.3f}]")
    print(f"  hazard ov   range [{h_ov.min():.3f}, {h_ov.max():.3f}]")

    # Try many score functions
    variants = {
        "turn:max[:20] (current)": lambda t, b, o: t[:20].max(),
        "turn:mean[:20]": lambda t, b, o: t[:20].mean(),
        "turn:max[:5]  (next 100ms)": lambda t, b, o: t[:5].max(),
        "turn:max[:10] (next 200ms)": lambda t, b, o: t[:10].max(),
        "turn:max[:40] (full horizon)": lambda t, b, o: t[:40].max(),
        "turn:sum[:20]": lambda t, b, o: t[:20].sum(),
        "turn:max[20:40] (later half)": lambda t, b, o: t[20:40].max(),
        "INVERT: 1 - turn:max[:20]": lambda t, b, o: 1.0 - t[:20].max(),
        "turn - bc:max[:20]": lambda t, b, o: (t[:20] - b[:20]).max(),
        "turn / (turn+bc):max[:20]": lambda t, b, o: (t[:20] / (t[:20] + b[:20] + 1e-6)).max(),
        "bc:max[:20] (backchannel)": lambda t, b, o: b[:20].max(),
        "overlap:max[:20]": lambda t, b, o: o[:20].max(),
    }

    print(f"\n{'variant':<40}{'n_pos':>8}{'n_neg':>8}{'ROC-AUC':>10}{'PR-AUC':>10}")
    print("-" * 76)
    for name, fn in variants.items():
        pairs = collect_with_score_fn(h_turn, h_bc, h_ov, gt_vad, 50, fn)
        m = compute_auc(pairs)
        print(f"{name:<40}{m['n_pos']:>8}{m['n_neg']:>8}{m['roc_auc']:>10.3f}{m['pr_auc']:>10.3f}")


if __name__ == "__main__":
    main()

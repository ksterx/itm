"""Visual-only shift discrimination probe (Codex v8 go/no-go test).

Trains a tiny MLP head **directly on MediaPipe visual features** (no audio)
to predict per-silence-segment shift/hold labels. Same train/val/test split
as v6-α/v8 (13 train meetings, 4 val incl. IS1000b). Eval is ROC-AUC on
IS1000b alone, computed identically to ``eval_itm_on_ami.py`` for direct
comparison.

Go/no-go (per Codex):
    visual-only ROC-AUC ≥ 0.56 on IS1000b  →  visual carries independent signal,
                                              continue improving fusion (v8b/v8c).
    visual-only ROC-AUC <  0.56            →  visual is noise, retire Phase 3
                                              fusion and move to Phase 4.

The probe uses frame_rate_hz=25 (visual native rate) so visual features
and shift labels are aligned by construction without resampling.

Usage::

    uv run python scripts/probe_visual_only.py --epochs 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from itm.data import AMIDataset, ami_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOT_ROOT = REPO_ROOT / "data" / "raw" / "ami" / "annotations" / "unpacked"
AUDIO_ROOT = REPO_ROOT / "data" / "raw" / "ami"
VISUAL_ROOT = REPO_ROOT / "data" / "processed" / "visual"
CKPT_DIR = REPO_ROOT / "checkpoints"

DEFAULT_MEETINGS = [
    "ES2002a", "ES2002b", "ES2002c",
    "ES2003a", "ES2003b", "ES2003c", "ES2003d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1000a", "IS1001a", "IS1001b", "IS1001c", "IS1001d",
    "IS1000b",  # last, will be in val + isolated test
]
N_TRAIN = 13


class VisualProbe(nn.Module):
    """Per-frame MLP: (B, T, 2, 56) → (B, T) logit. ~30K params."""

    def __init__(self, visual_dim: int = 56, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(visual_dim * 2),
            nn.Linear(visual_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        x = visual.flatten(-2, -1)  # (B, T, 2*56)
        return self.net(x).squeeze(-1)  # (B, T)


def segment_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: float = 1.5,
) -> torch.Tensor | None:
    """Per-silence-segment BCE. Mirrors training._segment_shift_bce."""
    seg_logits: list[torch.Tensor] = []
    seg_targets: list[torch.Tensor] = []
    bsz, n_t = mask.shape
    for b in range(bsz):
        m = mask[b]
        in_run = False
        run_start = 0
        for t in range(n_t):
            if m[t] > 0 and not in_run:
                in_run = True
                run_start = t
            elif m[t] == 0 and in_run:
                in_run = False
                seg_logits.append(logits[b, run_start:t].mean())
                seg_targets.append(target[b, run_start])
        if in_run:
            seg_logits.append(logits[b, run_start:n_t].mean())
            seg_targets.append(target[b, run_start])
    if not seg_logits:
        return None
    sl = torch.stack(seg_logits)
    st = torch.stack(seg_targets).float()
    pw = torch.tensor(pos_weight, device=sl.device, dtype=sl.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(sl, st, pos_weight=pw)


def compute_auc(pairs: list[tuple[str, float]]) -> tuple[float, float]:
    if not pairs:
        return float("nan"), float("nan")
    y = np.array([1 if g == "shift" else 0 for g, _ in pairs], dtype=np.int64)
    s = np.array([sc for _, sc in pairs], dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    sum_ranks_pos = ranks[y == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    roc = float(u / (n_pos * n_neg))
    desc = np.argsort(-s, kind="mergesort")
    y_sorted = y[desc]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    pr = float(np.sum(np.diff(recall) * precision[1:]))
    return roc, pr


@torch.no_grad()
def eval_probe(model: VisualProbe, meeting_id: str, chunk_sec: float, fps: int) -> dict:
    """Score every labelable silence on a meeting and return AUC + counts."""
    ds = AMIDataset(
        ANNOT_ROOT, AUDIO_ROOT, meeting_ids=[meeting_id],
        chunk_sec=chunk_sec, hop_sec=chunk_sec, frame_rate_hz=fps,
        visual_root=VISUAL_ROOT,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=ami_collate)
    pairs: list[tuple[str, float]] = []
    model.eval()
    for batch in loader:
        if "visual" not in batch:
            continue
        v = batch["visual"]
        logits = model(v)
        probs = logits.sigmoid()  # (1, T)
        st = batch["shift_target"][0]
        sm = batch["shift_mask"][0]
        # Walk silence runs, score each
        in_run = False
        run_start = 0
        n_t = sm.size(0)
        for t in range(n_t):
            if sm[t] > 0 and not in_run:
                in_run = True
                run_start = t
            elif sm[t] == 0 and in_run:
                in_run = False
                score = float(probs[0, run_start:t].mean())
                label = "shift" if st[run_start] > 0 else "hold"
                pairs.append((label, score))
        if in_run:
            score = float(probs[0, run_start:n_t].mean())
            label = "shift" if st[run_start] > 0 else "hold"
            pairs.append((label, score))
    roc, pr = compute_auc(pairs)
    n_pos = sum(1 for g, _ in pairs if g == "shift")
    return {"meeting": meeting_id, "n_pos": n_pos, "n_neg": len(pairs) - n_pos, "roc": roc, "pr": pr}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--chunk-sec", type=float, default=20.0)
    p.add_argument("--hop-sec", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=25, help="frame rate (= visual native rate)")
    p.add_argument("--pos-weight", type=float, default=1.5)
    p.add_argument("--save-name", default="visual_probe")
    args = p.parse_args()

    train_ms = DEFAULT_MEETINGS[:N_TRAIN]
    val_ms = DEFAULT_MEETINGS[N_TRAIN:]
    test_meeting = "IS1000b"
    print(f"train: {len(train_ms)} meetings  val: {len(val_ms)}  test: {test_meeting}")

    train_ds = AMIDataset(
        ANNOT_ROOT, AUDIO_ROOT, meeting_ids=train_ms,
        chunk_sec=args.chunk_sec, hop_sec=args.hop_sec, frame_rate_hz=args.fps,
        visual_root=VISUAL_ROOT,
    )
    val_ds = AMIDataset(
        ANNOT_ROOT, AUDIO_ROOT, meeting_ids=val_ms,
        chunk_sec=args.chunk_sec, hop_sec=args.chunk_sec, frame_rate_hz=args.fps,
        visual_root=VISUAL_ROOT,
    )
    print(f"train chunks: {len(train_ds)}  val chunks: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=ami_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ami_collate,
    )

    model = VisualProbe()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    CKPT_DIR.mkdir(exist_ok=True)
    log_path = CKPT_DIR / f"{args.save_name}_log.jsonl"
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            if "visual" not in batch:
                continue
            opt.zero_grad(set_to_none=True)
            logits = model(batch["visual"])
            loss = segment_bce(
                logits, batch["shift_target"], batch["shift_mask"],
                pos_weight=args.pos_weight,
            )
            if loss is None:
                continue
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"  ep{epoch} st{step} loss={loss.item():.4f}")

        # Val
        model.eval()
        vals: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                if "visual" not in batch:
                    continue
                logits = model(batch["visual"])
                loss = segment_bce(
                    logits, batch["shift_target"], batch["shift_mask"],
                    pos_weight=args.pos_weight,
                )
                if loss is not None:
                    vals.append(loss.item())
        vl = sum(vals) / max(1, len(vals))
        print(f"[val] ep{epoch} loss={vl:.4f}")
        with log_path.open("a") as f:
            f.write(json.dumps({"kind": "val", "epoch": epoch, "loss": vl}) + "\n")
        if vl < best_val:
            best_val = vl
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "args": vars(args)},
                       CKPT_DIR / f"{args.save_name}_best.pt")
            print(f"  saved best")

    print(f"\ntrain done in {(time.time() - t0) / 60:.1f} min\n")

    # Final eval on IS1000b only
    print(f"=== test eval on {test_meeting} ===")
    ckpt = torch.load(CKPT_DIR / f"{args.save_name}_best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    res = eval_probe(model, test_meeting, args.chunk_sec, args.fps)
    print(f"{res['meeting']}: n_pos={res['n_pos']}  n_neg={res['n_neg']}  "
          f"ROC-AUC={res['roc']:.3f}  PR-AUC={res['pr']:.3f}")

    threshold = 0.56
    verdict = "CONTINUE Phase 3" if res["roc"] >= threshold else "STOP Phase 3"
    print(f"\nCodex go/no-go threshold = {threshold}")
    print(f"→ {verdict}")


if __name__ == "__main__":
    main()

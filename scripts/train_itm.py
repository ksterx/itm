"""Train an ITMModel on AMI Corpus with multi-event survival hazard loss.

Phase 2-B: fine-tune the MaAI VAP backbone (transformer layers) plus
freshly-initialized hazard heads on AMI multi-party meetings.

Usage::

    # Tiny sanity run (1 meeting, 1 epoch, batch=2)
    python scripts/train_itm.py --meetings ES2002a --epochs 1 --batch-size 2

    # Full Phase 2-B run (5 meetings, multi epoch)
    python scripts/train_itm.py --all --epochs 8 --batch-size 4 --device cuda

CPU works for sanity; GPU strongly recommended for full runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from itm.data import AMIDataset, ami_collate
from itm.models import build_itm_model
from itm.training import eval_step, train_step

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOT_ROOT = REPO_ROOT / "data" / "raw" / "ami" / "annotations" / "unpacked"
AUDIO_ROOT = REPO_ROOT / "data" / "raw" / "ami"
CKPT_DIR = REPO_ROOT / "checkpoints"

DEFAULT_ALL = ["ES2002a", "ES2002b", "ES2002c", "IS1000a", "IS1000b"]


def split_meetings(meetings: list[str]) -> tuple[list[str], list[str]]:
    """80/20 train/val split by meeting (speaker-disjoint, simple)."""
    if len(meetings) == 1:
        return meetings, meetings  # tiny smoke test only
    n_train = max(1, int(0.8 * len(meetings)))
    return meetings[:n_train], meetings[n_train:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meetings", nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="Use all 5 default meetings")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3.63e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--chunk-sec", type=float, default=20.0)
    parser.add_argument("--hop-sec", type=float, default=10.0)
    parser.add_argument(
        "--frame-rate", type=int, default=20, help="Frame rate of MaAI checkpoint (5/10/12.5/20)"
    )
    parser.add_argument(
        "--target-frame-rate",
        type=int,
        default=50,
        help="Target frame rate (≈ encoder rate, 50 Hz for MaAI)",
    )
    parser.add_argument("--horizon-bins", type=int, default=40)
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
        help="Positive-class weight for survival NLL (try 30–100 to fight imbalance)",
    )
    parser.add_argument(
        "--use-vad-aux",
        action="store_true",
        help="Add per-channel VAD BCE as auxiliary loss to preserve VAD capability",
    )
    parser.add_argument("--vad-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--freeze-transformer",
        action="store_true",
        help="Freeze ar_channel/ar in the VAP backbone; train only hazard heads (+VAD if used)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-steps", type=int, default=0, help="If > 0, stop after this many steps (debugging)."
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-name", default="itm_phase2b")
    args = parser.parse_args()

    if args.all:
        meetings = DEFAULT_ALL
    elif args.meetings:
        meetings = args.meetings
    else:
        meetings = ["ES2002a"]

    train_ms, val_ms = split_meetings(meetings)
    print(f"train meetings: {train_ms}")
    print(f"val   meetings: {val_ms}")

    print("Building datasets...")
    train_ds = AMIDataset(
        ANNOT_ROOT,
        AUDIO_ROOT,
        meeting_ids=train_ms,
        chunk_sec=args.chunk_sec,
        hop_sec=args.hop_sec,
        frame_rate_hz=args.target_frame_rate,
        horizon_bins=args.horizon_bins,
    )
    val_ds = AMIDataset(
        ANNOT_ROOT,
        AUDIO_ROOT,
        meeting_ids=val_ms,
        chunk_sec=args.chunk_sec,
        hop_sec=args.chunk_sec,  # non-overlapping for clean val
        frame_rate_hz=args.target_frame_rate,
        horizon_bins=args.horizon_bins,
    )
    print(f"train: {len(train_ds)} chunks   val: {len(val_ds)} chunks")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ami_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ami_collate,
    )

    print(f"\nBuilding model (frame_rate={args.frame_rate}, device={args.device})...")
    model = build_itm_model(
        lang="en",
        frame_rate=args.frame_rate,
        context_len_sec=20,
        horizon_bins=args.horizon_bins,
        device=args.device,
        freeze_transformer=args.freeze_transformer,
    )
    model.to(args.device)
    print("Parameter counts:")
    for k, v in model.count_parameters().items():
        print(f"  {k:<14} {v:>10,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    CKPT_DIR.mkdir(exist_ok=True)
    log_path = CKPT_DIR / f"{args.save_name}_log.jsonl"
    print(f"Logging to {log_path}")
    print(f"Checkpoints: {CKPT_DIR / args.save_name}_*.pt\n")

    best_val = float("inf")
    global_step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        # -------- Train --------
        for _step, batch in enumerate(train_loader):
            batch = _move_batch(batch, args.device)
            info = train_step(
                model,
                batch,
                optim,
                pos_weight=args.pos_weight,
                use_vad_aux=args.use_vad_aux,
                vad_loss_weight=args.vad_loss_weight,
            )
            global_step += 1

            if global_step % args.log_every == 0:
                ev_str = " ".join(
                    f"{ev.value}={info.per_event_loss[ev]:.3f}" for ev in info.per_event_loss
                )
                vad_str = f" vad={info.vad_loss:.3f}" if info.vad_loss is not None else ""
                msg = (
                    f"epoch={epoch} step={global_step} loss={info.total_loss:.4f} {ev_str}{vad_str}"
                )
                print(msg)
                with log_path.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "kind": "train",
                                "epoch": epoch,
                                "step": global_step,
                                "loss": info.total_loss,
                                "per_event": {ev.value: v for ev, v in info.per_event_loss.items()},
                                "vad_loss": info.vad_loss,
                            }
                        )
                        + "\n"
                    )

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

        # -------- Validation --------
        val_losses: list[float] = []
        for batch in val_loader:
            batch = _move_batch(batch, args.device)
            info = eval_step(model, batch)
            val_losses.append(info.total_loss)
        val_loss = sum(val_losses) / max(1, len(val_losses))
        print(f"  [val] epoch={epoch} loss={val_loss:.4f}  (n_batches={len(val_losses)})")

        with log_path.open("a") as f:
            f.write(json.dumps({"kind": "val", "epoch": epoch, "loss": val_loss}) + "\n")

        # -------- Save checkpoint --------
        ckpt_path = CKPT_DIR / f"{args.save_name}_epoch{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            },
            ckpt_path,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_path = CKPT_DIR / f"{args.save_name}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  saved best → {best_path}")

    print(f"\nDone. {global_step} steps in {time.time() - t_start:.1f}s")


def _move_batch(batch: dict, device: str) -> dict:
    out = dict(batch)
    out["audio"] = batch["audio"].to(device)
    out["hazard"] = {ev: t.to(device) for ev, t in batch["hazard"].items()}
    out["mask"] = {ev: t.to(device) for ev, t in batch["mask"].items()}
    if "vad_target" in batch:
        out["vad_target"] = batch["vad_target"].to(device)
    return out


if __name__ == "__main__":
    main()

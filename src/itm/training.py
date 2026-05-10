"""Training utilities for ITM.

Glue between :class:`itm.models.ITMModel` and :class:`itm.data.AMIDataset`
for multi-event hazard fine-tuning on AMI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from itm.data.targets import survival_nll_loss

if TYPE_CHECKING:
    from itm.data.labels import EventType
    from itm.models.itm_model import ITMModel


@dataclass
class TrainStepOutput:
    """Per-step diagnostics."""

    total_loss: float
    per_event_loss: dict[EventType, float]
    n_observed: dict[EventType, int]
    """Number of bins that contributed to each event's loss."""

    vad_loss: float | None = None
    """Auxiliary VAD BCE loss when ``use_vad_aux=True``, else ``None``."""

    shift_loss: float | None = None
    """Shift-head BCE loss when the model has a shift head and labels are present."""


def _align_lengths(
    model_t: int,
    target_t: int,
) -> int:
    """Common length between model output and target (truncate to the shorter)."""
    return min(model_t, target_t)


def compute_loss(
    model_output_logits: dict[EventType, torch.Tensor],
    target_hazard: dict[EventType, torch.Tensor],
    target_mask: dict[EventType, torch.Tensor],
    *,
    event_weights: dict[EventType, float] | None = None,
    pos_weight: float = 1.0,
    vad_logits: torch.Tensor | None = None,
    vad_target: torch.Tensor | None = None,
    vad_loss_weight: float = 1.0,
    shift_logits: torch.Tensor | None = None,
    shift_target: torch.Tensor | None = None,
    shift_mask: torch.Tensor | None = None,
    shift_loss_weight: float = 1.0,
    shift_pos_weight: float = 1.0,
) -> tuple[torch.Tensor, TrainStepOutput]:
    """Multi-event survival NLL plus optional auxiliary VAD BCE.

    Args:
        model_output_logits: ``{ev: (B, T_model, K)}``.
        target_hazard: ``{ev: (B, T_target, K)}`` 0/1 labels.
        target_mask: ``{ev: (B, T_target, K)}`` observed-bin mask.
        event_weights: optional per-event scalar weights (default 1.0).
        pos_weight: positive-class weight for survival NLL (1.0 = unweighted).
        vad_logits: ``(B, T_model, 2)`` per-channel VAD logits, or None to skip
            the VAD auxiliary loss.
        vad_target: ``(B, T_target, 2)`` 0/1 VAD ground truth, required when
            ``vad_logits`` is given.
        vad_loss_weight: scalar weight on the VAD BCE term in the total loss.

    Returns:
        ``(total_loss, TrainStepOutput)``.
    """
    if not model_output_logits:
        raise ValueError("No event logits supplied")

    weights = event_weights or {ev: 1.0 for ev in model_output_logits}

    losses: list[torch.Tensor] = []
    per_event: dict[EventType, float] = {}
    n_observed: dict[EventType, int] = {}

    for ev, logits in model_output_logits.items():
        gt_h = target_hazard[ev]
        gt_m = target_mask[ev]
        # (B, T_model, K) vs (B, T_target, K) — truncate to common length
        common = _align_lengths(logits.size(1), gt_h.size(1))
        logits_a = logits[:, :common, :]
        gt_h_a = gt_h[:, :common, :]
        gt_m_a = gt_m[:, :common, :]

        loss_e = survival_nll_loss(
            logits_a, gt_h_a, gt_m_a, pos_weight=pos_weight, reduction="mean"
        )
        weighted = weights[ev] * loss_e
        losses.append(weighted)
        per_event[ev] = loss_e.detach().item()
        n_observed[ev] = int(gt_m_a.sum().item())

    survival_total = torch.stack(losses).mean()
    total = survival_total

    vad_loss_value: float | None = None
    if vad_logits is not None and vad_target is not None:
        common = _align_lengths(vad_logits.size(1), vad_target.size(1))
        vl = vad_logits[:, :common, :]
        vt = vad_target[:, :common, :].float()
        vad_bce = torch.nn.functional.binary_cross_entropy_with_logits(vl, vt)
        total = total + vad_loss_weight * vad_bce
        vad_loss_value = vad_bce.detach().item()

    shift_loss_value: float | None = None
    if shift_logits is not None and shift_target is not None and shift_mask is not None:
        common = _align_lengths(shift_logits.size(1), shift_target.size(1))
        sl = shift_logits[:, :common]
        st = shift_target[:, :common].float()
        sm = shift_mask[:, :common].float()
        if sm.sum() > 0:
            pos_w = torch.tensor(shift_pos_weight, device=sl.device, dtype=sl.dtype)
            elem = torch.nn.functional.binary_cross_entropy_with_logits(
                sl, st, pos_weight=pos_w, reduction="none"
            )
            shift_bce = (elem * sm).sum() / sm.sum().clamp(min=1.0)
            total = total + shift_loss_weight * shift_bce
            shift_loss_value = shift_bce.detach().item()

    return total, TrainStepOutput(
        total_loss=total.detach().item(),
        per_event_loss=per_event,
        n_observed=n_observed,
        vad_loss=vad_loss_value,
        shift_loss=shift_loss_value,
    )


def _vad_target_from_batch(batch: dict) -> torch.Tensor | None:
    """Pull the per-frame VAD target tensor out of a batch dict.

    :class:`itm.data.AMIDataset` populates ``batch["vad_target"]`` with a
    ``(B, T_target, 2)`` float tensor in {0, 1}. Returns None if absent
    (e.g. tests that build batches manually without VAD).
    """
    return batch.get("vad_target")


def train_step(
    model: ITMModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    *,
    event_weights: dict[EventType, float] | None = None,
    pos_weight: float = 1.0,
    use_vad_aux: bool = False,
    vad_loss_weight: float = 1.0,
    use_shift_head: bool = False,
    shift_loss_weight: float = 1.0,
    shift_pos_weight: float = 1.0,
    max_grad_norm: float | None = 1.0,
) -> TrainStepOutput:
    """Single training step: forward, loss, backward, step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    out = model(batch["audio"], return_vad=use_vad_aux)

    vad_logits = out.vad_logits if use_vad_aux else None
    vad_target = _vad_target_from_batch(batch) if use_vad_aux else None

    shift_logits = out.shift_logits if use_shift_head else None
    shift_target = batch.get("shift_target") if use_shift_head else None
    shift_mask = batch.get("shift_mask") if use_shift_head else None

    total_loss, info = compute_loss(
        out.hazard_logits,
        batch["hazard"],
        batch["mask"],
        event_weights=event_weights,
        pos_weight=pos_weight,
        vad_logits=vad_logits,
        vad_target=vad_target,
        vad_loss_weight=vad_loss_weight,
        shift_logits=shift_logits,
        shift_target=shift_target,
        shift_mask=shift_mask,
        shift_loss_weight=shift_loss_weight,
        shift_pos_weight=shift_pos_weight,
    )
    total_loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return info


@torch.no_grad()
def eval_step(
    model: ITMModel,
    batch: dict,
    *,
    pos_weight: float = 1.0,
    use_shift_head: bool = False,
    shift_loss_weight: float = 1.0,
    shift_pos_weight: float = 1.0,
) -> TrainStepOutput:
    """Single eval step (no grad, no optimizer)."""
    model.eval()
    out = model(batch["audio"])
    shift_logits = out.shift_logits if use_shift_head else None
    shift_target = batch.get("shift_target") if use_shift_head else None
    shift_mask = batch.get("shift_mask") if use_shift_head else None
    _, info = compute_loss(
        out.hazard_logits,
        batch["hazard"],
        batch["mask"],
        pos_weight=pos_weight,
        shift_logits=shift_logits,
        shift_target=shift_target,
        shift_mask=shift_mask,
        shift_loss_weight=shift_loss_weight,
        shift_pos_weight=shift_pos_weight,
    )
    return info

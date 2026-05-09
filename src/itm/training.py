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
) -> tuple[torch.Tensor, TrainStepOutput]:
    """Multi-event survival NLL.

    Args:
        model_output_logits: ``{ev: (B, T_model, K)}``.
        target_hazard: ``{ev: (B, T_target, K)}`` 0/1 labels.
        target_mask: ``{ev: (B, T_target, K)}`` observed-bin mask.
        event_weights: optional per-event scalar weights (default 1.0).

    Returns:
        ``(total_loss, TrainStepOutput)``. ``total_loss`` is the (weighted)
        mean over event types of per-event mean NLLs.
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

        loss_e = survival_nll_loss(logits_a, gt_h_a, gt_m_a, reduction="mean")
        weighted = weights[ev] * loss_e
        losses.append(weighted)
        per_event[ev] = loss_e.detach().item()
        n_observed[ev] = int(gt_m_a.sum().item())

    total = torch.stack(losses).mean()
    return total, TrainStepOutput(
        total_loss=total.detach().item(),
        per_event_loss=per_event,
        n_observed=n_observed,
    )


def train_step(
    model: ITMModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    *,
    event_weights: dict[EventType, float] | None = None,
    max_grad_norm: float | None = 1.0,
) -> TrainStepOutput:
    """Single training step: forward, loss, backward, step.

    Args:
        model: an :class:`ITMModel`.
        batch: from :func:`itm.data.ami_collate`.
        optimizer: torch optimizer.
        event_weights: per-event weight (defaults to 1.0).
        max_grad_norm: gradient clipping norm; None to disable.

    Returns:
        :class:`TrainStepOutput` with per-step loss diagnostics.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    out = model(batch["audio"])
    total_loss, info = compute_loss(
        out.hazard_logits,
        batch["hazard"],
        batch["mask"],
        event_weights=event_weights,
    )
    total_loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return info


@torch.no_grad()
def eval_step(model: ITMModel, batch: dict) -> TrainStepOutput:
    """Single eval step (no grad, no optimizer)."""
    model.eval()
    out = model(batch["audio"])
    _, info = compute_loss(out.hazard_logits, batch["hazard"], batch["mask"])
    return info

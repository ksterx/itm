"""Convert ITM event onsets to PyTorch survival hazard targets.

The output of ``itm.data.labels.survival_targets`` is a per-frame integer
target — the bin index (0..K-1) of the next event of each type, or -1 if
no event falls within the horizon (right-censored).

For training a discrete-time survival model we represent these as two
parallel tensors per event type:

* ``hazard_target[t, k]`` ∈ {0, 1} — 1 if an event occurs at bin k
  (relative to frame t), 0 otherwise. Only the bin containing the next
  event is positive.
* ``observed_mask[t, k]`` ∈ {0, 1} — 1 if bin k is observed (i.e. we
  haven't passed the event yet and we're still within the horizon).
  Used to mask the survival NLL.

This representation matches the discrete-time NLL described in
``docs/design/multi-event-hazard.md``::

    L_e(t) = - log h_e(t, k*) - sum_{j<k*} log(1 - h_e(t, j))     # event observed
           = - sum_{j=0}^{K-1} log(1 - h_e(t, j))                 # right-censored
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from itm.data.labels import EventType


def survival_to_tensors(
    targets: dict[EventType, list[int]],
    horizon_bins: int,
) -> dict[EventType, dict[str, torch.Tensor]]:
    """Convert per-frame target bins to (hazard, mask) tensors per event.

    Args:
        targets: from ``labels.survival_targets``. Keys: ``EventType``;
            values: list of length n_frames containing bin indices in
            [0, horizon_bins) or -1 for censored.
        horizon_bins: K, the number of future bins.

    Returns:
        For each event type, a dict with::

            {
                "hazard": Long[n_frames, horizon_bins]   ∈ {0, 1},
                "mask":   Float[n_frames, horizon_bins]  ∈ {0.0, 1.0},
            }
    """
    result: dict[EventType, dict[str, torch.Tensor]] = {}
    for ev_type, bin_list in targets.items():
        n_frames = len(bin_list)
        bins_t = torch.tensor(bin_list, dtype=torch.long)  # (n_frames,)

        hazard = torch.zeros(n_frames, horizon_bins, dtype=torch.long)
        mask = torch.zeros(n_frames, horizon_bins, dtype=torch.float32)

        # Censored frames (bin == -1): the entire horizon is "observed but no event"
        censored = bins_t == -1
        if censored.any():
            mask[censored] = 1.0  # all bins contribute to the survival term

        # Event frames: bin k_star marks the positive; bins 0..k_star inclusive are observed
        observed = ~censored
        if observed.any():
            ks = bins_t[observed]  # (m,)
            idxs = torch.arange(horizon_bins).unsqueeze(0)  # (1, K)
            ks_col = ks.unsqueeze(1)  # (m, 1)
            # mask: bins 0..k_star inclusive
            ev_mask = (idxs <= ks_col).to(torch.float32)
            mask[observed] = ev_mask
            # hazard: only k_star is 1
            ev_hazard = (idxs == ks_col).to(torch.long)
            hazard[observed] = ev_hazard

        result[ev_type] = {"hazard": hazard, "mask": mask}

    return result


def survival_nll_loss(
    hazard_logits: torch.Tensor,
    target_hazard: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-7,
    reduction: str = "mean",
) -> torch.Tensor:
    """Discrete-time survival NLL loss per (frame, bin) entry.

    Args:
        hazard_logits: Float[..., K] — pre-sigmoid logits.
        target_hazard: Long/Float[..., K] — 1 at the event bin, 0 elsewhere.
        mask: Float[..., K] — 1 where the bin contributes, 0 otherwise.
        reduction: ``"mean"`` over masked entries, ``"sum"``, or ``"none"``.

    Returns:
        Scalar (mean/sum reductions) or per-entry loss (none).
    """
    p = torch.sigmoid(hazard_logits)
    target_f = target_hazard.to(p.dtype)
    log_p = torch.log(p.clamp_min(eps))
    log_1mp = torch.log((1.0 - p).clamp_min(eps))
    # log h on event bin, log(1 - h) on observed-but-not-event bins
    per_entry = -(target_f * log_p + (1.0 - target_f) * log_1mp) * mask

    if reduction == "none":
        return per_entry
    total = per_entry.sum()
    if reduction == "sum":
        return total
    denom = mask.sum().clamp_min(1.0)
    return total / denom

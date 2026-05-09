"""Unit tests for ``itm.training``."""

from __future__ import annotations

import pytest
import torch

from itm.data.labels import EventType
from itm.training import compute_loss


def _logits_targets(t: int = 10, k: int = 5):
    """Helper: per-event logits, hazard, mask tensors."""
    logits = {ev: torch.zeros(2, t, k) for ev in EventType}
    hazard = {ev: torch.zeros(2, t, k, dtype=torch.long) for ev in EventType}
    mask = {ev: torch.ones(2, t, k) for ev in EventType}
    return logits, hazard, mask


class TestComputeLoss:
    def test_basic(self) -> None:
        logits, hazard, mask = _logits_targets()
        loss, info = compute_loss(logits, hazard, mask)
        # All zeros logits + zeros hazard + full mask: -log(0.5) ≈ 0.693 per entry
        assert pytest.approx(info.total_loss, abs=0.01) == 0.693
        assert set(info.per_event_loss) == set(EventType)

    def test_event_weights(self) -> None:
        logits, hazard, mask = _logits_targets()
        # Weight only TURN_SHIFT
        loss_uw, _ = compute_loss(logits, hazard, mask)
        loss_w, _ = compute_loss(
            logits,
            hazard,
            mask,
            event_weights={
                EventType.TURN_SHIFT: 2.0,
                EventType.BACKCHANNEL: 1.0,
                EventType.OVERLAP: 1.0,
            },
        )
        # weighted should be higher
        assert loss_w.item() > loss_uw.item()

    def test_length_mismatch_truncates(self) -> None:
        # Model output longer than target — should still compute on common length
        logits, _, _ = _logits_targets(t=15, k=5)
        _, hazard, mask = _logits_targets(t=10, k=5)
        loss, info = compute_loss(logits, hazard, mask)
        # Should use t=10 (the shorter)
        for ev in info.n_observed:
            assert info.n_observed[ev] == 2 * 10 * 5  # batch=2 × T × K

    def test_returns_finite(self) -> None:
        logits = {ev: torch.randn(2, 10, 5) * 5 for ev in EventType}
        _, hazard, mask = _logits_targets()
        loss, _ = compute_loss(logits, hazard, mask)
        assert torch.isfinite(loss)

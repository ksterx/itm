"""Tests for ``itm.data.targets``."""

from __future__ import annotations

import pytest
import torch

from itm.data.labels import EventType
from itm.data.targets import survival_nll_loss, survival_to_tensors


class TestSurvivalToTensors:
    def test_event_at_bin_3(self) -> None:
        # Single frame, one event type, event at bin 3 (out of K=5)
        targets = {EventType.TURN_SHIFT: [3]}
        out = survival_to_tensors(targets, horizon_bins=5)
        h = out[EventType.TURN_SHIFT]["hazard"]
        m = out[EventType.TURN_SHIFT]["mask"]
        assert h.shape == (1, 5)
        assert m.shape == (1, 5)
        # Hazard 1 only at bin 3
        assert h[0].tolist() == [0, 0, 0, 1, 0]
        # Mask: bins 0..3 inclusive (the survival NLL path)
        assert m[0].tolist() == [1, 1, 1, 1, 0]

    def test_censored_frame(self) -> None:
        # No event in window: bin == -1
        targets = {EventType.BACKCHANNEL: [-1]}
        out = survival_to_tensors(targets, horizon_bins=4)
        h = out[EventType.BACKCHANNEL]["hazard"]
        m = out[EventType.BACKCHANNEL]["mask"]
        # No positive
        assert h[0].tolist() == [0, 0, 0, 0]
        # All bins observed (we know no event happened across the full window)
        assert m[0].tolist() == [1, 1, 1, 1]

    def test_mixed_frames(self) -> None:
        # Three frames: event at bin 1, censored, event at bin 0
        targets = {EventType.OVERLAP: [1, -1, 0]}
        out = survival_to_tensors(targets, horizon_bins=3)
        h = out[EventType.OVERLAP]["hazard"]
        m = out[EventType.OVERLAP]["mask"]
        assert h.tolist() == [
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
        ]
        assert m.tolist() == [
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
        ]


class TestSurvivalNllLoss:
    def test_perfect_prediction_event_at_0(self) -> None:
        # Event at bin 0; perfect logit makes p=1 there → loss → 0
        target = torch.tensor([[1, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        # Very large positive logit at bin 0 → sigmoid≈1
        logits = torch.tensor([[10.0, 0.0, 0.0]])
        loss = survival_nll_loss(logits, target, mask)
        assert loss.item() < 1e-3

    def test_no_event_perfect_survival(self) -> None:
        # Censored: all bins should predict ≈0 hazard
        target = torch.tensor([[0, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1]], dtype=torch.float32)
        # Very large negative logits → sigmoid≈0
        logits = torch.tensor([[-10.0, -10.0, -10.0]])
        loss = survival_nll_loss(logits, target, mask)
        assert loss.item() < 1e-3

    def test_wrong_prediction_high_loss(self) -> None:
        # Event at bin 0 but model predicts ≈0 → log(0) → high loss
        target = torch.tensor([[1, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        logits = torch.tensor([[-10.0, 0.0, 0.0]])
        loss = survival_nll_loss(logits, target, mask)
        assert loss.item() > 5.0

    def test_mask_zeros_excluded(self) -> None:
        # Same logits, but only bin 0 contributes; bins 1-2 ignored
        logits = torch.tensor([[0.0, 100.0, -100.0]])
        target = torch.tensor([[1, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 0, 0]], dtype=torch.float32)
        loss = survival_nll_loss(logits, target, mask)
        # log(sigmoid(0)) = log(0.5) ≈ -0.693
        assert pytest.approx(loss.item(), abs=0.01) == 0.693

    def test_reduction_modes(self) -> None:
        target = torch.tensor([[1, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1]], dtype=torch.float32)
        logits = torch.zeros(1, 2)
        none_ = survival_nll_loss(logits, target, mask, reduction="none")
        assert none_.shape == (1, 2)
        sum_ = survival_nll_loss(logits, target, mask, reduction="sum")
        assert sum_.shape == ()
        mean_ = survival_nll_loss(logits, target, mask, reduction="mean")
        # mean over masked entries (2)
        assert pytest.approx(mean_.item(), abs=1e-5) == sum_.item() / 2.0

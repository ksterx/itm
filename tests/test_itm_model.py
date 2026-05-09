"""Unit tests for ``itm.models.itm_model``.

These tests use synthetic noise audio so they don't require AMI data,
but they DO require MaAI to be installed and the CPC checkpoint to be
downloadable. The first run will pull weights from the internet (~50 MB);
subsequent runs hit the local cache.

Marked ``slow`` for CI selection.
"""

from __future__ import annotations

import pytest
import torch

from itm.data.labels import EventType


@pytest.fixture(scope="module")
def itm_model():
    pytest.importorskip("maai")
    from itm.models import build_itm_model

    return build_itm_model(lang="en", frame_rate=20, context_len_sec=20)


@pytest.mark.slow
class TestITMModelForward:
    def test_output_shapes(self, itm_model) -> None:
        # 20s @ 16kHz = 320000 samples
        audio = torch.randn(2, 320000, 2) * 0.05
        itm_model.eval()
        with torch.no_grad():
            out = itm_model(audio)

        for ev in EventType:
            assert ev in out.hazard_logits
            shape = out.hazard_logits[ev].shape
            assert shape[0] == 2  # batch
            assert shape[2] == 40  # horizon_bins
            assert 950 < shape[1] < 1050  # encoder output ≈ 1000 frames at 50 Hz

    def test_param_counts(self, itm_model) -> None:
        counts = itm_model.count_parameters()
        # Sanity bounds: total under 15M, hazard heads small
        assert counts["total"] < 15_000_000
        assert counts["hazard_heads"] > 0
        # Encoder is frozen by default
        assert counts["trainable"] < counts["total"]

    def test_vad_optional(self, itm_model) -> None:
        audio = torch.randn(1, 320000, 2) * 0.05
        itm_model.eval()
        with torch.no_grad():
            out_no = itm_model(audio, return_vad=False)
            out_yes = itm_model(audio, return_vad=True)
        assert out_no.vad_logits is None
        assert out_yes.vad_logits is not None
        assert out_yes.vad_logits.shape[-1] == 2  # 2 speakers


@pytest.mark.slow
class TestITMModelBackward:
    def test_loss_backprop_runs(self, itm_model) -> None:
        from itm.data.targets import survival_nll_loss

        audio = torch.randn(2, 320000, 2, requires_grad=False) * 0.05
        itm_model.train()
        out = itm_model(audio)

        # Build dummy targets matching the model output length
        t_enc = next(iter(out.hazard_logits.values())).size(1)
        target_h = torch.zeros(2, t_enc, 40, dtype=torch.long)
        target_h[:, ::200, 5] = 1  # sparse positive bins
        mask = torch.ones(2, t_enc, 40)

        total = torch.zeros(())
        for ev in EventType:
            total = total + survival_nll_loss(out.hazard_logits[ev], target_h, mask)
        total.backward()

        # At least one trainable param should have a non-zero gradient
        any_grad = False
        for p in itm_model.parameters():
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
                any_grad = True
                break
        assert any_grad, "No trainable parameter received a gradient"

    def test_encoder_frozen(self, itm_model) -> None:
        # CPC encoder should NOT receive gradients (frozen by default)
        audio = torch.randn(1, 320000, 2) * 0.05
        itm_model.train()
        out = itm_model(audio)
        loss = sum(t.sum() for t in out.hazard_logits.values())
        loss.backward()

        for enc in (itm_model.backbone.encoder1, itm_model.backbone.encoder2):
            for p in enc.parameters():
                assert p.grad is None or p.grad.abs().sum() == 0


@pytest.mark.slow
class TestITMModelStateReset:
    def test_consecutive_forwards_consistent(self, itm_model) -> None:
        # Same audio twice → same hazard logits (because we reset state)
        audio = torch.randn(1, 320000, 2) * 0.05
        itm_model.eval()
        with torch.no_grad():
            out1 = itm_model(audio)
            out2 = itm_model(audio)
        for ev in EventType:
            assert torch.allclose(out1.hazard_logits[ev], out2.hazard_logits[ev], atol=1e-5)

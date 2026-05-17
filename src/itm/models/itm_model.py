"""ITM model: VAP backbone + multi-event hazard heads.

Architecture
------------

::

    audio (B, 2, T_samples)
        │
        ├─> ch1 ─┐
        │        │  CPC Encoder (frozen)        ── 50 Hz feature stream
        ├─> ch2 ─┤
        │        │  → x1, x2 (B, T_enc, dim)
        │        ▼
        ├─> ar_channel(x1), ar_channel(x2)
        │        │
        │        ▼
        ├─> ar(o1, o2)  — cross-attention transformer
        │        │
        │        ▼
        │   hidden state h ∈ (B, T_enc, dim)
        │        │
        │        ├─> ITM hazard head [turn_shift]   → (B, T_enc, K)
        │        ├─> ITM hazard head [backchannel]  → (B, T_enc, K)
        │        └─> ITM hazard head [overlap]      → (B, T_enc, K)
        │
        └─> [optional] VAD head (per channel)        → (B, T_enc, 1)

The CPC encoder is frozen by default (it ships pretrained from MaAI on
LibriSpeech). The transformer backbone (`ar_channel`, `ar`) is trainable
to adapt the cross-attended hidden state to the AMI domain. The 3 hazard
heads are randomly initialized and trained from scratch.

Caveats
-------

* The CPC encoder uses a stateful GRU (``keepHidden=True``); we reset it
  at the start of every forward pass for clean batch semantics.
* The encoder runs at ~50 Hz internally. If your targets are at a
  different rate, resample one of them before computing loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from itm.data.labels import EventType

# We use MaAI's VapGPT as the backbone. We import lazily to keep import-time
# cheap for users who don't need the model.


@dataclass
class ITMOutput:
    """Forward output of :class:`ITMModel`."""

    hazard_logits: dict[EventType, torch.Tensor]
    """Per-event hazard logits, shape ``(B, T_enc, horizon_bins)``."""

    vad_logits: torch.Tensor | None
    """Per-channel VAD logits ``(B, T_enc, 2)`` if ``return_vad`` else ``None``."""

    shift_logits: torch.Tensor | None
    """Per-frame shift logits ``(B, T_enc)`` if the model has a shift head."""

    encoder_frame_rate_hz: float
    """Effective output frame rate of the model (~50 Hz for MaAI VAP)."""


class ITMModel(nn.Module):
    """It's My Turn model: VAP backbone + 3 hazard heads.

    Args:
        vap_backbone: a configured ``maai.models.vap.VapGPT`` instance.
        horizon_bins: number of future bins per hazard head.
        event_types: list of :class:`itm.data.labels.EventType` for which to
            instantiate hazard heads.
        head_hidden: hidden dim of each hazard head MLP.
        freeze_encoder: if True (default), keeps the pretrained CPC encoders frozen.
        freeze_transformer: if True, also keeps the ar_channel / ar layers frozen.
            Default False — we want to adapt them on AMI.
    """

    def __init__(
        self,
        vap_backbone: nn.Module,
        *,
        horizon_bins: int = 40,
        event_types: list[EventType] | None = None,
        head_hidden: int = 128,
        freeze_encoder: bool = True,
        freeze_transformer: bool = False,
        enable_shift_head: bool = False,
        enable_visual: bool = False,
        visual_dim: int = 56,
        visual_hidden: int = 128,
    ) -> None:
        super().__init__()
        from itm.data.labels import EventType as _EventType

        self.backbone = vap_backbone
        self.horizon_bins = horizon_bins
        self._event_types = list(event_types) if event_types else list(_EventType)

        dim = self.backbone.conf.dim

        # Per-event hazard heads (MLP: Linear -> GELU -> Linear -> K logits)
        self.hazard_heads = nn.ModuleDict(
            {
                ev.value: nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, head_hidden),
                    nn.GELU(),
                    nn.Linear(head_hidden, horizon_bins),
                )
                for ev in self._event_types
            }
        )

        # Optional dedicated Shift head (binary, trained with BCE on silence frames).
        # Survival NLL alone has not produced useful shift discrimination
        # (v2/v3 measured ROC-AUC ≈ 0.5, baseline ≈ 0.7); a discriminative
        # head provides a direct gradient pathway for the eval task.
        self.shift_head: nn.Sequential | None
        if enable_shift_head:
            self.shift_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 1),
            )
        else:
            self.shift_head = None

        # Phase 3: optional visual encoder + late additive fusion.
        # Visual features (B, T_video, 2 speakers, visual_dim=56) come from
        # MediaPipe Face Landmarker (blendshapes + Euler + mouth open). The
        # encoder is a per-frame MLP that flattens both speakers and projects
        # to the backbone hidden dim, gated by a learnable scalar α so the
        # fusion can degrade gracefully if visual is unhelpful.
        self.visual_encoder: nn.Sequential | None
        self.visual_alpha: nn.Parameter | None
        if enable_visual:
            self.visual_encoder = nn.Sequential(
                nn.LayerNorm(visual_dim * 2),
                nn.Linear(visual_dim * 2, visual_hidden),
                nn.GELU(),
                nn.Linear(visual_hidden, dim),
            )
            # Start at 0 so initial behavior matches v6-α audio-only.
            self.visual_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.visual_encoder = None
            self.visual_alpha = None

        if freeze_encoder:
            for enc in (self.backbone.encoder1, self.backbone.encoder2):
                for p in enc.parameters():
                    p.requires_grad_(False)

        if freeze_transformer:
            for module_name in ("ar_channel", "ar", "vap_head", "va_classifier"):
                module = getattr(self.backbone, module_name, None)
                if module is None:
                    continue
                for p in module.parameters():
                    p.requires_grad_(False)

    # ----------------------------------------------------------------- helpers

    def reset_encoder_state(self) -> None:
        """Clear the stateful GRU hidden state in CPC encoders.

        Must be called before each batch forward to avoid cross-batch leakage.
        """
        for enc in (self.backbone.encoder1, self.backbone.encoder2):
            enc.encoder.gAR.hidden = None

    @property
    def event_types(self) -> list[EventType]:
        return list(self._event_types)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        audio: torch.Tensor,
        *,
        return_vad: bool = False,
        visual: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> ITMOutput:
        """Forward pass.

        Args:
            audio: ``(B, T_samples, 2)`` float32 tensor at 16 kHz.
            return_vad: if True, also return VAD logits per channel.
            visual: optional ``(B, T_video, 2, visual_dim)`` per-speaker
                MediaPipe features at video native rate (≈25 Hz). Only used
                when the model was built with ``enable_visual=True``.
            visual_mask: optional ``(B, T_video)`` float in {0, 1} marking
                video frames where a face was detected.

        Returns:
            :class:`ITMOutput` with per-event hazard logits.
        """
        if audio.ndim != 3 or audio.shape[-1] != 2:
            raise ValueError(f"audio must be (B, T, 2); got {tuple(audio.shape)}")

        # (B, T, 2) → two (B, 1, T) channels
        x1 = audio[..., 0].unsqueeze(1)
        x2 = audio[..., 1].unsqueeze(1)

        self.reset_encoder_state()
        e1, e2 = self.backbone.encode_audio(x1, x2)  # (B, T_enc, dim)

        o1 = self.backbone.ar_channel(e1, past_kv=None)
        o2 = self.backbone.ar_channel(e2, past_kv=None)
        out = self.backbone.ar(o1["x"], o2["x"])
        h = out["x"]  # (B, T_enc, dim) cross-attended hidden state

        # Phase 3: late additive visual fusion. Encode per-speaker features,
        # upsample from video rate (~25 Hz) to encoder rate (~50 Hz) via
        # nearest-neighbor interpolation, and add a scaled residual.
        if self.visual_encoder is not None and visual is not None:
            v_flat = visual.flatten(-2, -1)  # (B, T_video, 2 * visual_dim)
            v_enc = self.visual_encoder(v_flat)  # (B, T_video, dim)
            if visual_mask is not None:
                v_enc = v_enc * visual_mask.unsqueeze(-1)
            # Resize to encoder time axis (T_video → T_enc) with nearest-neighbor.
            v_enc_t = v_enc.transpose(1, 2)  # (B, dim, T_video)
            v_up = torch.nn.functional.interpolate(
                v_enc_t, size=h.size(1), mode="nearest"
            )
            v_up = v_up.transpose(1, 2)  # (B, T_enc, dim)
            assert self.visual_alpha is not None
            h = h + self.visual_alpha * v_up

        hazard_logits: dict[EventType, torch.Tensor] = {}
        for ev in self._event_types:
            hazard_logits[ev] = self.hazard_heads[ev.value](h)

        vad_logits: torch.Tensor | None = None
        if return_vad:
            vad1 = self.backbone.va_classifier(o1["x"]).squeeze(-1)  # (B, T_enc)
            vad2 = self.backbone.va_classifier(o2["x"]).squeeze(-1)
            vad_logits = torch.stack([vad1, vad2], dim=-1)  # (B, T_enc, 2)

        shift_logits: torch.Tensor | None = None
        if self.shift_head is not None:
            shift_logits = self.shift_head(h).squeeze(-1)  # (B, T_enc)

        # Estimate encoder frame rate from output length and audio duration
        sr_audio = audio.size(1)
        t_enc = h.size(1)
        # 16 kHz audio: total seconds = sr_audio / 16000
        seconds = sr_audio / 16000.0
        frame_rate = t_enc / seconds if seconds > 0 else 0.0

        return ITMOutput(
            hazard_logits=hazard_logits,
            vad_logits=vad_logits,
            shift_logits=shift_logits,
            encoder_frame_rate_hz=float(frame_rate),
        )

    # ----------------------------------------------------------- num parameters

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts for backbone, heads, and total."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.hazard_heads.parameters())
        encoder_params = sum(
            p.numel()
            for enc in (self.backbone.encoder1, self.backbone.encoder2)
            for p in enc.parameters()
        )
        return {
            "total": total,
            "trainable": trainable,
            "encoder": encoder_params,
            "hazard_heads": head_params,
            "transformer": total - encoder_params - head_params,
        }


# ---------------------------------------------------------------- builder


def build_itm_model(
    *,
    lang: str = "en",
    frame_rate: int = 20,
    context_len_sec: int = 20,
    cache_dir: str | None = None,
    horizon_bins: int = 40,
    head_hidden: int = 128,
    freeze_encoder: bool = True,
    freeze_transformer: bool = False,
    enable_shift_head: bool = False,
    enable_visual: bool = False,
    visual_dim: int = 56,
    device: str = "cpu",
) -> ITMModel:
    """Construct an :class:`ITMModel` initialized from a MaAI pretrained VAP.

    This loads a MaAI English / Japanese / Chinese / French / trilingual
    VapGPT checkpoint (see ``maai.util.get_available_models``), wraps it
    with :class:`ITMModel`, and returns it ready for training.
    """
    from maai.models.config import VapConfig
    from maai.models.vap import VapGPT
    from maai.util import load_vap_model

    # Build a fresh VapGPT and load weights via maai's helper
    cpc_path = Path.home() / ".cache" / "cpc" / "60k_epoch4-d0f474de.pt"
    conf = VapConfig()
    backbone = VapGPT(conf)
    backbone.load_encoder(cpc_model=str(cpc_path))

    sd = load_vap_model(
        mode="vap",
        frame_rate=frame_rate,
        context_len_sec=context_len_sec,
        language=lang,
        device=device,
        cache_dir=cache_dir,
        force_download=False,
    )
    backbone.load_state_dict(sd, strict=False)
    backbone.to(device)

    return ITMModel(
        backbone,
        horizon_bins=horizon_bins,
        head_hidden=head_hidden,
        freeze_encoder=freeze_encoder,
        freeze_transformer=freeze_transformer,
        enable_shift_head=enable_shift_head,
        enable_visual=enable_visual,
        visual_dim=visual_dim,
    )

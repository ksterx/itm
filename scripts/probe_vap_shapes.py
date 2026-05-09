"""Probe VAP model internals to understand input/output shapes for ITM wrapping."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
from maai import Maai, MaaiInput

# Build a Maai instance just to get a configured VapGPT model
print("Loading MaAI English VAP (10 Hz, 5 s context)...")

sf.write("/tmp/probe1.wav", np.zeros(16000 * 6, dtype=np.float32), 16000)
sf.write("/tmp/probe2.wav", np.zeros(16000 * 6, dtype=np.float32), 16000)

maai = Maai(
    mode="vap",
    lang="en",
    frame_rate=10,
    context_len_sec=5,
    audio_ch1=MaaiInput.Wav("/tmp/probe1.wav"),
    audio_ch2=MaaiInput.Wav("/tmp/probe2.wav"),
    device="cpu",
)
vap = maai.vap
vap.eval()

print(f"\nVapGPT class: {type(vap).__name__}")
print(f"conf.dim: {vap.conf.dim}")
print(f"objective.n_classes: {vap.objective.n_classes}")
print(f"frame_contxt_padding (from Maai): {maai.frame_contxt_padding}")
print(f"audio_frame_size (from Maai): {maai.audio_frame_size}")


def reset_encoder_state(vap_model: torch.nn.Module) -> None:
    """Clear stateful GRU hidden in the CPC encoder for clean batch forward."""
    for enc in (vap_model.encoder1, vap_model.encoder2):
        enc.encoder.gAR.hidden = None


# Try encoding 20s chunk (320000 samples) — this is what we want for training
x1_long = torch.zeros(1, 1, 320000)
x2_long = torch.zeros(1, 1, 320000)
reset_encoder_state(vap)
with torch.no_grad():
    e1_long, e2_long = vap.encode_audio(x1_long, x2_long)
print(f"encode_audio(320000 samples) → e1: {tuple(e1_long.shape)}")

# Now feed encoded features to ar_channel + ar to see hidden state shape
with torch.no_grad():
    o1 = vap.ar_channel(e1_long, past_kv=None)
    o2 = vap.ar_channel(e2_long, past_kv=None)
    out = vap.ar(o1["x"], o2["x"])
    print(f"ar_channel(o1['x']): {tuple(o1['x'].shape)}")
    print(f"ar(out['x']): {tuple(out['x'].shape)}")
    vad1 = vap.va_classifier(o1["x"])
    print(f"va_classifier(vad1): {tuple(vad1.shape)}")
    vap_logits = vap.vap_head(out["x"])
    print(f"vap_head(logits): {tuple(vap_logits.shape)}")

# What if we batch?
xb = torch.zeros(2, 1, 320000)
reset_encoder_state(vap)
with torch.no_grad():
    eb, _ = vap.encode_audio(xb, xb)
    print(f"\nBatch test: encode_audio(B=2, 1, 320000) → {tuple(eb.shape)}")

"""Audio loading utilities for AMI Corpus.

Each AMI meeting has four per-speaker headset recordings::

    data/raw/ami/<meeting_id>/audio/<meeting_id>.Headset-{0,1,2,3}.wav

Speakers A/B/C/D map to channels 0/1/2/3. This module loads the two
selected speaker channels as a synchronized ``(n_samples, 2)`` float32
array, ready to feed into VAP-style models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLING_RATE = 16000
SPEAKER_TO_CHANNEL = {"A": 0, "B": 1, "C": 2, "D": 3}


def headset_path(audio_dir: Path | str, meeting_id: str, speaker: str) -> Path:
    """Return the headset wav path for a given meeting + speaker letter."""
    ch = SPEAKER_TO_CHANNEL[speaker]
    return Path(audio_dir) / f"{meeting_id}.Headset-{ch}.wav"


def load_two_channel_audio(
    audio_dir: Path | str,
    meeting_id: str,
    speakers: tuple[str, str],
) -> tuple[np.ndarray, float]:
    """Load two synchronized speaker channels from an AMI meeting.

    Args:
        audio_dir: e.g. ``data/raw/ami/ES2002a/audio``.
        meeting_id: e.g. ``"ES2002a"``.
        speakers: 2-tuple of speaker letters (e.g. ``("B", "D")``).

    Returns:
        ``(audio, duration_sec)`` where ``audio`` has shape
        ``(min(len_ch1, len_ch2), 2)`` and dtype ``float32``.

    Raises:
        FileNotFoundError: If a wav file is missing.
        ValueError: If sample rates differ from 16 kHz.
    """
    audio_dir = Path(audio_dir)
    paths = [headset_path(audio_dir, meeting_id, spk) for spk in speakers]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing wav: {p}")

    arrays: list[np.ndarray] = []
    for p in paths:
        info = sf.info(p)
        if info.samplerate != SAMPLING_RATE:
            raise ValueError(
                f"{p.name} has sample rate {info.samplerate}, expected {SAMPLING_RATE}"
            )
        data, _ = sf.read(p, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]  # mono safety
        arrays.append(data)

    n = min(len(arrays[0]), len(arrays[1]))
    audio = np.stack([arrays[0][:n], arrays[1][:n]], axis=1)
    duration = n / SAMPLING_RATE
    return audio, duration


def slice_chunk(
    audio: np.ndarray,
    start_sec: float,
    chunk_sec: float,
    sampling_rate: int = SAMPLING_RATE,
    pad: bool = True,
) -> np.ndarray:
    """Extract ``chunk_sec`` worth of audio starting at ``start_sec``.

    If the requested window extends past the end of ``audio`` and ``pad`` is
    True, the result is right-padded with zeros to ``int(chunk_sec * sr)``
    samples. Otherwise the truncated slice is returned.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"audio must be (n_samples, 2), got {audio.shape}")

    i_start = int(start_sec * sampling_rate)
    n_target = int(chunk_sec * sampling_rate)
    i_end = i_start + n_target

    chunk = audio[i_start : min(i_end, len(audio))]
    if pad and len(chunk) < n_target:
        pad_len = n_target - len(chunk)
        chunk = np.concatenate([chunk, np.zeros((pad_len, 2), dtype=chunk.dtype)], axis=0)
    return chunk

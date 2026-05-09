"""Tests for ``itm.data.audio``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from itm.data.audio import (
    SAMPLING_RATE,
    headset_path,
    load_two_channel_audio,
    slice_chunk,
)


def _write_wav(path: Path, duration_sec: float, freq: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_sec * SAMPLING_RATE)
    t = np.arange(n) / SAMPLING_RATE
    sig = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, sig, SAMPLING_RATE)


@pytest.fixture()
def fake_audio_dir(tmp_path: Path) -> Path:
    audio_dir = tmp_path / "ES2002a" / "audio"
    _write_wav(audio_dir / "ES2002a.Headset-0.wav", duration_sec=5.0, freq=200.0)
    _write_wav(audio_dir / "ES2002a.Headset-1.wav", duration_sec=5.0, freq=400.0)
    _write_wav(audio_dir / "ES2002a.Headset-2.wav", duration_sec=5.0, freq=600.0)
    _write_wav(audio_dir / "ES2002a.Headset-3.wav", duration_sec=5.0, freq=800.0)
    return audio_dir


class TestHeadsetPath:
    def test_a_to_channel_0(self, tmp_path: Path) -> None:
        p = headset_path(tmp_path, "ES2002a", "A")
        assert p.name == "ES2002a.Headset-0.wav"

    def test_d_to_channel_3(self, tmp_path: Path) -> None:
        p = headset_path(tmp_path, "IS1000a", "D")
        assert p.name == "IS1000a.Headset-3.wav"


class TestLoadTwoChannel:
    def test_basic(self, fake_audio_dir: Path) -> None:
        audio, dur = load_two_channel_audio(fake_audio_dir, "ES2002a", ("A", "B"))
        assert audio.shape == (5 * SAMPLING_RATE, 2)
        assert audio.dtype == np.float32
        assert dur == pytest.approx(5.0)

    def test_picks_correct_channels(self, fake_audio_dir: Path) -> None:
        audio_ab, _ = load_two_channel_audio(fake_audio_dir, "ES2002a", ("A", "B"))
        audio_cd, _ = load_two_channel_audio(fake_audio_dir, "ES2002a", ("C", "D"))
        # Channels should differ (different frequencies)
        assert not np.allclose(audio_ab[:, 0], audio_cd[:, 0])
        assert not np.allclose(audio_ab[:, 1], audio_cd[:, 1])

    def test_missing_wav_raises(self, fake_audio_dir: Path, tmp_path: Path) -> None:
        # Remove one wav
        (fake_audio_dir / "ES2002a.Headset-0.wav").unlink()
        with pytest.raises(FileNotFoundError):
            load_two_channel_audio(fake_audio_dir, "ES2002a", ("A", "B"))


class TestSliceChunk:
    def test_pad_extends_short_window(self) -> None:
        audio = np.ones((5 * SAMPLING_RATE, 2), dtype=np.float32)
        # Slice last 4s starting at 3s of a 5s recording → 4s window pads 2s
        chunk = slice_chunk(audio, start_sec=3.0, chunk_sec=4.0, pad=True)
        assert chunk.shape == (4 * SAMPLING_RATE, 2)
        assert (chunk[: 2 * SAMPLING_RATE] == 1.0).all()
        assert (chunk[2 * SAMPLING_RATE :] == 0.0).all()

    def test_no_pad_truncates(self) -> None:
        audio = np.ones((5 * SAMPLING_RATE, 2), dtype=np.float32)
        chunk = slice_chunk(audio, start_sec=3.0, chunk_sec=4.0, pad=False)
        assert chunk.shape == (2 * SAMPLING_RATE, 2)

    def test_full_window_no_pad_needed(self) -> None:
        audio = np.ones((10 * SAMPLING_RATE, 2), dtype=np.float32)
        chunk = slice_chunk(audio, start_sec=2.0, chunk_sec=3.0, pad=True)
        assert chunk.shape == (3 * SAMPLING_RATE, 2)
        assert (chunk == 1.0).all()

    def test_wrong_shape_raises(self) -> None:
        audio_1d = np.zeros((1000,), dtype=np.float32)
        with pytest.raises(ValueError):
            slice_chunk(audio_1d, start_sec=0.0, chunk_sec=1.0)

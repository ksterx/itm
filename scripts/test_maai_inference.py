"""Smoke test for MaAI: load English VAP model and run inference on a synthetic stereo wav."""

import time
from pathlib import Path

import numpy as np
import soundfile as sf
from maai import Maai, MaaiInput

REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = REPO_ROOT / "tmp"
TMP_DIR.mkdir(exist_ok=True)


def make_synth_wav(path: Path, duration_sec: float, sr: int = 16000, silent: bool = False) -> None:
    """Create a 1ch wav. If silent, all zeros; else alternating speech-like burst and silence."""
    if silent:
        sf.write(path, np.zeros(int(sr * duration_sec), dtype=np.float32), sr)
        return
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    envelope = np.sin(2 * np.pi * 5 * t) ** 2
    carrier = np.sin(2 * np.pi * 200 * t)
    signal = carrier * envelope * 0.3
    signal[len(signal) // 2 :] = 0  # silence in second half
    sf.write(path, signal.astype(np.float32), sr)


def main() -> None:
    # 10s of audio, 5s context model — should produce results after ~5s
    duration = 10.0
    wav1 = TMP_DIR / "synth_speaker_a_10s.wav"
    wav2 = TMP_DIR / "synth_speaker_b_silent_10s.wav"
    make_synth_wav(wav1, duration, silent=False)
    make_synth_wav(wav2, duration, silent=True)

    print("Loading MaAI English VAP (10Hz, 5s context)...")
    t0 = time.time()
    maai = Maai(
        mode="vap",
        lang="en",
        frame_rate=10,
        context_len_sec=5,
        audio_ch1=MaaiInput.Wav(str(wav1)),
        audio_ch2=MaaiInput.Wav(str(wav2)),
        device="cpu",
    )
    print(f"  loaded in {time.time() - t0:.2f}s")

    print("Running inference for 12s wall clock (audio is 10s)...")
    t0 = time.time()
    maai.start()

    # Use the underlying queue with timeout to avoid blocking forever
    result_q = maai.result_dict_queue
    deadline = t0 + 12.0
    results: list = []
    while time.time() < deadline:
        try:
            r = result_q.get(timeout=0.5)
        except Exception:
            continue
        if r is not None:
            results.append(r)

    elapsed = time.time() - t0
    print(
        f"Got {len(results)} result frames in {elapsed:.2f}s "
        f"(rate ≈ {len(results) / elapsed:.1f}/s)"
    )
    if results:
        first = results[0]
        print(f"Result type: {type(first).__name__}")
        if isinstance(first, dict):
            print(f"Result keys: {list(first.keys())}")
            for k, v in first.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    s = repr(v)
                    print(f"  {k}: {s[:120]}")
        print()
        print(f"Last result snippet: {repr(results[-1])[:300]}")


if __name__ == "__main__":
    main()

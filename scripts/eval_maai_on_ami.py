"""Evaluate MaAI English VAP on a single AMI meeting.

This is the Phase 1 baseline: pick the two most active speakers in a
meeting, run MaAI VAP on their headset channels, and compute:

1. **Frame-level VAD agreement** — does MaAI's argmax(p_now) match the
   ground-truth speaker who is actually talking at that frame?
2. **Hold/shift accuracy** — at each mutual-silence boundary (≥ 200 ms
   of mutual silence), did MaAI's ``p_future`` correctly predict whether
   the next active speaker is the same (hold) or different (shift)?

Hold/shift accuracy at mutual silences is the canonical VAP evaluation
metric used by Ekstedt & Skantze (2022) — typical published numbers are
in the 75–85% range on Switchboard / CANDOR.

Usage::

    python scripts/eval_maai_on_ami.py [MEETING_ID]

Defaults to ES2002a. Requires:
- AMI annotations unpacked at ``data/raw/ami/annotations/unpacked``
- AMI headset audio at ``data/raw/ami/<meeting>/audio/<meeting>.Headset-{0..3}.wav``
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from maai import Maai, MaaiInput

from itm.data import Segment, load_meeting

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOT_ROOT = REPO_ROOT / "data" / "raw" / "ami" / "annotations" / "unpacked"
AUDIO_ROOT = REPO_ROOT / "data" / "raw" / "ami"

# MaAI English VAP supports 5/10/12.5/20 Hz only (no 50 Hz pretrained ckpt).
# 20 Hz is the highest available frame rate.
FRAME_RATE_HZ = 20
MUTUAL_SILENCE_THRESHOLD_SEC = 0.2

# Speaker label → headset channel index. AMI A→0, B→1, C→2, D→3 by convention.
SPEAKER_TO_CHANNEL = {"A": 0, "B": 1, "C": 2, "D": 3}


@dataclass
class FrameResult:
    """One frame of MaAI output, normalized to the two evaluated speakers."""

    t: float
    p_now: tuple[float, float]  # (speaker1, speaker2)
    p_future: tuple[float, float]
    vad: tuple[float, float]


# ---------------------------------------------------------------------------
# Speaker selection
# ---------------------------------------------------------------------------


def pick_two_most_active_speakers(segments_by_speaker: dict[str, list[Segment]]) -> tuple[str, str]:
    talk_time = {spk: sum(s.duration for s in segs) for spk, segs in segments_by_speaker.items()}
    if len(talk_time) < 2:
        raise ValueError(f"Need at least 2 speakers, got: {list(talk_time)}")
    ranked = sorted(talk_time.items(), key=lambda kv: -kv[1])
    return ranked[0][0], ranked[1][0]


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------


def build_vad_arrays(
    segments_by_speaker: dict[str, list[Segment]],
    speakers: tuple[str, str],
    duration_sec: float,
    frame_rate: int,
) -> np.ndarray:
    """Return a (n_frames, 2) bool array — 1 where the speaker is talking.

    Computed at ``frame_rate`` Hz from segment start/end times.
    """
    n_frames = int(duration_sec * frame_rate)
    vad = np.zeros((n_frames, 2), dtype=bool)
    dt = 1.0 / frame_rate
    for ch, spk in enumerate(speakers):
        for seg in segments_by_speaker.get(spk, []):
            i_start = max(0, int(seg.start / dt))
            i_end = min(n_frames, int(seg.end / dt) + 1)
            vad[i_start:i_end, ch] = True
    return vad


def find_mutual_silences(
    vad: np.ndarray,
    frame_rate: int,
    min_duration_sec: float = MUTUAL_SILENCE_THRESHOLD_SEC,
) -> list[tuple[int, int]]:
    """Return contiguous frame ranges (i_start, i_end) where both speakers are silent.

    Endpoints in frames; `[i_start, i_end)`.
    """
    silent = ~(vad[:, 0] | vad[:, 1])
    min_frames = int(min_duration_sec * frame_rate)

    ranges: list[tuple[int, int]] = []
    i = 0
    n = len(silent)
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        if (j - i) >= min_frames:
            ranges.append((i, j))
        i = j
    return ranges


def classify_hold_shift(
    vad: np.ndarray, silence_start: int, silence_end: int
) -> tuple[int | None, int | None, str | None]:
    """Identify the speaker before and after a mutual silence.

    Returns (prev_speaker_ch, next_speaker_ch, label) where label is "hold"
    or "shift". Returns (None, None, None) if either side is empty (silence
    at the very start or end of the recording).
    """
    prev_ch = None
    for i in range(silence_start - 1, -1, -1):
        if vad[i, 0] and not vad[i, 1]:
            prev_ch = 0
            break
        if vad[i, 1] and not vad[i, 0]:
            prev_ch = 1
            break
        if vad[i, 0] and vad[i, 1]:
            # Overlap right before silence — skip
            continue

    next_ch = None
    n = len(vad)
    for i in range(silence_end, n):
        if vad[i, 0] and not vad[i, 1]:
            next_ch = 0
            break
        if vad[i, 1] and not vad[i, 0]:
            next_ch = 1
            break

    if prev_ch is None or next_ch is None:
        return prev_ch, next_ch, None
    return prev_ch, next_ch, ("hold" if prev_ch == next_ch else "shift")


# ---------------------------------------------------------------------------
# MaAI driver
# ---------------------------------------------------------------------------


def run_maai_inference(
    wav1: Path,
    wav2: Path,
    *,
    duration_sec: float,
    frame_rate: int,
) -> list[FrameResult]:
    """Stream both wavs through MaAI English VAP and collect per-frame results."""
    print(f"Loading MaAI English VAP ({frame_rate} Hz, 20 s context)...")
    t0 = time.time()
    maai = Maai(
        mode="vap",
        lang="en",
        frame_rate=frame_rate,
        context_len_sec=20,
        audio_ch1=MaaiInput.Wav(str(wav1)),
        audio_ch2=MaaiInput.Wav(str(wav2)),
        device="cpu",
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    maai.start()
    queue = maai.result_dict_queue

    results: list[FrameResult] = []
    deadline = time.time() + duration_sec * 1.2 + 30  # generous timeout
    last_log = time.time()
    while time.time() < deadline:
        try:
            r = queue.get(timeout=0.5)
        except Exception:
            # Stop when audio file is exhausted (queue has been idle for a while)
            if results and (time.time() - results[-1].t > 2.0 or queue.empty()):
                break
            continue
        if r is None:
            continue
        results.append(
            FrameResult(
                t=float(r["t"]),
                p_now=(float(r["p_now"][0]), float(r["p_now"][1])),
                p_future=(float(r["p_future"][0]), float(r["p_future"][1])),
                vad=(float(r["vad"][0]), float(r["vad"][1])),
            )
        )
        if time.time() - last_log > 10:
            print(f"  collected {len(results)} frames...")
            last_log = time.time()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    meeting_id = sys.argv[1] if len(sys.argv) > 1 else "ES2002a"

    if not ANNOT_ROOT.is_dir():
        sys.exit(
            f"AMI annotations not found at {ANNOT_ROOT}. "
            "Run: python scripts/download_ami_subset.py --annotations-only"
        )
    audio_dir = AUDIO_ROOT / meeting_id / "audio"
    if not audio_dir.is_dir():
        sys.exit(
            f"AMI audio not found at {audio_dir}. "
            f"Run: python scripts/download_ami_subset.py --meetings {meeting_id} --audio-only"
        )

    # 1. Load annotations and pick 2 most active speakers
    print(f"=== {meeting_id} ===")
    meeting = load_meeting(ANNOT_ROOT, meeting_id)
    spk1, spk2 = pick_two_most_active_speakers(meeting.segments_by_speaker)
    talk = {s: sum(seg.duration for seg in segs) for s, segs in meeting.segments_by_speaker.items()}
    print(f"Speakers (talk time s): {[(s, round(talk[s], 1)) for s in meeting.speakers]}")
    print(f"Evaluating channels: ch1={spk1}, ch2={spk2}")

    # 2. Resolve wav paths and read durations
    wav1 = audio_dir / f"{meeting_id}.Headset-{SPEAKER_TO_CHANNEL[spk1]}.wav"
    wav2 = audio_dir / f"{meeting_id}.Headset-{SPEAKER_TO_CHANNEL[spk2]}.wav"
    if not wav1.exists() or not wav2.exists():
        sys.exit(f"Missing wav: {wav1} or {wav2}")
    info1 = sf.info(wav1)
    info2 = sf.info(wav2)
    duration_sec = min(info1.duration, info2.duration)
    print(
        f"Audio: ch1={info1.duration:.1f}s ({info1.samplerate} Hz), "
        f"ch2={info2.duration:.1f}s ({info2.samplerate} Hz)"
    )

    # 3. Build ground truth VAD at FRAME_RATE_HZ
    gt_vad = build_vad_arrays(
        meeting.segments_by_speaker, (spk1, spk2), duration_sec, FRAME_RATE_HZ
    )
    print(
        f"Ground truth: {gt_vad.sum(axis=0)} frames per speaker "
        f"(out of {len(gt_vad)} total @ {FRAME_RATE_HZ} Hz)"
    )

    # 4. Run MaAI inference (one frame per 1/frame_rate sec, indexed in-order)
    results = run_maai_inference(wav1, wav2, duration_sec=duration_sec, frame_rate=FRAME_RATE_HZ)
    print(
        f"MaAI produced {len(results)} frames over {duration_sec:.0f}s "
        f"(expected ~{int(duration_sec * FRAME_RATE_HZ)})"
    )

    if not results:
        sys.exit("No MaAI frames collected; check input/audio.")

    # MaAI emits in real-time order; align by index, truncate to overlap
    n = min(len(results), len(gt_vad))
    p_now = np.array([r.p_now for r in results[:n]])
    p_future = np.array([r.p_future for r in results[:n]])
    gt = gt_vad[:n]

    # 5. Frame-level VAD agreement
    # (Only count frames where exactly one speaker is active in GT)
    single_speaker_mask = gt[:, 0] ^ gt[:, 1]
    if single_speaker_mask.sum() > 0:
        gt_speaker = gt[:, 1].astype(int)  # 0 if A active, 1 if B active
        pred_speaker = (p_now[:, 1] > p_now[:, 0]).astype(int)
        frame_acc = (gt_speaker[single_speaker_mask] == pred_speaker[single_speaker_mask]).mean()
    else:
        frame_acc = float("nan")

    # 6. Hold/shift accuracy at mutual-silence boundaries
    silences = find_mutual_silences(gt, FRAME_RATE_HZ)
    print(f"\nFound {len(silences)} mutual-silence regions (≥{MUTUAL_SILENCE_THRESHOLD_SEC}s)")

    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for i_s, i_e in silences:
        prev_ch, next_ch, label = classify_hold_shift(gt, i_s, i_e)
        if label is None:
            continue
        # Use p_future at the silence midpoint
        mid = (i_s + i_e) // 2
        if mid >= n:
            continue
        # MaAI's p_future is who will speak in the near future given the buffer
        pred_next_ch = 1 if p_future[mid, 1] > p_future[mid, 0] else 0
        pred_label = "hold" if pred_next_ch == prev_ch else "shift"
        counts[label] += 1
        if pred_label == label:
            correct[label] += 1
        confusion[(label, pred_label)] += 1

    # 7. Report
    print("\n=== Results ===")
    print(f"Frame-level VAD argmax accuracy (single-speaker frames): {frame_acc:.3f}")
    total_eval = sum(counts.values())
    print(f"\nHold/shift evaluation: {total_eval} eligible silences")
    for label in ["hold", "shift"]:
        if counts[label]:
            acc = correct[label] / counts[label]
            print(f"  {label}: {correct[label]}/{counts[label]} = {acc:.3f}")
    if total_eval:
        overall = sum(correct.values()) / total_eval
        print(f"  overall: {sum(correct.values())}/{total_eval} = {overall:.3f}")
        print("\n  Confusion (gt → pred):")
        for gt_lbl in ["hold", "shift"]:
            for pred_lbl in ["hold", "shift"]:
                n_ = confusion.get((gt_lbl, pred_lbl), 0)
                print(f"    {gt_lbl} → {pred_lbl}: {n_}")


if __name__ == "__main__":
    main()

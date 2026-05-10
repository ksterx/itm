"""Evaluate MaAI English VAP on one or more AMI meetings.

Phase 1 baseline: pick the two most active speakers in each meeting,
run MaAI VAP on their headset channels, and compute:

1. **Frame-level VAD agreement** — argmax(p_now) vs ground-truth speaker
2. **Hold/shift accuracy** — at mutual-silence boundaries (≥ 200 ms gap),
   does p_future correctly predict same speaker (hold) or different (shift)?

Hold/shift accuracy at mutual silences is the canonical VAP evaluation
metric used by Ekstedt & Skantze (2022) — typical published numbers are
in the 75–85% range on Switchboard / CANDOR.

Usage::

    python scripts/eval_maai_on_ami.py [MEETING_ID ...]
    python scripts/eval_maai_on_ami.py --all              # ES2002a/b/c, IS1000a/b
    python scripts/eval_maai_on_ami.py --fast ES2002a     # skip real-time playback (~4x speedup)

Defaults to ES2002a. Requires:
- AMI annotations unpacked at ``data/raw/ami/annotations/unpacked``
- AMI headset audio at ``data/raw/ami/<meeting>/audio/<meeting>.Headset-{0..3}.wav``
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from maai import Maai, MaaiInput
from maai.input import Wav as _MaaiWav

from itm.data import Segment, load_meeting

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOT_ROOT = REPO_ROOT / "data" / "raw" / "ami" / "annotations" / "unpacked"
AUDIO_ROOT = REPO_ROOT / "data" / "raw" / "ami"

# MaAI English VAP supports 5/10/12.5/20 Hz only (no 50 Hz pretrained ckpt).
FRAME_RATE_HZ = 20
MUTUAL_SILENCE_THRESHOLD_SEC = 0.2

SPEAKER_TO_CHANNEL = {"A": 0, "B": 1, "C": 2, "D": 3}

DEFAULT_ALL_MEETINGS = ["ES2002a", "ES2002b", "ES2002c", "IS1000a", "IS1000b"]


# ---------------------------------------------------------------------------
# Fast Wav input — skips real-time playback timing for ~4x speedup
# ---------------------------------------------------------------------------


class FastWav(_MaaiWav):
    """Drop-in for ``MaaiInput.Wav`` that emits frames as fast as MaAI can consume.

    .. warning::
       Empirically produces *different* outputs than real-time playback on the
       same audio (validated on ES2002a: real-time 58.7% vs fast 73.4%
       hold/shift accuracy). Root cause not yet pinned down — likely interacts
       with MaAI's internal buffer trim or the model worker's iteration cadence.
       Use only for exploratory runs; ``--fast`` results are not canonical.

    MaAI's worker drops queued audio whenever its internal queue exceeds 100 frames
    (a safety against runaway input). We throttle to stay below that bound.
    """

    QUEUE_CAP = 80  # leave headroom under MaAI's 100-frame overflow threshold

    def _read_wav(self):  # noqa: D401  — keep parent signature
        while not self.raw_wav_queue.empty():
            # Backpressure: wait until subscribers have drained below cap.
            while True:
                with self._lock:
                    max_depth = max((q.qsize() for q in self._subscriber_queues), default=0)
                if max_depth < self.QUEUE_CAP:
                    break
                time.sleep(0.005)
            data = self.raw_wav_queue.get()
            if data is None:
                continue
            self._put_to_all_queues(data)

    def start(self):  # noqa: D401
        if not self._is_thread_started:
            threading.Thread(target=self._read_wav, daemon=True).start()
            self._is_thread_started = True


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FrameResult:
    t: float
    p_now: tuple[float, float]
    p_future: tuple[float, float]
    vad: tuple[float, float]


@dataclass
class MeetingResult:
    meeting_id: str
    duration_sec: float
    frame_acc: float
    hold_correct: int
    hold_total: int
    shift_correct: int
    shift_total: int
    confusion: dict[tuple[str, str], int]
    score_pairs: list[tuple[str, float]] | None = None  # (gt_label, shift_score) per silence

    @property
    def overall_correct(self) -> int:
        return self.hold_correct + self.shift_correct

    @property
    def overall_total(self) -> int:
        return self.hold_total + self.shift_total

    @property
    def overall_acc(self) -> float:
        if self.overall_total == 0:
            return float("nan")
        return self.overall_correct / self.overall_total


# ---------------------------------------------------------------------------
# Speaker selection / GT helpers
# ---------------------------------------------------------------------------


def pick_two_most_active_speakers(segments_by_speaker: dict[str, list[Segment]]) -> tuple[str, str]:
    talk_time = {spk: sum(s.duration for s in segs) for spk, segs in segments_by_speaker.items()}
    if len(talk_time) < 2:
        raise ValueError(f"Need at least 2 speakers, got: {list(talk_time)}")
    ranked = sorted(talk_time.items(), key=lambda kv: -kv[1])
    return ranked[0][0], ranked[1][0]


def build_vad_arrays(
    segments_by_speaker: dict[str, list[Segment]],
    speakers: tuple[str, str],
    duration_sec: float,
    frame_rate: int,
) -> np.ndarray:
    """Return a (n_frames, 2) bool array — True where the speaker is talking."""
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
    """Identify the speaker before and after a mutual silence."""
    prev_ch = None
    for i in range(silence_start - 1, -1, -1):
        if vad[i, 0] and not vad[i, 1]:
            prev_ch = 0
            break
        if vad[i, 1] and not vad[i, 0]:
            prev_ch = 1
            break

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
# MaAI inference driver
# ---------------------------------------------------------------------------


def _make_input(wav_path: Path, fast: bool) -> _MaaiWav:
    return FastWav(str(wav_path)) if fast else MaaiInput.Wav(str(wav_path))


def run_maai_inference(
    wav1: Path,
    wav2: Path,
    *,
    duration_sec: float,
    frame_rate: int,
    fast: bool,
) -> list[FrameResult]:
    """Stream both wavs through MaAI English VAP and collect per-frame results."""
    print(f"Loading MaAI English VAP ({frame_rate} Hz, 20 s context, fast={fast})...")
    t0 = time.time()
    maai = Maai(
        mode="vap",
        lang="en",
        frame_rate=frame_rate,
        context_len_sec=20,
        audio_ch1=_make_input(wav1, fast),
        audio_ch2=_make_input(wav2, fast),
        device="cpu",
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    maai.start()
    queue = maai.result_dict_queue

    # Generous deadline; in fast mode the bottleneck is inference (~12ms/frame).
    rate_factor = 0.4 if fast else 1.2
    deadline = time.time() + duration_sec * rate_factor + 60
    last_log = time.time()
    last_size = 0
    stagnant_since = time.time()

    results: list[FrameResult] = []
    while time.time() < deadline:
        try:
            r = queue.get(timeout=0.5)
        except Exception:
            r = None
        if r is None:
            # Detect end-of-stream by lack of progress
            if results and (time.time() - stagnant_since) > 5.0:
                break
            continue

        results.append(
            FrameResult(
                t=float(r["t"]),
                p_now=(float(r["p_now"][0]), float(r["p_now"][1])),
                p_future=(float(r["p_future"][0]), float(r["p_future"][1])),
                vad=(float(r["vad"][0]), float(r["vad"][1])),
            )
        )
        if len(results) != last_size:
            stagnant_since = time.time()
            last_size = len(results)

        if time.time() - last_log > 15:
            print(f"  collected {len(results)} frames...")
            last_log = time.time()

    return results


# ---------------------------------------------------------------------------
# Per-meeting eval
# ---------------------------------------------------------------------------


def evaluate_meeting(meeting_id: str, *, fast: bool) -> MeetingResult | None:
    print(f"\n=== {meeting_id} ===")
    audio_dir = AUDIO_ROOT / meeting_id / "audio"
    if not audio_dir.is_dir():
        print(f"  SKIP: audio not found at {audio_dir}")
        return None

    meeting = load_meeting(ANNOT_ROOT, meeting_id)
    spk1, spk2 = pick_two_most_active_speakers(meeting.segments_by_speaker)
    talk = {s: sum(seg.duration for seg in segs) for s, segs in meeting.segments_by_speaker.items()}
    print(f"  Speakers (talk s): {[(s, round(talk[s], 1)) for s in meeting.speakers]}")
    print(f"  Evaluating channels: ch1={spk1}, ch2={spk2}")

    wav1 = audio_dir / f"{meeting_id}.Headset-{SPEAKER_TO_CHANNEL[spk1]}.wav"
    wav2 = audio_dir / f"{meeting_id}.Headset-{SPEAKER_TO_CHANNEL[spk2]}.wav"
    if not wav1.exists() or not wav2.exists():
        print(f"  SKIP: missing wav: {wav1} or {wav2}")
        return None

    info1 = sf.info(wav1)
    info2 = sf.info(wav2)
    duration_sec = min(info1.duration, info2.duration)
    print(f"  Audio: {duration_sec:.1f}s @ {info1.samplerate} Hz")

    gt_vad = build_vad_arrays(
        meeting.segments_by_speaker, (spk1, spk2), duration_sec, FRAME_RATE_HZ
    )

    results = run_maai_inference(
        wav1, wav2, duration_sec=duration_sec, frame_rate=FRAME_RATE_HZ, fast=fast
    )
    print(f"  MaAI produced {len(results)} frames (expected ~{int(duration_sec * FRAME_RATE_HZ)})")

    if not results:
        print("  WARN: no frames collected")
        return None

    n = min(len(results), len(gt_vad))
    p_now = np.array([r.p_now for r in results[:n]])
    p_future = np.array([r.p_future for r in results[:n]])
    gt = gt_vad[:n]

    # Frame-level VAD agreement on single-speaker frames
    single_mask = gt[:, 0] ^ gt[:, 1]
    if single_mask.sum() > 0:
        gt_speaker = gt[:, 1].astype(int)
        pred_speaker = (p_now[:, 1] > p_now[:, 0]).astype(int)
        frame_acc = float((gt_speaker[single_mask] == pred_speaker[single_mask]).mean())
    else:
        frame_acc = float("nan")

    # Hold/shift accuracy at mutual silences
    silences = find_mutual_silences(gt, FRAME_RATE_HZ)
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    score_pairs: list[tuple[str, float]] = []
    for i_s, i_e in silences:
        prev_ch, next_ch, label = classify_hold_shift(gt, i_s, i_e)
        if label is None:
            continue
        mid = (i_s + i_e) // 2
        if mid >= n:
            continue
        pred_next_ch = 1 if p_future[mid, 1] > p_future[mid, 0] else 0
        pred_label = "hold" if pred_next_ch == prev_ch else "shift"
        # Shift score: probability mass on the speaker who was NOT speaking before silence
        shift_score = float(p_future[mid, 1 - prev_ch])
        score_pairs.append((label, shift_score))
        counts[label] += 1
        if pred_label == label:
            correct[label] += 1
        confusion[(label, pred_label)] += 1

    result = MeetingResult(
        meeting_id=meeting_id,
        duration_sec=duration_sec,
        frame_acc=frame_acc,
        hold_correct=correct["hold"],
        hold_total=counts["hold"],
        shift_correct=correct["shift"],
        shift_total=counts["shift"],
        confusion=dict(confusion),
        score_pairs=score_pairs,
    )

    print(f"  Frame VAD acc: {frame_acc:.3f}")
    print(
        f"  Hold:  {result.hold_correct}/{result.hold_total} = "
        f"{(result.hold_correct / result.hold_total) if result.hold_total else 0:.3f}"
    )
    print(
        f"  Shift: {result.shift_correct}/{result.shift_total} = "
        f"{(result.shift_correct / result.shift_total) if result.shift_total else 0:.3f}"
    )
    print(f"  Overall: {result.overall_correct}/{result.overall_total} = {result.overall_acc:.3f}")
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def print_aggregate(results: list[MeetingResult]) -> None:
    if not results:
        print("\nNo results to aggregate.")
        return

    print("\n" + "=" * 72)
    print("AGGREGATE RESULTS")
    print("=" * 72)
    print(f"{'meeting':<12}{'dur(s)':>9}{'frame_acc':>11}{'hold':>14}{'shift':>14}{'overall':>14}")
    for r in results:
        hold_str = f"{r.hold_correct}/{r.hold_total}" if r.hold_total else "-"
        shift_str = f"{r.shift_correct}/{r.shift_total}" if r.shift_total else "-"
        overall_str = f"{r.overall_correct}/{r.overall_total} ({r.overall_acc:.3f})"
        print(
            f"{r.meeting_id:<12}{r.duration_sec:>9.0f}"
            f"{r.frame_acc:>11.3f}{hold_str:>14}{shift_str:>14}{overall_str:>14}"
        )

    total_hold_c = sum(r.hold_correct for r in results)
    total_hold_n = sum(r.hold_total for r in results)
    total_shift_c = sum(r.shift_correct for r in results)
    total_shift_n = sum(r.shift_total for r in results)
    total_c = total_hold_c + total_shift_c
    total_n = total_hold_n + total_shift_n
    weighted_frame_acc = sum(r.frame_acc * r.duration_sec for r in results) / sum(
        r.duration_sec for r in results
    )

    print("-" * 72)
    if total_n > 0:
        overall_str = f"{total_c}/{total_n} ({total_c / total_n:.3f})"
    else:
        overall_str = "no eligible silences"
    print(
        f"{'POOLED':<12}{'':<9}{weighted_frame_acc:>11.3f}"
        f"{f'{total_hold_c}/{total_hold_n}':>14}"
        f"{f'{total_shift_c}/{total_shift_n}':>14}"
        f"{overall_str:>14}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meetings", nargs="*", help="Meeting IDs (default: ES2002a)")
    parser.add_argument("--all", action="store_true", help="Run all 5 default meetings")
    parser.add_argument(
        "--fast", action="store_true", help="Skip real-time wav playback timing (~4x faster)"
    )
    parser.add_argument(
        "--auc", action="store_true", help="Report ROC-AUC and PR-AUC for shift detection"
    )
    args = parser.parse_args()

    if args.all:
        meetings = DEFAULT_ALL_MEETINGS
    elif args.meetings:
        meetings = args.meetings
    else:
        meetings = ["ES2002a"]

    if not ANNOT_ROOT.is_dir():
        sys.exit(
            f"AMI annotations not found at {ANNOT_ROOT}. "
            "Run: python scripts/download_ami_subset.py --annotations-only"
        )

    results: list[MeetingResult] = []
    for mid in meetings:
        r = evaluate_meeting(mid, fast=args.fast)
        if r is not None:
            results.append(r)

    print_aggregate(results)

    if args.auc:
        print_auc(results)


def _auc_from_pairs(pairs: list[tuple[str, float]]) -> tuple[float, float, int, int]:
    """Compute (ROC-AUC, PR-AUC, n_pos, n_neg). Positive class = 'shift'."""
    if not pairs:
        return float("nan"), float("nan"), 0, 0
    y = np.array([1 if g == "shift" else 0 for g, _ in pairs], dtype=np.int64)
    s = np.array([sc for _, sc in pairs], dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), n_pos, n_neg
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_ranks_pos = ranks[y == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    roc_auc = float(u / (n_pos * n_neg))

    desc = np.argsort(-s, kind="mergesort")
    y_sorted = y[desc]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    pr_auc = float(np.sum(np.diff(recall) * precision[1:]))
    return roc_auc, pr_auc, n_pos, n_neg


def print_auc(results: list[MeetingResult]) -> None:
    print("\n" + "=" * 72)
    print("THRESHOLD-FREE METRICS (positive class = 'shift')")
    print("=" * 72)
    print(f"{'meeting':<12}{'n_pos':>8}{'n_neg':>8}{'ROC-AUC':>12}{'PR-AUC':>10}")
    pooled: list[tuple[str, float]] = []
    for r in results:
        if r.score_pairs is None:
            continue
        pooled.extend(r.score_pairs)
        roc, pr, p, n = _auc_from_pairs(r.score_pairs)
        print(f"{r.meeting_id:<12}{p:>8}{n:>8}{roc:>12.3f}{pr:>10.3f}")
    roc, pr, p, n = _auc_from_pairs(pooled)
    print(f"{'POOLED':<12}{p:>8}{n:>8}{roc:>12.3f}{pr:>10.3f}")


if __name__ == "__main__":
    main()

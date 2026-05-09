"""PyTorch Dataset for ITM training on AMI Corpus.

Each item is a fixed-duration chunk of synchronized 2-channel audio plus
the multi-event survival hazard targets aligned to that chunk.

Typical use::

    from itm.data.dataset import AMIDataset
    from torch.utils.data import DataLoader

    ds = AMIDataset(
        annot_root="data/raw/ami/annotations/unpacked",
        audio_root="data/raw/ami",
        meeting_ids=["ES2002a", "ES2002b"],
        chunk_sec=20.0,
        hop_sec=10.0,
        frame_rate_hz=20,
        horizon_bins=40,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ami_collate)
    batch = next(iter(loader))

Audio for each meeting is loaded once and cached in memory. With 5
meetings @ ~20–40 min each, the working set is ~1 GB which is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from itm.data.ami import Meeting, load_meeting
from itm.data.audio import SAMPLING_RATE, load_two_channel_audio, slice_chunk
from itm.data.labels import EventType, extract_event_onsets, survival_targets
from itm.data.targets import survival_to_tensors


@dataclass
class _MeetingCache:
    """Per-meeting cached state."""

    meeting_id: str
    speakers: tuple[str, str]
    audio: np.ndarray  # (n_samples, 2) float32
    duration_sec: float
    targets_full: dict[EventType, dict[str, torch.Tensor]]
    """Full-meeting hazard/mask tensors at frame_rate_hz."""
    vad_full: torch.Tensor
    """Full-meeting per-frame VAD: ``(n_frames_full, 2)`` float32 in {0, 1}."""


def _pick_two_most_active(meeting: Meeting) -> tuple[str, str]:
    talk = {spk: sum(s.duration for s in segs) for spk, segs in meeting.segments_by_speaker.items()}
    ranked = sorted(talk.items(), key=lambda kv: -kv[1])
    return ranked[0][0], ranked[1][0]


class AMIDataset(Dataset):
    """Stream fixed-length audio + hazard target chunks from a list of AMI meetings.

    Each item is::

        {
            "audio":   Float[n_audio_samples, 2],
            "hazard":  {ev: Long[n_frames, K]},
            "mask":    {ev: Float[n_frames, K]},
            "meeting": str,
            "start_sec": float,
        }

    Args:
        annot_root: AMI manual annotations dir (with ``words/`` etc.)
        audio_root: AMI audio root (with one subdir per meeting)
        meeting_ids: list of meeting identifiers (e.g. ``["ES2002a"]``).
        chunk_sec: duration of each training chunk.
        hop_sec: stride between consecutive chunk starts.
        frame_rate_hz: model output frame rate (must match training config).
        horizon_bins: number of future bins K (e.g. 40 for 2 s @ 50 ms each).
        bin_size_sec: width of each future bin (default 0.05 s).
        speakers_override: optional dict mapping meeting_id → 2-tuple of
            speaker letters. Defaults to the two most-active per meeting.
    """

    def __init__(
        self,
        annot_root: Path | str,
        audio_root: Path | str,
        meeting_ids: list[str],
        *,
        chunk_sec: float = 20.0,
        hop_sec: float = 10.0,
        frame_rate_hz: int = 20,
        horizon_bins: int = 40,
        bin_size_sec: float = 0.05,
        speakers_override: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        if chunk_sec <= 0 or hop_sec <= 0:
            raise ValueError("chunk_sec and hop_sec must be positive")

        self.annot_root = Path(annot_root)
        self.audio_root = Path(audio_root)
        self.chunk_sec = float(chunk_sec)
        self.hop_sec = float(hop_sec)
        self.frame_rate_hz = int(frame_rate_hz)
        self.horizon_bins = int(horizon_bins)
        self.bin_size_sec = float(bin_size_sec)

        self._caches: list[_MeetingCache] = []
        self._index: list[tuple[int, float]] = []  # (cache_idx, start_sec)

        for mid in meeting_ids:
            cache = self._load_meeting_cache(mid, speakers_override)
            self._caches.append(cache)
            cache_idx = len(self._caches) - 1
            n_chunks = max(1, int((cache.duration_sec - chunk_sec) // hop_sec) + 1)
            for i in range(n_chunks):
                start = i * self.hop_sec
                if start + self.chunk_sec > cache.duration_sec + self.chunk_sec:
                    break
                self._index.append((cache_idx, start))

    # ---------------------------------------------------------------- internals

    def _load_meeting_cache(
        self,
        meeting_id: str,
        speakers_override: dict[str, tuple[str, str]] | None,
    ) -> _MeetingCache:
        meeting = load_meeting(self.annot_root, meeting_id)
        if speakers_override and meeting_id in speakers_override:
            speakers = speakers_override[meeting_id]
        else:
            speakers = _pick_two_most_active(meeting)

        audio_dir = self.audio_root / meeting_id / "audio"
        audio, duration = load_two_channel_audio(audio_dir, meeting_id, speakers)

        # Extract event onsets and convert to per-frame survival targets
        onsets = extract_event_onsets(meeting)
        targets = survival_targets(
            onsets,
            duration=duration,
            frame_rate=self.frame_rate_hz,
            horizon_bins=self.horizon_bins,
            bin_size_sec=self.bin_size_sec,
        )
        targets_t = survival_to_tensors(targets, horizon_bins=self.horizon_bins)

        # Build per-frame VAD tensor for the two evaluated speakers.
        n_frames_full = int(duration * self.frame_rate_hz)
        vad = torch.zeros(n_frames_full, 2, dtype=torch.float32)
        dt = 1.0 / self.frame_rate_hz
        for ch, spk in enumerate(speakers):
            for seg in meeting.segments_by_speaker.get(spk, []):
                i_start = max(0, int(seg.start / dt))
                i_end = min(n_frames_full, int(seg.end / dt) + 1)
                vad[i_start:i_end, ch] = 1.0

        return _MeetingCache(
            meeting_id=meeting_id,
            speakers=speakers,
            audio=audio,
            duration_sec=duration,
            targets_full=targets_t,
            vad_full=vad,
        )

    # -------------------------------------------------------------- Dataset API

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        cache_idx, start_sec = self._index[idx]
        cache = self._caches[cache_idx]

        audio_chunk = slice_chunk(
            cache.audio, start_sec, self.chunk_sec, sampling_rate=SAMPLING_RATE, pad=True
        )
        # Convert to torch tensor: (n_samples, 2) float32
        audio_t = torch.from_numpy(audio_chunk.copy())

        # Slice targets at frame_rate_hz
        n_frames_chunk = int(self.chunk_sec * self.frame_rate_hz)
        f_start = int(start_sec * self.frame_rate_hz)
        f_end = f_start + n_frames_chunk

        hazard: dict[EventType, torch.Tensor] = {}
        mask: dict[EventType, torch.Tensor] = {}
        for ev_type, full in cache.targets_full.items():
            full_h = full["hazard"]  # (n_frames_full, K)
            full_m = full["mask"]
            # If chunk extends past end, right-pad with zero hazard / mask
            if f_end <= full_h.size(0):
                hazard[ev_type] = full_h[f_start:f_end]
                mask[ev_type] = full_m[f_start:f_end]
            else:
                pad_len = f_end - full_h.size(0)
                avail = max(0, full_h.size(0) - f_start)
                pad_h = torch.zeros(pad_len, self.horizon_bins, dtype=full_h.dtype)
                pad_m = torch.zeros(pad_len, self.horizon_bins, dtype=full_m.dtype)
                if avail > 0:
                    hazard[ev_type] = torch.cat([full_h[f_start:], pad_h], dim=0)
                    mask[ev_type] = torch.cat([full_m[f_start:], pad_m], dim=0)
                else:
                    hazard[ev_type] = pad_h
                    mask[ev_type] = pad_m

        # VAD slice
        vf_start = f_start
        vf_end = f_end
        if vf_end <= cache.vad_full.size(0):
            vad_slice = cache.vad_full[vf_start:vf_end]
        else:
            avail = max(0, cache.vad_full.size(0) - vf_start)
            pad_v = torch.zeros(vf_end - cache.vad_full.size(0), 2, dtype=torch.float32)
            if avail > 0:
                vad_slice = torch.cat([cache.vad_full[vf_start:], pad_v], dim=0)
            else:
                vad_slice = pad_v.new_zeros(n_frames_chunk, 2)

        return {
            "audio": audio_t,
            "hazard": hazard,
            "mask": mask,
            "vad_target": vad_slice,
            "meeting": cache.meeting_id,
            "speakers": cache.speakers,
            "start_sec": float(start_sec),
        }


# ---------------------------------------------------------------- collate fn


def ami_collate(batch: list[dict]) -> dict:
    """Stack dataset items into a batch dict.

    Tensors gain a leading batch dim of size B. Strings/floats become lists.
    """
    out: dict = {
        "audio": torch.stack([b["audio"] for b in batch], dim=0),
        "hazard": {},
        "mask": {},
        "vad_target": torch.stack([b["vad_target"] for b in batch], dim=0),
        "meeting": [b["meeting"] for b in batch],
        "speakers": [b["speakers"] for b in batch],
        "start_sec": torch.tensor([b["start_sec"] for b in batch], dtype=torch.float32),
    }
    for ev_type in batch[0]["hazard"]:
        out["hazard"][ev_type] = torch.stack([b["hazard"][ev_type] for b in batch], dim=0)
        out["mask"][ev_type] = torch.stack([b["mask"][ev_type] for b in batch], dim=0)
    return out

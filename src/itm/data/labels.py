"""Multi-event label generation for ITM.

Converts AMI dialog acts and segments into onset events for ITM's three
hazard heads: turn-shift, backchannel, and overlap. Also provides a helper
to build per-frame survival targets used during training.

Event definitions (see docs/design/label-generation.md for rationale):

* **Backchannel** — dialog act type ``bck``, short (< 1.5s by default), occurring
  while another speaker is mid-utterance.
* **Turn-shift** — a different speaker starts a substantive utterance after
  ``min_silence_gap`` of mutual silence; the new utterance must persist
  for at least ``min_substantive_duration``.
* **Overlap** — a speaker starts during another speaker's ongoing speech,
  excluding short backchannels.

Event onset times are returned as wall-clock seconds. ``survival_targets``
converts those into discrete-time hazard targets per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from itm.data.ami import Meeting, Segment

# AMI dialog act type groupings (from ontologies/da-types.xml)
BACKCHANNEL_DA_TYPES: frozenset[str] = frozenset({"bck"})
HOLD_DA_TYPES: frozenset[str] = frozenset({"stl", "fra"})  # stalls and fragments

# Substantive turn-taking acts: task / elicit / certain "other" categories.
# Anything in this set, when starting after a gap, counts as a turn-shift candidate.
SUBSTANTIVE_DA_TYPES: frozenset[str] = frozenset(
    {
        "inf",
        "sug",
        "ass",
        "el.inf",
        "el.sug",
        "el.ass",
        "el.und",
        "off",
        "und",
    }
)


class EventType(StrEnum):
    """ITM event taxonomy."""

    TURN_SHIFT = "turn_shift"
    BACKCHANNEL = "backchannel"
    OVERLAP = "overlap"


@dataclass(frozen=True, slots=True)
class EventOnset:
    """A single ITM event occurrence."""

    type: EventType
    time: float  # onset time in seconds
    speaker: str  # the speaker who initiates the event


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


def _is_speaking_at(segments: list[Segment], t: float) -> bool:
    """O(n) scan; fine for typical AMI segment counts (a few hundred per speaker)."""
    return any(s.start <= t < s.end for s in segments)


def _any_other_speaker_active(
    segments_by_speaker: dict[str, list[Segment]],
    speaker: str,
    t: float,
) -> bool:
    return any(
        _is_speaking_at(segs, t) for spk, segs in segments_by_speaker.items() if spk != speaker
    )


def extract_event_onsets(
    meeting: Meeting,
    *,
    min_silence_gap: float = 0.2,
    min_substantive_duration: float = 1.5,
    max_backchannel_duration: float = 1.5,
    min_overlap_duration: float = 0.5,
) -> list[EventOnset]:
    """Extract turn-shift / backchannel / overlap onsets from a parsed AMI meeting.

    The thresholds are conservative defaults intended to yield high-precision
    automatic labels. Tune via the keyword arguments.

    Args:
        meeting: parsed ``Meeting`` from ``itm.data.ami``.
        min_silence_gap: minimum silence (seconds) before a substantive
            utterance to count as a turn-shift onset.
        min_substantive_duration: minimum duration of the new utterance for
            it to be considered substantive (vs. a near-backchannel).
        max_backchannel_duration: dialog acts longer than this are not
            counted as backchannels.
        min_overlap_duration: overlapping speech shorter than this is
            ignored (likely backchannel territory).

    Returns:
        ``list[EventOnset]`` sorted by time.
    """
    onsets: list[EventOnset] = []

    # --- Backchannel ---
    for spk, dialog_acts in meeting.dialog_acts_by_speaker.items():
        for da in dialog_acts:
            if da.da_type not in BACKCHANNEL_DA_TYPES:
                continue
            if da.duration > max_backchannel_duration:
                continue
            if not _any_other_speaker_active(meeting.segments_by_speaker, spk, da.start):
                continue
            onsets.append(EventOnset(type=EventType.BACKCHANNEL, time=da.start, speaker=spk))

    # --- Turn-shift (uses dialog acts to find substantive starts) ---
    # In multi-party meetings (AMI: 4 speakers), requiring "nobody else is
    # speaking" is too strict — there's almost always concurrent backchannel
    # activity. Instead we require:
    #   1. The new substantive utterance is not the same speaker continuing.
    #   2. *Substantive* (non-backchannel) speech from another speaker has
    #      ended at least ``min_silence_gap`` ago — i.e. the floor was open.
    def _substantive_speech_at(t: float, exclude_spk: str) -> bool:
        for other_spk, other_das in meeting.dialog_acts_by_speaker.items():
            if other_spk == exclude_spk:
                continue
            for other_da in other_das:
                if other_da.da_type in BACKCHANNEL_DA_TYPES:
                    continue
                if other_da.start <= t < other_da.end:
                    return True
        return False

    for spk, dialog_acts in meeting.dialog_acts_by_speaker.items():
        for da in dialog_acts:
            if da.da_type not in SUBSTANTIVE_DA_TYPES:
                continue
            if da.duration < min_substantive_duration:
                continue
            gap_start = da.start - min_silence_gap

            # Same speaker continuing? Look for any of their own substantive
            # speech ending in (gap_start, da.start). If so, it's a hold.
            same_speaker_continuing = False
            for own_da in dialog_acts:
                if own_da is da:
                    continue
                if own_da.da_type in BACKCHANNEL_DA_TYPES:
                    continue
                if gap_start < own_da.end <= da.start:
                    same_speaker_continuing = True
                    break
                if own_da.start < da.start < own_da.end:
                    same_speaker_continuing = True
                    break
            if same_speaker_continuing:
                continue

            # Was the floor open at gap_start? (No other speaker substantive)
            if _substantive_speech_at(gap_start, exclude_spk=spk):
                continue

            onsets.append(EventOnset(type=EventType.TURN_SHIFT, time=da.start, speaker=spk))

    # --- Overlap (segment-based) ---
    speakers = list(meeting.segments_by_speaker.keys())
    for i, spk_a in enumerate(speakers):
        for spk_b in speakers[i + 1 :]:
            segs_a = meeting.segments_by_speaker[spk_a]
            segs_b = meeting.segments_by_speaker[spk_b]
            for seg_a in segs_a:
                for seg_b in segs_b:
                    # B starts during A's speech
                    if seg_a.start < seg_b.start < seg_a.end:
                        overlap_dur = min(seg_a.end, seg_b.end) - seg_b.start
                        if overlap_dur >= min_overlap_duration:
                            onsets.append(
                                EventOnset(type=EventType.OVERLAP, time=seg_b.start, speaker=spk_b)
                            )
                    # A starts during B's speech
                    if seg_b.start < seg_a.start < seg_b.end:
                        overlap_dur = min(seg_a.end, seg_b.end) - seg_a.start
                        if overlap_dur >= min_overlap_duration:
                            onsets.append(
                                EventOnset(type=EventType.OVERLAP, time=seg_a.start, speaker=spk_a)
                            )

    onsets.sort(key=lambda o: o.time)
    return onsets


# ---------------------------------------------------------------------------
# Survival target construction
# ---------------------------------------------------------------------------


def survival_targets(
    onsets: list[EventOnset],
    *,
    duration: float,
    frame_rate: float = 50.0,
    horizon_bins: int = 40,
    bin_size_sec: float = 0.05,
) -> dict[EventType, list[int]]:
    """Build per-frame discrete-time survival targets.

    For every frame ``t``, returns ``k_e`` — the bin index of the next
    ``e``-event in ``[t, t + horizon_bins * bin_size_sec)``, or ``-1`` if
    no event falls in that window (right-censored).

    Args:
        onsets: events from ``extract_event_onsets``.
        duration: length of the meeting in seconds (frames will span
            ``[0, duration)``).
        frame_rate: model output frame rate. ``50 Hz`` matches VAP.
        horizon_bins: number of future bins to consider (e.g. 40 = 2 s).
        bin_size_sec: width of each future bin (default 50 ms).

    Returns:
        Dict mapping each ``EventType`` to a list of length
        ``int(duration * frame_rate)`` of target bin indices (or ``-1``).
    """
    n_frames = max(0, int(duration * frame_rate))
    horizon_sec = horizon_bins * bin_size_sec

    onsets_by_event: dict[EventType, list[float]] = {e: [] for e in EventType}
    for o in onsets:
        onsets_by_event[o.type].append(o.time)
    for lst in onsets_by_event.values():
        lst.sort()

    targets: dict[EventType, list[int]] = {e: [-1] * n_frames for e in EventType}

    frame_dt = 1.0 / frame_rate
    for e, times in onsets_by_event.items():
        # Two-pointer: advance through onsets as we walk frames forward
        i = 0
        for f in range(n_frames):
            t = f * frame_dt
            # Skip onsets that are already in the past
            while i < len(times) and times[i] < t:
                i += 1
            if i >= len(times):
                break  # no more future events for this type
            dt = times[i] - t
            if dt < horizon_sec:
                bin_idx = int(dt / bin_size_sec)
                if 0 <= bin_idx < horizon_bins:
                    targets[e][f] = bin_idx
            # if dt >= horizon_sec, frame f stays -1 (censored within window)

    return targets

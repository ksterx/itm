"""Unit tests for ``itm.data.labels``."""

from __future__ import annotations

import pytest

from itm.data.ami import DialogAct, Meeting, Segment
from itm.data.labels import (
    EventOnset,
    EventType,
    extract_event_onsets,
    survival_targets,
)


def _make_meeting(
    *,
    segments: dict[str, list[Segment]] | None = None,
    dialog_acts: dict[str, list[DialogAct]] | None = None,
) -> Meeting:
    segments = segments or {}
    dialog_acts = dialog_acts or {}
    speakers = tuple(sorted(set(segments) | set(dialog_acts)))
    return Meeting(
        id="TEST",
        speakers=speakers,
        words_by_speaker=dict.fromkeys(speakers, {}),
        segments_by_speaker={s: segments.get(s, []) for s in speakers},
        dialog_acts_by_speaker={s: dialog_acts.get(s, []) for s in speakers},
    )


def _seg(spk: str, start: float, end: float, sid: str = "seg") -> Segment:
    return Segment(id=f"{spk}.{sid}", speaker=spk, start=start, end=end)


def _da(spk: str, start: float, end: float, da_type: str, did: str = "da") -> DialogAct:
    return DialogAct(
        id=f"{spk}.{did}",
        speaker=spk,
        start=start,
        end=end,
        da_type=da_type,
        da_id=f"ami_da_{did}",
    )


class TestExtractTurnShift:
    def test_clean_turn_shift_after_silence(self) -> None:
        # A speaks 0.0–2.0, then silence, then B speaks substantive 3.0–6.0
        meeting = _make_meeting(
            segments={
                "A": [_seg("A", 0.0, 2.0)],
                "B": [_seg("B", 3.0, 6.0)],
            },
            dialog_acts={
                "A": [_da("A", 0.0, 2.0, "inf")],
                "B": [_da("B", 3.0, 6.0, "inf")],
            },
        )
        onsets = extract_event_onsets(meeting)
        turn_onsets = [o for o in onsets if o.type == EventType.TURN_SHIFT]
        # Both A's first inf and B's inf could be candidate turn-shifts; A's has
        # no preceding speaker so it is ALSO a turn-shift in this corner case.
        # Assert at least B's onset is captured.
        speakers = {o.speaker for o in turn_onsets}
        assert "B" in speakers
        b_onset = next(o for o in turn_onsets if o.speaker == "B")
        assert b_onset.time == pytest.approx(3.0)

    def test_too_short_substantive_rejected(self) -> None:
        # B's "inf" lasts only 0.5s — under min_substantive_duration
        meeting = _make_meeting(
            segments={
                "A": [_seg("A", 0.0, 2.0)],
                "B": [_seg("B", 3.0, 3.5)],
            },
            dialog_acts={
                "A": [_da("A", 0.0, 2.0, "inf")],
                "B": [_da("B", 3.0, 3.5, "inf")],
            },
        )
        onsets = extract_event_onsets(meeting)
        b_turn_onsets = [o for o in onsets if o.type == EventType.TURN_SHIFT and o.speaker == "B"]
        assert b_turn_onsets == []

    def test_no_silence_gap_rejected(self) -> None:
        # A speaks until 3.05, B starts at 3.0 (overlap, no silence gap)
        meeting = _make_meeting(
            segments={
                "A": [_seg("A", 0.0, 3.05)],
                "B": [_seg("B", 3.0, 6.0)],
            },
            dialog_acts={
                "A": [_da("A", 0.0, 3.05, "inf")],
                "B": [_da("B", 3.0, 6.0, "inf")],
            },
        )
        onsets = extract_event_onsets(meeting)
        b_turn_onsets = [o for o in onsets if o.type == EventType.TURN_SHIFT and o.speaker == "B"]
        assert b_turn_onsets == []  # B's start during A's speech is overlap, not turn-shift

    def test_concurrent_backchannel_does_not_block_turn_shift(self) -> None:
        # Multi-party realism: A talks, C drops a "yeah" backchannel,
        # then A's substantive talk ends, and B picks up after a gap.
        # The concurrent backchannel from C must not block detection of B's turn-shift.
        meeting = _make_meeting(
            segments={
                "A": [_seg("A", 0.0, 2.0)],
                "C": [_seg("C", 1.0, 1.3)],  # backchannel timing
                "B": [_seg("B", 3.0, 6.0)],
            },
            dialog_acts={
                "A": [_da("A", 0.0, 2.0, "inf")],
                "C": [_da("C", 1.0, 1.3, "bck")],  # bck, not substantive
                "B": [_da("B", 3.0, 6.0, "inf")],
            },
        )
        onsets = extract_event_onsets(meeting)
        b_turn_onsets = [o for o in onsets if o.type == EventType.TURN_SHIFT and o.speaker == "B"]
        assert len(b_turn_onsets) == 1
        assert b_turn_onsets[0].time == pytest.approx(3.0)

    def test_same_speaker_continuation_rejected(self) -> None:
        # A's two adjacent inf utterances with only a 50ms gap → continuation, not turn-shift
        meeting = _make_meeting(
            segments={
                "A": [_seg("A", 0.0, 2.0, "s1"), _seg("A", 2.05, 5.0, "s2")],
            },
            dialog_acts={
                "A": [_da("A", 0.0, 2.0, "inf", "d1"), _da("A", 2.05, 5.0, "inf", "d2")],
            },
        )
        onsets = extract_event_onsets(meeting)
        # First inf is itself an unconstrained start (no prior speaker), so 1 turn-shift max.
        # Second inf is a continuation by same speaker → rejected.
        a_turn_onsets = [o for o in onsets if o.type == EventType.TURN_SHIFT and o.speaker == "A"]
        assert len(a_turn_onsets) <= 1  # only the first counts (or zero)


class TestExtractBackchannel:
    def test_backchannel_during_other_speech(self) -> None:
        # A speaks 0–10, B drops "yeah" (bck) at t=3
        meeting = _make_meeting(
            segments={
                "A": [_seg("A", 0.0, 10.0)],
                "B": [_seg("B", 3.0, 3.3)],
            },
            dialog_acts={
                "A": [_da("A", 0.0, 10.0, "inf")],
                "B": [_da("B", 3.0, 3.3, "bck")],
            },
        )
        onsets = extract_event_onsets(meeting)
        bcs = [o for o in onsets if o.type == EventType.BACKCHANNEL]
        assert len(bcs) == 1
        assert bcs[0].speaker == "B"
        assert bcs[0].time == pytest.approx(3.0)

    def test_backchannel_during_silence_rejected(self) -> None:
        # No one else speaking when B says "uh"
        meeting = _make_meeting(
            segments={
                "A": [_seg("A", 0.0, 1.0)],
                "B": [_seg("B", 5.0, 5.3)],
            },
            dialog_acts={
                "A": [_da("A", 0.0, 1.0, "inf")],
                "B": [_da("B", 5.0, 5.3, "bck")],
            },
        )
        onsets = extract_event_onsets(meeting)
        bcs = [o for o in onsets if o.type == EventType.BACKCHANNEL]
        assert bcs == []  # not a backchannel — no one to backchannel against

    def test_long_bck_excluded(self) -> None:
        # 2-second "bck" — exceeds max_backchannel_duration default (1.5s)
        meeting = _make_meeting(
            segments={
                "A": [_seg("A", 0.0, 10.0)],
                "B": [_seg("B", 3.0, 5.0)],
            },
            dialog_acts={
                "A": [_da("A", 0.0, 10.0, "inf")],
                "B": [_da("B", 3.0, 5.0, "bck")],
            },
        )
        onsets = extract_event_onsets(meeting)
        bcs = [o for o in onsets if o.type == EventType.BACKCHANNEL]
        assert bcs == []


class TestExtractOverlap:
    def test_b_starts_during_a(self) -> None:
        meeting = _make_meeting(
            segments={
                "A": [_seg("A", 0.0, 5.0)],
                "B": [_seg("B", 2.0, 4.0)],  # 2s overlap
            },
        )
        onsets = extract_event_onsets(meeting)
        overlaps = [o for o in onsets if o.type == EventType.OVERLAP]
        assert len(overlaps) == 1
        assert overlaps[0].speaker == "B"
        assert overlaps[0].time == pytest.approx(2.0)

    def test_short_overlap_ignored(self) -> None:
        # Only 0.1s of overlap
        meeting = _make_meeting(
            segments={
                "A": [_seg("A", 0.0, 2.1)],
                "B": [_seg("B", 2.0, 5.0)],
            },
        )
        onsets = extract_event_onsets(meeting)
        overlaps = [o for o in onsets if o.type == EventType.OVERLAP]
        assert overlaps == []


class TestSurvivalTargets:
    def test_simple_event(self) -> None:
        # Single backchannel at t=1.0, frame_rate=10Hz, horizon=20 bins × 0.1s = 2s
        onset = EventOnset(type=EventType.BACKCHANNEL, time=1.0, speaker="B")
        targets = survival_targets(
            [onset],
            duration=2.0,
            frame_rate=10.0,
            horizon_bins=20,
            bin_size_sec=0.1,
        )
        # 20 frames total
        assert len(targets[EventType.BACKCHANNEL]) == 20
        # Frame 0 (t=0.0): event at t=1.0 → 1.0/0.1 = bin 10
        assert targets[EventType.BACKCHANNEL][0] == 10
        # Frame 5 (t=0.5): event in 0.5s → bin 5
        assert targets[EventType.BACKCHANNEL][5] == 5
        # Frame 10 (t=1.0): event NOW → bin 0
        assert targets[EventType.BACKCHANNEL][10] == 0
        # Frame 11 (t=1.1): past, no future events → -1
        assert targets[EventType.BACKCHANNEL][11] == -1
        # Other event types should all be -1
        assert all(v == -1 for v in targets[EventType.TURN_SHIFT])

    def test_event_outside_horizon(self) -> None:
        # Event at t=10s, horizon only 2s → all frames in 0–8s should be -1
        onset = EventOnset(type=EventType.TURN_SHIFT, time=10.0, speaker="A")
        targets = survival_targets(
            [onset],
            duration=8.0,
            frame_rate=10.0,
            horizon_bins=20,
            bin_size_sec=0.1,
        )
        assert all(v == -1 for v in targets[EventType.TURN_SHIFT])

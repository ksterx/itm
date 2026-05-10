"""Integration test for ``itm.data.dataset``.

Builds a tiny self-contained AMI-shaped corpus on disk (XML + WAVs),
constructs an :class:`AMIDataset`, and verifies item shapes, batching,
and target alignment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from itm.data.audio import SAMPLING_RATE
from itm.data.dataset import AMIDataset, ami_collate
from itm.data.labels import EventType

NITE_NS = 'xmlns:nite="http://nite.sourceforge.net/"'


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _write_wav(p: Path, duration_sec: float, freq: float = 220.0) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_sec * SAMPLING_RATE)
    t = np.arange(n) / SAMPLING_RATE
    sig = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(p, sig, SAMPLING_RATE)


@pytest.fixture()
def fake_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(annot_root, audio_root)`` for a 60-second 2-speaker corpus."""
    annot_root = tmp_path / "annotations" / "unpacked"
    audio_root = tmp_path / "audio"
    mid = "MEET01"

    _write(
        annot_root / "ontologies" / "da-types.xml",
        f"""<?xml version="1.0"?>
<da-type {NITE_NS} nite:id="cmrda" name="da-type">
  <da-type nite:id="ami_da_1" name="bck"/>
  <da-type nite:id="ami_da_4" name="inf"/>
</da-type>
""",
    )
    # Speaker A talks 0–10, then 30–40
    _write(
        annot_root / "words" / f"{mid}.A.words.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.A.words">
  <w nite:id="{mid}.A.words0" starttime="0.0" endtime="5.0">first</w>
  <w nite:id="{mid}.A.words1" starttime="5.0" endtime="10.0">utterance</w>
  <w nite:id="{mid}.A.words2" starttime="30.0" endtime="35.0">second</w>
  <w nite:id="{mid}.A.words3" starttime="35.0" endtime="40.0">utterance</w>
</nite:root>
""",
    )
    _write(
        annot_root / "segments" / f"{mid}.A.segments.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.A.segs">
  <segment nite:id="{mid}.A.seg.1" transcriber_start="0.0" transcriber_end="10.0">
    <nite:child href="{mid}.A.words.xml#id({mid}.A.words0)..id({mid}.A.words1)"/>
  </segment>
  <segment nite:id="{mid}.A.seg.2" transcriber_start="30.0" transcriber_end="40.0">
    <nite:child href="{mid}.A.words.xml#id({mid}.A.words2)..id({mid}.A.words3)"/>
  </segment>
</nite:root>
""",
    )
    _write(
        annot_root / "dialogueActs" / f"{mid}.A.dialog-act.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.A.dialog-act">
  <dact nite:id="{mid}.A.da.1">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_4)"/>
    <nite:child href="{mid}.A.words.xml#id({mid}.A.words0)..id({mid}.A.words1)"/>
  </dact>
  <dact nite:id="{mid}.A.da.2">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_4)"/>
    <nite:child href="{mid}.A.words.xml#id({mid}.A.words2)..id({mid}.A.words3)"/>
  </dact>
</nite:root>
""",
    )

    # Speaker B talks 15–25, with one bck at 5.0
    _write(
        annot_root / "words" / f"{mid}.B.words.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.B.words">
  <w nite:id="{mid}.B.words0" starttime="5.0" endtime="5.3">yeah</w>
  <w nite:id="{mid}.B.words1" starttime="15.0" endtime="20.0">my</w>
  <w nite:id="{mid}.B.words2" starttime="20.0" endtime="25.0">turn</w>
</nite:root>
""",
    )
    _write(
        annot_root / "segments" / f"{mid}.B.segments.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.B.segs">
  <segment nite:id="{mid}.B.seg.1" transcriber_start="5.0" transcriber_end="5.3">
    <nite:child href="{mid}.B.words.xml#id({mid}.B.words0)"/>
  </segment>
  <segment nite:id="{mid}.B.seg.2" transcriber_start="15.0" transcriber_end="25.0">
    <nite:child href="{mid}.B.words.xml#id({mid}.B.words1)..id({mid}.B.words2)"/>
  </segment>
</nite:root>
""",
    )
    _write(
        annot_root / "dialogueActs" / f"{mid}.B.dialog-act.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.B.dialog-act">
  <dact nite:id="{mid}.B.da.1">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_1)"/>
    <nite:child href="{mid}.B.words.xml#id({mid}.B.words0)"/>
  </dact>
  <dact nite:id="{mid}.B.da.2">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_4)"/>
    <nite:child href="{mid}.B.words.xml#id({mid}.B.words1)..id({mid}.B.words2)"/>
  </dact>
</nite:root>
""",
    )

    audio_dir = audio_root / mid / "audio"
    _write_wav(audio_dir / f"{mid}.Headset-0.wav", duration_sec=60.0, freq=220.0)
    _write_wav(audio_dir / f"{mid}.Headset-1.wav", duration_sec=60.0, freq=440.0)

    return annot_root, audio_root


class TestAMIDataset:
    def test_basic_item(self, fake_corpus: tuple[Path, Path]) -> None:
        annot_root, audio_root = fake_corpus
        ds = AMIDataset(
            annot_root,
            audio_root,
            meeting_ids=["MEET01"],
            chunk_sec=10.0,
            hop_sec=10.0,
            frame_rate_hz=20,
            horizon_bins=40,
            speakers_override={"MEET01": ("A", "B")},
        )
        assert len(ds) >= 1
        item = ds[0]
        assert "audio" in item
        assert "hazard" in item
        assert "mask" in item
        assert item["audio"].shape == (10 * SAMPLING_RATE, 2)
        # Check all event types present
        assert set(item["hazard"]) == set(EventType)
        for ev in EventType:
            assert item["hazard"][ev].shape == (10 * 20, 40)
            assert item["mask"][ev].shape == (10 * 20, 40)

    def test_chunk_index_count(self, fake_corpus: tuple[Path, Path]) -> None:
        annot_root, audio_root = fake_corpus
        ds = AMIDataset(
            annot_root,
            audio_root,
            meeting_ids=["MEET01"],
            chunk_sec=20.0,
            hop_sec=20.0,
            frame_rate_hz=20,
            horizon_bins=40,
            speakers_override={"MEET01": ("A", "B")},
        )
        # 60s / 20s hop → 3 chunks (and one extra at the end gets padded)
        assert len(ds) >= 3

    def test_dataloader_collates(self, fake_corpus: tuple[Path, Path]) -> None:
        annot_root, audio_root = fake_corpus
        ds = AMIDataset(
            annot_root,
            audio_root,
            meeting_ids=["MEET01"],
            chunk_sec=10.0,
            hop_sec=10.0,
            frame_rate_hz=20,
            horizon_bins=40,
            speakers_override={"MEET01": ("A", "B")},
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ami_collate)
        batch = next(iter(loader))
        assert batch["audio"].shape[0] == 2  # batch dim
        assert batch["audio"].shape[1] == 10 * SAMPLING_RATE
        assert batch["audio"].shape[2] == 2
        for ev in EventType:
            assert batch["hazard"][ev].shape == (2, 10 * 20, 40)
            assert batch["mask"][ev].shape == (2, 10 * 20, 40)

    def test_speakers_default_picks_two_most_active(self, fake_corpus: tuple[Path, Path]) -> None:
        annot_root, audio_root = fake_corpus
        # No override; with only 2 speakers in fake corpus this still works
        ds = AMIDataset(
            annot_root,
            audio_root,
            meeting_ids=["MEET01"],
            chunk_sec=10.0,
            hop_sec=10.0,
            frame_rate_hz=20,
            horizon_bins=40,
        )
        item = ds[0]
        # Speakers should include both A and B (the only two)
        assert set(item["speakers"]) == {"A", "B"}

    def test_shift_target_shape_and_mask(self, fake_corpus: tuple[Path, Path]) -> None:
        annot_root, audio_root = fake_corpus
        ds = AMIDataset(
            annot_root,
            audio_root,
            meeting_ids=["MEET01"],
            chunk_sec=10.0,
            hop_sec=10.0,
            frame_rate_hz=20,
            horizon_bins=40,
            speakers_override={"MEET01": ("A", "B")},
        )
        item = ds[0]
        assert "shift_target" in item
        assert "shift_mask" in item
        assert item["shift_target"].shape == (10 * 20,)
        assert item["shift_mask"].shape == (10 * 20,)
        # Mask values must be 0 or 1
        assert torch.all((item["shift_mask"] == 0) | (item["shift_mask"] == 1))
        # Where mask is 0, target must also be 0
        assert torch.all(item["shift_target"][item["shift_mask"] == 0] == 0)

    def test_compute_shift_targets_unit(self) -> None:
        from itm.data.dataset import _compute_shift_targets

        # 100 frames at 20 Hz = 5s.
        # Frames 0-39: speaker 0 (channel 0 active, 1 silent)
        # Frames 40-69: BOTH silent (mutual silence, 1.5s ≥ 0.2s)
        # Frames 70-99: speaker 1 (channel 1 active) → SHIFT
        vad = torch.zeros(100, 2)
        vad[0:40, 0] = 1.0
        vad[70:100, 1] = 1.0
        target, mask = _compute_shift_targets(vad, frame_rate=20)
        assert mask[40:70].sum() == 30, "expected mask=1 over silence frames"
        assert target[40:70].sum() == 30, "shift case should give target=1 over silence"
        assert mask[:40].sum() == 0  # outside silence
        assert mask[70:].sum() == 0

        # Hold case: speaker 0 resumes after silence
        vad = torch.zeros(100, 2)
        vad[0:40, 0] = 1.0
        vad[70:100, 0] = 1.0
        target, mask = _compute_shift_targets(vad, frame_rate=20)
        assert mask[40:70].sum() == 30
        assert target[40:70].sum() == 0  # hold → target stays 0

        # Silence too short (< 0.2s = 4 frames) → no labelable boundary
        vad = torch.zeros(100, 2)
        vad[0:40, 0] = 1.0
        vad[42:100, 1] = 1.0  # only 2-frame gap, below 0.2s threshold
        target, mask = _compute_shift_targets(vad, frame_rate=20)
        assert mask.sum() == 0

    def test_targets_have_some_events(self, fake_corpus: tuple[Path, Path]) -> None:
        annot_root, audio_root = fake_corpus
        ds = AMIDataset(
            annot_root,
            audio_root,
            meeting_ids=["MEET01"],
            chunk_sec=60.0,  # full meeting in one chunk
            hop_sec=60.0,
            frame_rate_hz=20,
            horizon_bins=40,
            speakers_override={"MEET01": ("A", "B")},
        )
        item = ds[0]
        # Backchannel from B at t=5.0 should produce >=1 positive in hazard
        bc = item["hazard"][EventType.BACKCHANNEL]
        assert bc.sum() > 0, "Expected at least one backchannel event in fake corpus"


class TestEndToEndPipeline:
    def test_loss_runs_on_batch(self, fake_corpus: tuple[Path, Path]) -> None:
        from itm.data.targets import survival_nll_loss

        annot_root, audio_root = fake_corpus
        ds = AMIDataset(
            annot_root,
            audio_root,
            meeting_ids=["MEET01"],
            chunk_sec=10.0,
            hop_sec=10.0,
            frame_rate_hz=20,
            horizon_bins=40,
            speakers_override={"MEET01": ("A", "B")},
        )
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ami_collate)
        batch = next(iter(loader))
        # Pretend a model emitted random hazard logits at the right shape
        hazard_logits = torch.zeros(2, 10 * 20, 40)
        for ev in EventType:
            loss = survival_nll_loss(hazard_logits, batch["hazard"][ev], batch["mask"][ev])
            assert torch.isfinite(loss)

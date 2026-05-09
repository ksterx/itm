"""Unit tests for ``itm.data.ami``.

These tests construct minimal in-memory XML files in a temporary directory
that mirrors the real AMI annotation layout. They do not depend on the
full corpus being downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from itm.data.ami import (
    _parse_href,
    load_meeting,
    parse_da_ontology,
    parse_dialog_acts,
    parse_segments,
    parse_words,
)

NITE_NS = 'xmlns:nite="http://nite.sourceforge.net/"'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


@pytest.fixture()
def fake_corpus(tmp_path: Path) -> Path:
    """A tiny 2-speaker corpus rooted at tmp_path with realistic structure."""
    root = tmp_path / "annotations" / "unpacked"
    mid = "MEET01"

    # da-types ontology
    _write(
        root / "ontologies" / "da-types.xml",
        f"""<?xml version="1.0"?>
<da-type {NITE_NS} nite:id="cmrda" name="da-type">
  <da-type nite:id="ami_daclass_0" name="minor">
    <da-type nite:id="ami_da_1" name="bck"/>
    <da-type nite:id="ami_da_2" name="stl"/>
  </da-type>
  <da-type nite:id="ami_daclass_1" name="task">
    <da-type nite:id="ami_da_4" name="inf"/>
    <da-type nite:id="ami_da_6" name="sug"/>
  </da-type>
</da-type>
""",
    )

    # Speaker A: 5 words at 0.0, 1.0, 2.0, 3.0, 4.0 (each 0.5s long)
    _write(
        root / "words" / f"{mid}.A.words.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.A.words">
  <w nite:id="{mid}.A.words0" starttime="0.0" endtime="0.5">Hello</w>
  <w nite:id="{mid}.A.words1" starttime="1.0" endtime="1.5">world</w>
  <w nite:id="{mid}.A.words2" starttime="2.0" endtime="2.5">how</w>
  <w nite:id="{mid}.A.words3" starttime="3.0" endtime="3.5">are</w>
  <w nite:id="{mid}.A.words4" starttime="4.0" endtime="4.5">you</w>
</nite:root>
""",
    )

    # Speaker A: one segment covering all 5 words, plus a second segment for the substantive turn-shift case
    _write(
        root / "segments" / f"{mid}.A.segments.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.A.segs">
  <segment nite:id="{mid}.A.seg.1" transcriber_start="0.0" transcriber_end="4.5">
    <nite:child href="{mid}.A.words.xml#id({mid}.A.words0)..id({mid}.A.words4)"/>
  </segment>
</nite:root>
""",
    )

    # Speaker A dialog acts: words0 = bck (backchannel), words1..4 = inf (substantive)
    _write(
        root / "dialogueActs" / f"{mid}.A.dialog-act.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.A.dialog-act">
  <dact nite:id="{mid}.A.da.1">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_1)"/>
    <nite:child href="{mid}.A.words.xml#id({mid}.A.words0)"/>
  </dact>
  <dact nite:id="{mid}.A.da.2">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_4)"/>
    <nite:child href="{mid}.A.words.xml#id({mid}.A.words1)..id({mid}.A.words4)"/>
  </dact>
</nite:root>
""",
    )

    # Speaker B: 3 words at 0.2, 1.2, 5.0
    _write(
        root / "words" / f"{mid}.B.words.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.B.words">
  <w nite:id="{mid}.B.words0" starttime="0.2" endtime="0.4">um</w>
  <w nite:id="{mid}.B.words1" starttime="1.2" endtime="1.4">yeah</w>
  <w nite:id="{mid}.B.words2" starttime="5.0" endtime="9.0">substantive</w>
</nite:root>
""",
    )

    _write(
        root / "segments" / f"{mid}.B.segments.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.B.segs">
  <segment nite:id="{mid}.B.seg.1" transcriber_start="0.2" transcriber_end="0.4">
    <nite:child href="{mid}.B.words.xml#id({mid}.B.words0)"/>
  </segment>
  <segment nite:id="{mid}.B.seg.2" transcriber_start="1.2" transcriber_end="1.4">
    <nite:child href="{mid}.B.words.xml#id({mid}.B.words1)"/>
  </segment>
  <segment nite:id="{mid}.B.seg.3" transcriber_start="5.0" transcriber_end="9.0">
    <nite:child href="{mid}.B.words.xml#id({mid}.B.words2)"/>
  </segment>
</nite:root>
""",
    )

    _write(
        root / "dialogueActs" / f"{mid}.B.dialog-act.xml",
        f"""<?xml version="1.0"?>
<nite:root {NITE_NS} nite:id="{mid}.B.dialog-act">
  <dact nite:id="{mid}.B.da.1">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_2)"/>
    <nite:child href="{mid}.B.words.xml#id({mid}.B.words0)"/>
  </dact>
  <dact nite:id="{mid}.B.da.2">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_1)"/>
    <nite:child href="{mid}.B.words.xml#id({mid}.B.words1)"/>
  </dact>
  <dact nite:id="{mid}.B.da.3">
    <nite:pointer role="da-aspect" href="da-types.xml#id(ami_da_4)"/>
    <nite:child href="{mid}.B.words.xml#id({mid}.B.words2)"/>
  </dact>
</nite:root>
""",
    )

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHrefParsing:
    def test_range_href(self) -> None:
        f, s, e = _parse_href("ES2002a.A.words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)")
        assert f == "ES2002a.A.words.xml"
        assert s == "ES2002a.A.words0"
        assert e == "ES2002a.A.words12"

    def test_single_href(self) -> None:
        f, s, e = _parse_href("ES2002a.A.words.xml#id(ES2002a.A.words49)")
        assert f == "ES2002a.A.words.xml"
        assert s == "ES2002a.A.words49"
        assert e is None

    def test_invalid_href(self) -> None:
        with pytest.raises(ValueError):
            _parse_href("not a real href")


class TestParseWords:
    def test_basic(self, fake_corpus: Path) -> None:
        words = parse_words(fake_corpus / "words" / "MEET01.A.words.xml")
        assert len(words) == 5
        assert words["MEET01.A.words0"].text == "Hello"
        assert words["MEET01.A.words0"].start == pytest.approx(0.0)
        assert words["MEET01.A.words4"].end == pytest.approx(4.5)


class TestParseSegments:
    def test_basic(self, fake_corpus: Path) -> None:
        segs = parse_segments(fake_corpus / "segments" / "MEET01.A.segments.xml", speaker="A")
        assert len(segs) == 1
        assert segs[0].speaker == "A"
        assert segs[0].start == pytest.approx(0.0)
        assert segs[0].end == pytest.approx(4.5)
        assert segs[0].duration == pytest.approx(4.5)


class TestDialogActOntology:
    def test_loads_known_types(self, fake_corpus: Path) -> None:
        onto = parse_da_ontology(fake_corpus / "ontologies" / "da-types.xml")
        assert onto["ami_da_1"] == "bck"
        assert onto["ami_da_4"] == "inf"
        assert onto["ami_da_6"] == "sug"


class TestParseDialogActs:
    def test_resolves_word_times(self, fake_corpus: Path) -> None:
        words = parse_words(fake_corpus / "words" / "MEET01.A.words.xml")
        onto = parse_da_ontology(fake_corpus / "ontologies" / "da-types.xml")
        das = parse_dialog_acts(
            fake_corpus / "dialogueActs" / "MEET01.A.dialog-act.xml",
            speaker="A",
            words=words,
            da_ontology=onto,
        )
        assert len(das) == 2
        # First DA: bck on words0 alone
        assert das[0].da_type == "bck"
        assert das[0].start == pytest.approx(0.0)
        assert das[0].end == pytest.approx(0.5)
        # Second DA: inf on words1..words4
        assert das[1].da_type == "inf"
        assert das[1].start == pytest.approx(1.0)
        assert das[1].end == pytest.approx(4.5)
        assert das[1].duration == pytest.approx(3.5)


class TestLoadMeeting:
    def test_full_load(self, fake_corpus: Path) -> None:
        meeting = load_meeting(fake_corpus, "MEET01")
        assert meeting.id == "MEET01"
        assert meeting.speakers == ("A", "B")
        assert len(meeting.words_by_speaker["A"]) == 5
        assert len(meeting.dialog_acts_by_speaker["B"]) == 3

        all_segs = meeting.all_segments()
        assert len(all_segs) == 4
        assert all_segs[0].start <= all_segs[-1].start

        all_das = meeting.all_dialog_acts()
        assert len(all_das) == 5

    def test_missing_meeting(self, fake_corpus: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_meeting(fake_corpus, "DOES_NOT_EXIST")

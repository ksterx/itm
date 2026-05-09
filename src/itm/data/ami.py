"""AMI Meeting Corpus XML annotation parser.

The AMI annotation files use NXT (NITE XML Toolkit) standoff format, where
multiple XML files cross-reference each other via ``nite:child`` and
``nite:pointer`` elements. This module flattens that representation into
plain dataclasses with concrete time spans, ready for downstream label
generation.

Typical use:

    >>> from itm.data.ami import load_meeting
    >>> meeting = load_meeting("data/raw/ami/annotations/unpacked", "ES2002a")
    >>> for da in meeting.dialog_acts_by_speaker["A"][:3]:
    ...     print(da.speaker, da.da_type, da.start, da.end)

The parser only depends on the Python standard library.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

NITE_NS = "{http://nite.sourceforge.net/}"

# Regex for parsing ``nite:child`` href values like:
#   "ES2002a.A.words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)"
#   "ES2002a.A.words.xml#id(ES2002a.A.words49)"
_HREF_RANGE_RE = re.compile(
    r"^(?P<file>[^#]+)#id\((?P<start>[^)]+)\)(?:\.\.id\((?P<end>[^)]+)\))?$"
)


@dataclass(frozen=True, slots=True)
class Word:
    """A word with timing. ``end < start`` is allowed for AMI punctuation tokens."""

    id: str
    start: float
    end: float
    text: str
    is_punctuation: bool = False


@dataclass(frozen=True, slots=True)
class Segment:
    """A speech segment (IPU/utterance unit) for one speaker."""

    id: str
    speaker: str
    start: float
    end: float
    word_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True, slots=True)
class DialogAct:
    """A dialog act labelled span, time-aligned via referenced word IDs.

    ``da_type`` is the short name from the AMI ontology (e.g. ``"bck"``,
    ``"inf"``). ``da_id`` is the original ``ami_da_N`` identifier.
    """

    id: str
    speaker: str
    start: float
    end: float
    da_type: str
    da_id: str
    word_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Meeting:
    """All annotations for one AMI meeting, indexed by speaker."""

    id: str
    speakers: tuple[str, ...]
    words_by_speaker: dict[str, dict[str, Word]]
    segments_by_speaker: dict[str, list[Segment]]
    dialog_acts_by_speaker: dict[str, list[DialogAct]]

    def all_segments(self) -> list[Segment]:
        return sorted(
            (s for segs in self.segments_by_speaker.values() for s in segs),
            key=lambda s: s.start,
        )

    def all_dialog_acts(self) -> list[DialogAct]:
        return sorted(
            (d for das in self.dialog_acts_by_speaker.values() for d in das),
            key=lambda d: d.start,
        )


# ---------------------------------------------------------------------------
# href / id parsing helpers
# ---------------------------------------------------------------------------


def _parse_href(href: str) -> tuple[str, str, str | None]:
    """Parse a ``nite:child`` href. Returns ``(referenced_file, id_start, id_end)``.

    ``id_end`` is ``None`` when the href targets a single id rather than a range.
    """
    m = _HREF_RANGE_RE.match(href.strip())
    if not m:
        raise ValueError(f"Cannot parse NITE href: {href!r}")
    return m.group("file"), m.group("start"), m.group("end")


def _id_range(start_id: str, end_id: str | None, ordered_ids: list[str]) -> list[str]:
    """Resolve ``id(start)..id(end)`` to a contiguous slice of ordered ids.

    AMI word ids in a single file form a continuous sequence, so we look them
    up by index in ``ordered_ids``. Missing ids raise ``KeyError``.
    """
    try:
        i_start = ordered_ids.index(start_id)
    except ValueError as e:
        raise KeyError(f"start id {start_id!r} not in word list") from e
    if end_id is None:
        return [ordered_ids[i_start]]
    try:
        i_end = ordered_ids.index(end_id)
    except ValueError as e:
        raise KeyError(f"end id {end_id!r} not in word list") from e
    if i_end < i_start:
        i_start, i_end = i_end, i_start
    return ordered_ids[i_start : i_end + 1]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_words(path: Path | str) -> dict[str, Word]:
    """Parse one ``<meeting>.<speaker>.words.xml`` file.

    Returns a dict keyed by word id, preserving file order via insertion order.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    words: dict[str, Word] = {}
    for w in root.iter():
        if not w.tag.endswith("}w") and w.tag != "w":
            continue
        wid = w.attrib.get(f"{NITE_NS}id") or w.attrib.get("id")
        if wid is None:
            continue
        try:
            start = float(w.attrib["starttime"])
            end = float(w.attrib["endtime"])
        except (KeyError, ValueError):
            # AMI has occasional words without timing; skip them.
            continue
        text = (w.text or "").strip()
        is_punc = w.attrib.get("punc") == "true"
        words[wid] = Word(id=wid, start=start, end=end, text=text, is_punctuation=is_punc)
    return words


def parse_segments(path: Path | str, speaker: str) -> list[Segment]:
    """Parse one ``<meeting>.<speaker>.segments.xml`` file.

    Segments carry their own ``transcriber_start``/``transcriber_end``
    attributes, so word resolution is optional here.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    segments: list[Segment] = []
    for seg in root.iter():
        if not seg.tag.endswith("}segment") and seg.tag != "segment":
            continue
        sid = seg.attrib.get(f"{NITE_NS}id") or seg.attrib.get("id")
        if sid is None:
            continue
        start_attr = seg.attrib.get("transcriber_start") or seg.attrib.get("starttime")
        end_attr = seg.attrib.get("transcriber_end") or seg.attrib.get("endtime")
        if start_attr is None or end_attr is None:
            continue
        try:
            start = float(start_attr)
            end = float(end_attr)
        except ValueError:
            continue

        # Collect referenced word ids (if any)
        word_ids: list[str] = []
        for child in seg.iter():
            if not child.tag.endswith("}child"):
                continue
            href = child.attrib.get("href")
            if not href:
                continue
            try:
                _, id_start, id_end = _parse_href(href)
            except ValueError:
                continue
            word_ids.append(id_start)
            if id_end and id_end != id_start:
                word_ids.append(id_end)  # range marker; full expansion not needed here

        segments.append(
            Segment(
                id=sid,
                speaker=speaker,
                start=start,
                end=end,
                word_ids=tuple(word_ids),
            )
        )
    segments.sort(key=lambda s: s.start)
    return segments


def parse_da_ontology(path: Path | str) -> dict[str, str]:
    """Parse ``ontologies/da-types.xml`` to a ``{ami_da_N: short_name}`` dict.

    Example: ``{"ami_da_1": "bck", "ami_da_4": "inf", ...}``.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    mapping: dict[str, str] = {}
    for el in root.iter():
        if not el.tag.endswith("}da-type") and el.tag != "da-type":
            continue
        dat_id = el.attrib.get(f"{NITE_NS}id") or el.attrib.get("id")
        name = el.attrib.get("name")
        if dat_id and name and dat_id.startswith("ami_da_"):
            mapping[dat_id] = name
    return mapping


def parse_dialog_acts(
    path: Path | str,
    speaker: str,
    words: dict[str, Word],
    da_ontology: dict[str, str],
) -> list[DialogAct]:
    """Parse ``<meeting>.<speaker>.dialog-act.xml``.

    Time spans are resolved by looking up the referenced word ids in
    ``words``. The ``words`` dict must come from the same speaker's
    words.xml file. Dialog acts whose word ids cannot be resolved
    (e.g. malformed annotations) are skipped.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    ordered_ids = list(words.keys())
    dialog_acts: list[DialogAct] = []

    for d in root.iter():
        if not d.tag.endswith("}dact") and d.tag != "dact":
            continue
        did = d.attrib.get(f"{NITE_NS}id") or d.attrib.get("id")
        if did is None:
            continue

        # da-aspect pointer → ami_da_N → short name
        da_id = ""
        da_type = ""
        for ptr in d.iter():
            if not ptr.tag.endswith("}pointer"):
                continue
            if ptr.attrib.get("role") != "da-aspect":
                continue
            href = ptr.attrib.get("href", "")
            m = re.search(r"id\((ami_da_\d+)\)", href)
            if m:
                da_id = m.group(1)
                da_type = da_ontology.get(da_id, "")
                break

        if not da_type:
            continue

        # Resolve word range for timing
        word_ids: list[str] = []
        for child in d.iter():
            if not child.tag.endswith("}child"):
                continue
            href = child.attrib.get("href")
            if not href:
                continue
            try:
                _, id_start, id_end = _parse_href(href)
            except ValueError:
                continue
            try:
                word_ids.extend(_id_range(id_start, id_end, ordered_ids))
            except KeyError:
                continue

        if not word_ids:
            continue

        word_objs = [words[wid] for wid in word_ids if wid in words]
        if not word_objs:
            continue
        start = min(w.start for w in word_objs)
        end = max(w.end for w in word_objs)

        dialog_acts.append(
            DialogAct(
                id=did,
                speaker=speaker,
                start=start,
                end=end,
                da_type=da_type,
                da_id=da_id,
                word_ids=tuple(word_ids),
            )
        )

    dialog_acts.sort(key=lambda da: da.start)
    return dialog_acts


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def _detect_speakers(annotations_root: Path, meeting_id: str) -> list[str]:
    """Find speaker letters (A, B, C, ...) for which words.xml exists."""
    words_dir = annotations_root / "words"
    speakers: list[str] = []
    for p in sorted(words_dir.glob(f"{meeting_id}.*.words.xml")):
        # Filename: ES2002a.A.words.xml → speaker = "A"
        parts = p.name.split(".")
        if len(parts) >= 4:
            speakers.append(parts[1])
    return speakers


def load_meeting(annotations_root: Path | str, meeting_id: str) -> Meeting:
    """Load all annotations for one AMI meeting.

    Args:
        annotations_root: Path to the unpacked AMI annotations directory
            (the one containing ``words/``, ``segments/``, ``dialogueActs/``,
            ``ontologies/``).
        meeting_id: e.g. ``"ES2002a"``.

    Returns:
        ``Meeting`` with parsed words, segments, and dialog acts indexed by speaker.
    """
    root = Path(annotations_root)
    if not root.is_dir():
        raise FileNotFoundError(f"AMI annotations dir not found: {root}")

    da_ontology = parse_da_ontology(root / "ontologies" / "da-types.xml")
    speakers = _detect_speakers(root, meeting_id)
    if not speakers:
        raise FileNotFoundError(f"No speaker word files found for {meeting_id} under {root}")

    words_by_speaker: dict[str, dict[str, Word]] = {}
    segments_by_speaker: dict[str, list[Segment]] = {}
    dialog_acts_by_speaker: dict[str, list[DialogAct]] = {}

    for spk in speakers:
        words = parse_words(root / "words" / f"{meeting_id}.{spk}.words.xml")
        words_by_speaker[spk] = words

        seg_path = root / "segments" / f"{meeting_id}.{spk}.segments.xml"
        if seg_path.exists():
            segments_by_speaker[spk] = parse_segments(seg_path, spk)
        else:
            segments_by_speaker[spk] = []

        da_path = root / "dialogueActs" / f"{meeting_id}.{spk}.dialog-act.xml"
        if da_path.exists():
            dialog_acts_by_speaker[spk] = parse_dialog_acts(da_path, spk, words, da_ontology)
        else:
            dialog_acts_by_speaker[spk] = []

    return Meeting(
        id=meeting_id,
        speakers=tuple(speakers),
        words_by_speaker=words_by_speaker,
        segments_by_speaker=segments_by_speaker,
        dialog_acts_by_speaker=dialog_acts_by_speaker,
    )

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

# scripts/ is not an installed package (unlike `api`), so it isn't importable via a plain
# `from scripts import ...` under pytest's rootdir-relative sys.path. Load it directly by
# file path instead, matching the convention in tests/test_benchmark_transcription.py.
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "transcription_alignment.py"
_spec = importlib.util.spec_from_file_location("transcription_alignment", _SCRIPT_PATH)
ta = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = ta
_spec.loader.exec_module(ta)

align_words = ta.align_words
diarization_accuracy = ta.diarization_accuracy


@dataclass(frozen=True)
class _Word:
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class _TimedWord:
    text: str
    start: float
    speaker: str | None = None


def test_align_words_matches_in_order_ignoring_case_and_punctuation():
    reference = [_Word("Hello,"), _Word("world.")]
    provider = [_Word("hello"), _Word("world")]

    pairs = align_words(reference, provider)

    assert len(pairs) == 2


def test_align_words_captures_start_times_when_present():
    reference = [_TimedWord("hello", start=0.0), _TimedWord("world", start=0.5)]
    provider = [_TimedWord("hello", start=0.1), _TimedWord("world", start=0.62)]

    pairs = align_words(reference, provider)

    assert [p.reference_start for p in pairs] == [0.0, 0.5]
    assert [p.provider_start for p in pairs] == [0.1, 0.62]


def test_align_words_leaves_start_none_when_either_side_lacks_it():
    reference = [_Word("hello")]  # no `start` attribute at all
    provider = [_TimedWord("hello", start=0.1)]

    pairs = align_words(reference, provider)

    assert pairs[0].reference_start is None
    assert pairs[0].provider_start == 0.1


def test_align_words_skips_unmatched_reference_words():
    reference = [_Word("one"), _Word("two"), _Word("three")]
    provider = [_Word("one"), _Word("three")]

    pairs = align_words(reference, provider)

    assert len(pairs) == 2


def test_diarization_accuracy_none_when_nothing_labeled():
    pairs = align_words([_Word("hi")], [_Word("hi")])

    assert diarization_accuracy(pairs) is None


def test_diarization_accuracy_perfect_under_relabeling():
    # Provider's speaker IDs ("0"/"1") are consistently swapped relative to our
    # reference labels ("A"/"B") -- the optimal permutation should still find 100%.
    reference = [
        _Word("hi", speaker="A"),
        _Word("there", speaker="A"),
        _Word("hey", speaker="B"),
        _Word("back", speaker="B"),
    ]
    provider = [
        _Word("hi", speaker="1"),
        _Word("there", speaker="1"),
        _Word("hey", speaker="0"),
        _Word("back", speaker="0"),
    ]

    pairs = align_words(reference, provider)
    result = diarization_accuracy(pairs)

    assert result is not None
    accuracy, mapping = result
    assert accuracy == 1.0
    assert mapping == {"1": "A", "0": "B"}


def test_diarization_accuracy_partial_when_provider_misattributes_a_word():
    reference = [
        _Word("hi", speaker="A"),
        _Word("there", speaker="A"),
        _Word("hey", speaker="B"),
    ]
    provider = [
        _Word("hi", speaker="1"),
        _Word("there", speaker="0"),  # misattributed
        _Word("hey", speaker="0"),
    ]

    pairs = align_words(reference, provider)
    result = diarization_accuracy(pairs)

    assert result is not None
    accuracy, _mapping = result
    assert accuracy == 2 / 3


def test_diarization_accuracy_handles_provider_under_detecting_speaker_count():
    # Provider only ever emits one speaker label even though the reference has two --
    # a real failure mode for providers with weak diarization. Best mapping can only
    # ever get the majority speaker's words right.
    reference = [
        _Word("a1", speaker="A"),
        _Word("a2", speaker="A"),
        _Word("b1", speaker="B"),
    ]
    provider = [
        _Word("a1", speaker="0"),
        _Word("a2", speaker="0"),
        _Word("b1", speaker="0"),
    ]

    pairs = align_words(reference, provider)
    result = diarization_accuracy(pairs)

    assert result is not None
    accuracy, _mapping = result
    assert accuracy == 2 / 3

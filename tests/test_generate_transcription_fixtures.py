"""Tests for scripts/generate_transcription_fixtures.py's pure word-timing helper.

Never calls the OpenAI TTS API or touches ffmpeg -- those parts of the script (audio
synthesis, noise mixing, mp3 export) require live credentials and a real audio backend and
aren't covered here. This only exercises `_evenly_spaced_words`, the approximation used to
assign per-word start times within a turn's measured audio duration.
"""

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_transcription_fixtures.py"
_spec = importlib.util.spec_from_file_location("generate_transcription_fixtures", _SCRIPT_PATH)
gtf = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = gtf
_spec.loader.exec_module(gtf)


def test_evenly_spaced_words_empty_input():
    assert gtf._evenly_spaced_words([], 10.0, speaker=None) == []


def test_evenly_spaced_words_spans_the_full_duration_in_order():
    words = gtf._evenly_spaced_words(["one", "two", "three", "four"], 4.0, speaker="A")

    assert [w["text"] for w in words] == ["one", "two", "three", "four"]
    assert [w["start"] for w in words] == [0.0, 1.0, 2.0, 3.0]
    assert all(w["speaker"] == "A" for w in words)


def test_evenly_spaced_words_carries_speaker_label_through():
    words = gtf._evenly_spaced_words(["hi"], 1.0, speaker="B")

    assert words == [{"text": "hi", "start": 0.0, "speaker": "B"}]

"""Tests for scripts/eval_providers.py's pure helpers (parsing, cost fallbacks).

Never makes real network calls -- the provider classes themselves (WhisperEvalProvider etc.)
are thin wrappers around SDK calls and aren't covered here; these tests exercise only the
free functions that don't touch a network client: the "Speaker N:" transcript parser and the
per-model cost calculators (both the usage-based and duration-fallback paths).
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# scripts/ is not an installed package (unlike `api`), so it isn't importable via a plain
# `from scripts import ...` under pytest's rootdir-relative sys.path. Load it directly by
# file path instead, matching the convention in tests/test_benchmark_transcription.py.
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_providers.py"
_spec = importlib.util.spec_from_file_location("eval_providers", _SCRIPT_PATH)
ep = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = ep
_spec.loader.exec_module(ep)


def test_parse_speaker_labeled_text_splits_turns_by_speaker():
    raw = "Speaker 1: hello there\nSpeaker 2: hey back\nSpeaker 1: bye now"

    text, words = ep._parse_speaker_labeled_text(raw)

    assert text == "hello there hey back bye now"
    assert [w.speaker for w in words] == ["1", "1", "2", "2", "1", "1"]
    assert [w.text for w in words] == ["hello", "there", "hey", "back", "bye", "now"]
    assert all(w.start is None and w.end is None for w in words)


def test_parse_speaker_labeled_text_keeps_unlabeled_lines_as_text_only():
    raw = "(inaudible)\nSpeaker 1: hello"

    text, words = ep._parse_speaker_labeled_text(raw)

    assert "(inaudible)" in text
    assert len(words) == 1  # only the labeled line contributes speaker-tagged words


def test_words_from_diarized_segments_splits_evenly_within_each_segment():
    segments = [
        SimpleNamespace(text="hi there", start=0.0, end=2.0, speaker="A"),
        SimpleNamespace(text="hey", start=2.5, end=3.0, speaker="B"),
    ]

    words = ep._words_from_diarized_segments(segments)

    assert [(w.text, w.speaker) for w in words] == [("hi", "A"), ("there", "A"), ("hey", "B")]
    assert words[0].start == 0.0
    assert words[1].start == 1.0  # second of two words evenly split across a 2s segment
    assert words[2].start == 2.5


def test_words_from_diarized_segments_skips_empty_text_segments():
    segments = [SimpleNamespace(text="   ", start=0.0, end=1.0, speaker="A")]

    words = ep._words_from_diarized_segments(segments)

    assert words == ()


def test_words_from_azure_phrases_converts_milliseconds_and_speaker_when_diarizing():
    phrases = [
        {
            "speaker": 1,
            "text": "hi there",
            "words": [
                {"text": "hi", "offsetMilliseconds": 100, "durationMilliseconds": 200},
                {"text": "there", "offsetMilliseconds": 300, "durationMilliseconds": 400},
            ],
        },
        {
            "speaker": 2,
            "text": "hey",
            "words": [{"text": "hey", "offsetMilliseconds": 900, "durationMilliseconds": 300}],
        },
    ]

    words = ep._words_from_azure_phrases(phrases, diarize=True)

    assert [(w.text, w.speaker) for w in words] == [("hi", "1"), ("there", "1"), ("hey", "2")]
    assert words[0].start == pytest.approx(0.1)
    assert words[0].end == pytest.approx(0.3)


def test_words_from_azure_phrases_omits_speaker_when_not_diarizing():
    phrases = [{"speaker": 1, "text": "hi", "words": [{"text": "hi", "offsetMilliseconds": 0, "durationMilliseconds": 200}]}]

    words = ep._words_from_azure_phrases(phrases, diarize=False)

    assert words[0].speaker is None


def test_normalize_azure_locale_maps_bare_iso_code_to_bcp47():
    assert ep._normalize_azure_locale("en") == "en-US"


def test_normalize_azure_locale_passes_through_full_locale():
    assert ep._normalize_azure_locale("en-GB") == "en-GB"


def test_normalize_azure_locale_passes_through_unknown_bare_code():
    assert ep._normalize_azure_locale("xx") == "xx"


def test_cost_from_openai_transcribe_usage_prefers_real_usage_tokens():
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200)

    cost, basis = ep._cost_from_openai_transcribe_usage("gpt-4o-transcribe", usage, fallback_duration_seconds=60.0)

    expected = (1000 / 1_000_000) * ep.GPT4O_TRANSCRIBE_USD_PER_1M_INPUT_AUDIO_TOKENS + (
        200 / 1_000_000
    ) * ep.GPT4O_TRANSCRIBE_USD_PER_1M_OUTPUT_TOKENS
    assert cost == pytest.approx(expected)
    assert basis == "usage_tokens"


def test_cost_from_openai_transcribe_usage_falls_back_to_duration_when_usage_missing():
    cost, basis = ep._cost_from_openai_transcribe_usage("gpt-4o-mini-transcribe", None, fallback_duration_seconds=120.0)

    assert cost == pytest.approx(2.0 * ep.GPT4O_MINI_TRANSCRIBE_USD_PER_MINUTE_FALLBACK)
    assert basis == "duration_estimate"


def test_cost_from_openai_transcribe_usage_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model_id"):
        ep._cost_from_openai_transcribe_usage("not-a-real-model", None, fallback_duration_seconds=1.0)


def test_cost_from_openai_chat_audio_usage_prefers_real_usage_tokens():
    usage = SimpleNamespace(
        prompt_tokens_details=SimpleNamespace(audio_tokens=3000),
        completion_tokens=50,
    )

    cost, basis = ep._cost_from_openai_chat_audio_usage(usage, fallback_duration_seconds=60.0)

    expected = (3000 / 1_000_000) * ep.OPENAI_AUDIO_CHAT_USD_PER_1M_INPUT_AUDIO_TOKENS + (
        50 / 1_000_000
    ) * ep.OPENAI_AUDIO_CHAT_USD_PER_1M_OUTPUT_TEXT_TOKENS
    assert cost == pytest.approx(expected)
    assert basis == "usage_tokens"


def test_cost_from_openai_chat_audio_usage_falls_back_to_duration_estimate():
    cost, basis = ep._cost_from_openai_chat_audio_usage(None, fallback_duration_seconds=60.0)

    estimated_tokens = 60.0 * ep.OPENAI_AUDIO_CHAT_FALLBACK_AUDIO_TOKENS_PER_SECOND
    expected = (estimated_tokens / 1_000_000) * ep.OPENAI_AUDIO_CHAT_USD_PER_1M_INPUT_AUDIO_TOKENS
    assert cost == pytest.approx(expected)
    assert basis == "duration_estimate"


def test_cost_from_gemini_usage_prefers_real_usage_tokens():
    usage_metadata = SimpleNamespace(prompt_token_count=1500, candidates_token_count=100)

    cost, basis = ep._cost_from_gemini_usage(usage_metadata, fallback_duration_seconds=60.0)

    expected = (1500 / 1_000_000) * ep.GEMINI_USD_PER_1M_INPUT_AUDIO_TOKENS + (
        100 / 1_000_000
    ) * ep.GEMINI_USD_PER_1M_OUTPUT_TEXT_TOKENS
    assert cost == pytest.approx(expected)
    assert basis == "usage_tokens"


def test_cost_from_gemini_usage_falls_back_to_duration_estimate():
    cost, basis = ep._cost_from_gemini_usage(None, fallback_duration_seconds=60.0)

    estimated_tokens = 60.0 * ep.GEMINI_FALLBACK_AUDIO_TOKENS_PER_SECOND
    expected = (estimated_tokens / 1_000_000) * ep.GEMINI_USD_PER_1M_INPUT_AUDIO_TOKENS
    assert cost == pytest.approx(expected)
    assert basis == "duration_estimate"


def test_audio_format_for_openai_chat_accepts_wav_and_mp3():
    assert ep._audio_format_for_openai_chat("clip.wav") == "wav"
    assert ep._audio_format_for_openai_chat("clip.mp3") == "mp3"


def test_audio_format_for_openai_chat_rejects_unsupported_suffix():
    with pytest.raises(ValueError, match="wav/mp3"):
        ep._audio_format_for_openai_chat("clip.m4a")


def test_mime_type_for_gemini_covers_common_suffixes():
    assert ep._mime_type_for_gemini("clip.wav") == "audio/wav"
    assert ep._mime_type_for_gemini("clip.mp3") == "audio/mpeg"
    assert ep._mime_type_for_gemini("clip.m4a") == "audio/mp4"


def test_mime_type_for_gemini_rejects_unsupported_suffix():
    with pytest.raises(ValueError, match="unsupported audio suffix"):
        ep._mime_type_for_gemini("clip.ogg")

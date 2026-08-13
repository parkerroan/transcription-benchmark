"""Tests for scripts/benchmark_multimodal_transcription.py.

Runs entirely offline: no network calls, no real TTS/STT providers. Fake providers stand in
for the real ones, and the fixtures-safety boundary is exercised by monkeypatching the module
under test's own FIXTURES_DIR (it owns ensure_within_fixtures/percentile/mean_or_none/
keyterm_miss_rate directly -- no separate helper script to depend on).

Each script under test is loaded by file path and registered in sys.modules under its own
name *before* the next one is loaded, so that scripts/benchmark_multimodal_transcription.py's
top-level `import eval_providers as ep` / `from transcription_alignment import ...` resolve to
these same monkeypatchable module objects rather than re-importing fresh copies.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ep = _load_module("eval_providers", "eval_providers.py")
_load_module("transcription_alignment", "transcription_alignment.py")
bmt = _load_module("benchmark_multimodal_transcription", "benchmark_multimodal_transcription.py")


@pytest.fixture
def fixtures_dir(tmp_path, monkeypatch):
    """An isolated stand-in for tests/integration/fixtures/, wired into the module under test."""
    fixtures = tmp_path / "tests" / "integration" / "fixtures"
    fixtures.mkdir(parents=True)
    monkeypatch.setattr(bmt, "FIXTURES_DIR", fixtures.resolve())
    return fixtures


class FakeProvider:
    """A fake provider returning a fixed, predictable EvalTranscript."""

    provider_id = "fake"
    model_id = "fake-model"

    def __init__(self, transcript, *, supports_word_timestamps=True, can_attempt_diarization=False):
        self._transcript = transcript
        self.supports_word_timestamps = supports_word_timestamps
        self.can_attempt_diarization = can_attempt_diarization

    def transcribe(self, path, language, *, diarize=False):  # noqa: ARG002 — signature match only
        return self._transcript


class RaisingProvider:
    """A fake provider whose transcribe() always fails with a message that must never leak."""

    provider_id = "fake"
    model_id = "fake-raising"
    supports_word_timestamps = False
    can_attempt_diarization = False

    def transcribe(self, path, language, *, diarize=False):  # noqa: ARG002 — signature match only
        raise RuntimeError("super-secret-provider-error-detail")


def _write_manifest(fixtures_dir, *, fixture_type="single_speaker", reference_words=None, keyterms=()):
    (fixtures_dir / "sample.mp3").write_bytes(b"fake-audio-bytes")
    manifest_path = fixtures_dir / "manifest.jsonl"
    record = {
        "audio_path": "sample.mp3",
        "reference_text": "hello world",
        "language": "en",
        "reference_words": reference_words or [],
        "keyterms": list(keyterms),
        "fixture_type": fixture_type,
    }
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return bmt.load_manifest(manifest_path)


def test_load_manifest_requires_fixture_type(fixtures_dir):
    (fixtures_dir / "sample.mp3").write_bytes(b"fake-audio-bytes")
    manifest_path = fixtures_dir / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"audio_path": "sample.mp3", "reference_text": "hi"}) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="fixture_type"):
        bmt.load_manifest(manifest_path)


def test_load_manifest_parses_speaker_field(fixtures_dir):
    entries = _write_manifest(
        fixtures_dir,
        fixture_type="multi_speaker",
        reference_words=[{"text": "hi", "start": 0.0, "speaker": "A"}],
    )

    assert entries[0].reference_words[0].speaker == "A"
    assert entries[0].fixture_type == "multi_speaker"


def test_run_benchmark_records_failures_without_leaking_error_detail(fixtures_dir):
    entries = _write_manifest(fixtures_dir)
    providers = {"raising": RaisingProvider()}

    report = bmt.run_benchmark(entries, providers)

    fixture_report = report["providers"]["raising"]["fixtures"][0]
    assert fixture_report["error_type"] == "RuntimeError"
    assert "super-secret" not in json.dumps(report)
    assert report["providers"]["raising"]["summary"]["failure_count"] == 1


def test_run_benchmark_computes_wer_keyterms_drift_and_cost(fixtures_dir):
    entries = _write_manifest(
        fixtures_dir,
        reference_words=[
            {"text": "hello", "start": 0.0, "speaker": None},
            {"text": "world", "start": 0.5, "speaker": None},
        ],
        keyterms=["hello"],
    )
    transcript = ep.EvalTranscript(
        text="hello world",
        words=(ep.EvalWord("hello", 0.02, 0.3, None), ep.EvalWord("world", 0.55, 0.9, None)),
        supports_word_timestamps=True,
        supports_diarization=False,
        cost_usd=0.001,
        cost_basis="flat_rate",
        duration_seconds=60.0,
    )
    providers = {"fake": FakeProvider(transcript)}

    report = bmt.run_benchmark(entries, providers)

    fixture_report = report["providers"]["fake"]["fixtures"][0]
    assert fixture_report["wer"] == 0.0
    assert fixture_report["keyterm_miss_rate"] == 0.0
    assert fixture_report["cost_usd"] == 0.001
    assert fixture_report["cost_basis"] == "flat_rate"
    assert fixture_report["cost_usd_per_minute"] == pytest.approx(0.001)  # duration is exactly 1 minute
    assert fixture_report["median_drift_seconds"] == pytest.approx(0.035, abs=1e-6)
    assert report["providers"]["fake"]["summary"]["total_cost_usd"] == pytest.approx(0.001)
    assert report["providers"]["fake"]["summary"]["mean_cost_usd_per_minute"] == pytest.approx(0.001)


def test_run_benchmark_scores_diarization_for_multi_speaker_fixtures(fixtures_dir):
    entries = _write_manifest(
        fixtures_dir,
        fixture_type="multi_speaker",
        reference_words=[
            {"text": "hi", "start": 0.0, "speaker": "A"},
            {"text": "there", "start": 0.3, "speaker": "A"},
            {"text": "hey", "start": 1.0, "speaker": "B"},
        ],
    )
    transcript = ep.EvalTranscript(
        text="hi there hey",
        words=(
            ep.EvalWord("hi", None, None, "1"),
            ep.EvalWord("there", None, None, "1"),
            ep.EvalWord("hey", None, None, "0"),
        ),
        supports_word_timestamps=False,
        supports_diarization=True,
        cost_usd=0.01,
        cost_basis="usage_tokens",
        duration_seconds=5.0,
    )
    providers = {"fake": FakeProvider(transcript, supports_word_timestamps=False, can_attempt_diarization=True)}

    report = bmt.run_benchmark(entries, providers)

    fixture_report = report["providers"]["fake"]["fixtures"][0]
    assert fixture_report["diarization_accuracy"] == 1.0
    assert fixture_report["diarization_label_mapping"] == {"1": "A", "0": "B"}
    assert report["providers"]["fake"]["summary"]["mean_diarization_accuracy"] == 1.0


def test_run_benchmark_leaves_diarization_accuracy_none_when_provider_cant_diarize(fixtures_dir):
    entries = _write_manifest(
        fixtures_dir,
        fixture_type="multi_speaker",
        reference_words=[{"text": "hi", "start": 0.0, "speaker": "A"}],
    )
    transcript = ep.EvalTranscript(
        text="hi",
        words=(ep.EvalWord("hi", None, None, None),),
        supports_word_timestamps=False,
        supports_diarization=False,
        cost_usd=0.0006,
        cost_basis="flat_rate",
        duration_seconds=6.0,
    )
    providers = {"fake": FakeProvider(transcript, supports_word_timestamps=False, can_attempt_diarization=False)}

    report = bmt.run_benchmark(entries, providers)

    fixture_report = report["providers"]["fake"]["fixtures"][0]
    assert fixture_report["diarization_accuracy"] is None
    assert report["providers"]["fake"]["summary"]["mean_diarization_accuracy"] is None


def test_run_benchmark_reports_static_provider_capabilities(fixtures_dir):
    entries = _write_manifest(fixtures_dir)
    providers = {"raising": RaisingProvider()}

    report = bmt.run_benchmark(entries, providers)

    assert report["providers"]["raising"]["model_id"] == "fake-raising"
    assert report["providers"]["raising"]["supports_word_timestamps"] is False
    assert report["providers"]["raising"]["can_attempt_diarization"] is False

#!/usr/bin/env python3
"""Benchmark dedicated ASR models against general multimodal chat models on cost, accuracy,
latency, and speaker diarization -- built for the "multimodal vs. transcription models" blog
comparison, run against fixtures from scripts/generate_transcription_fixtures.py.

Providers (see scripts/eval_providers.py for details/pricing sources -- every model id was
verified live against each vendor's current model list on 2026-08-13, not assumed from memory):
    whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe, gpt-4o-transcribe-diarize, gpt-audio
    (OpenAI); gemini-3.7-flash (Google); grok-stt (xAI); azure-fast-transcription (Microsoft).
    Each is skipped automatically if its credential isn't set (OPENAI_API_KEY / XAI_API_KEY /
    GEMINI_API_KEY / AZURE_SPEECH_KEY + AZURE_SPEECH_REGION).

Manifest format (JSON Lines -- one object per line), written by
scripts/generate_transcription_fixtures.py:
    {
      "audio_path": "clean-narration.mp3",
      "reference_text": "...",
      "language": "en",
      "reference_words": [{"text": "...", "start": 0.0, "speaker": null}, ...],
      "keyterms": ["..."],
      "fixture_type": "single_speaker" | "noisy" | "multi_speaker" | "real_world"
    }
`reference_words[].start` may be null for fixtures with only text/speaker ground truth and no
verified per-word timing (e.g. a real-world clip with an officially transcribed but not
forced-aligned reference) -- timestamp-drift scoring is simply skipped for those words.

Safety: both --manifest and every resolved audio_path must be descendants of the repository's
tests/integration/fixtures/ directory. This is a hard guardrail so this tool can never be
pointed at files outside the repo's own evaluation fixtures by mistake.

For multi_speaker/real_world fixtures, providers are asked to diarize (diarize=True) and
scored on speaker-label accuracy via scripts/transcription_alignment.py's permutation-matched
scorer; providers with EvalTranscript.supports_diarization == False are recorded as
"diarization_accuracy": null rather than skipped, so the report itself documents which
providers can't do this at all.

Usage:
    uv run python scripts/benchmark_multimodal_transcription.py \\
        --manifest tests/integration/fixtures/multimodal_eval/manifest.jsonl \\
        --output /tmp/multimodal-transcription-benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import eval_providers as ep
from dotenv import load_dotenv
from transcription_alignment import align_words, diarization_accuracy

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = (REPO_ROOT / "tests" / "integration" / "fixtures").resolve()


def ensure_within_fixtures(path: Path, *, label: str) -> Path:
    """Resolve `path` and raise ValueError unless it is a descendant of FIXTURES_DIR."""
    resolved = path.resolve()
    if not resolved.is_relative_to(FIXTURES_DIR):
        raise ValueError(
            f"{label} must resolve inside {FIXTURES_DIR} (got {resolved}). "
            "This tool refuses to run against paths outside the repo's own evaluation "
            "fixtures directory."
        )
    return resolved


def keyterm_miss_rate(keyterms: tuple[str, ...], transcript_text: str) -> float | None:
    if not keyterms:
        return None
    haystack = transcript_text.lower()
    misses = sum(1 for term in keyterms if term.lower() not in haystack)
    return misses / len(keyterms)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


@dataclass(frozen=True)
class ReferenceWord:
    text: str
    start: float | None  # None for fixtures with only text/speaker ground truth, no verified timing
    speaker: str | None


@dataclass(frozen=True)
class ManifestEntry:
    audio_path: Path
    reference_text: str
    language: str | None
    reference_words: tuple[ReferenceWord, ...]
    keyterms: tuple[str, ...]
    fixture_type: str


def load_manifest(manifest_path: Path) -> list[ManifestEntry]:
    manifest_dir = manifest_path.parent
    entries: list[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest line {line_number} is not valid JSON: {exc}") from exc

            missing = [key for key in ("audio_path", "reference_text", "fixture_type") if key not in record]
            if missing:
                raise ValueError(f"manifest line {line_number} is missing required field(s): {missing}")

            audio_path = ensure_within_fixtures(
                manifest_dir / record["audio_path"],
                label=f"audio_path on manifest line {line_number}",
            )
            reference_words = tuple(
                ReferenceWord(
                    text=word["text"],
                    start=float(word["start"]) if word.get("start") is not None else None,
                    speaker=word.get("speaker"),
                )
                for word in record.get("reference_words", [])
            )
            entries.append(ManifestEntry(
                audio_path=audio_path,
                reference_text=record["reference_text"],
                language=record.get("language"),
                reference_words=reference_words,
                keyterms=tuple(record.get("keyterms", [])),
                fixture_type=record["fixture_type"],
            ))
    return entries


def run_benchmark(manifest_entries: list[ManifestEntry], providers: dict[str, ep.EvalProvider]) -> dict[str, Any]:
    import jiwer  # noqa: PLC0415  (deferred: only needed when actually benchmarking)

    per_provider_report: dict[str, Any] = {}

    for provider_id, provider in providers.items():
        fixture_reports: list[dict[str, Any]] = []
        wer_values: list[float] = []
        miss_rate_values: list[float] = []
        all_drifts: list[float] = []
        diarization_accuracies: list[float] = []
        cost_values: list[float] = []
        cost_per_minute_values: list[float] = []
        failure_count = 0

        for entry in manifest_entries:
            fixture_label = entry.audio_path.name
            wants_diarization = entry.fixture_type in ("multi_speaker", "real_world")

            started = time.perf_counter()
            try:
                result = provider.transcribe(str(entry.audio_path), entry.language, diarize=wants_diarization)
            except Exception as exc:  # noqa: BLE001 — recorded per-fixture, not fatal to the run
                failure_count += 1
                fixture_reports.append({
                    "fixture": fixture_label,
                    "fixture_type": entry.fixture_type,
                    "error_type": type(exc).__name__,
                })
                continue
            latency_seconds = time.perf_counter() - started

            wer = jiwer.wer(entry.reference_text, result.text) if result.text.strip() else 1.0
            miss_rate = keyterm_miss_rate(entry.keyterms, result.text)

            pairs = align_words(entry.reference_words, result.words)
            drifts = [
                abs(pair.provider_start - pair.reference_start)
                for pair in pairs
                if pair.reference_start is not None and pair.provider_start is not None
            ]

            speaker_accuracy: float | None = None
            speaker_mapping: dict[str, str] | None = None
            if wants_diarization and result.supports_diarization:
                scored = diarization_accuracy(pairs)
                if scored is not None:
                    speaker_accuracy, speaker_mapping = scored
                    diarization_accuracies.append(speaker_accuracy)
            # else: provider can't diarize at all (or the prompt-parsed transcript never
            # produced a labeled word) -- leave speaker_accuracy as None, which the report
            # distinguishes from "0.0 accuracy" (a real attempt that scored zero).

            duration_seconds = result.duration_seconds
            cost_usd_per_minute = result.cost_usd / (duration_seconds / 60.0) if duration_seconds > 0 else None

            wer_values.append(wer)
            if miss_rate is not None:
                miss_rate_values.append(miss_rate)
            all_drifts.extend(drifts)
            cost_values.append(result.cost_usd)
            if cost_usd_per_minute is not None:
                cost_per_minute_values.append(cost_usd_per_minute)

            fixture_reports.append({
                "fixture": fixture_label,
                "fixture_type": entry.fixture_type,
                "duration_seconds": duration_seconds,
                "wer": wer,
                "keyterm_miss_rate": miss_rate,
                "matched_word_count": len(pairs),
                "reference_word_count": len(entry.reference_words),
                "median_drift_seconds": statistics.median(drifts) if drifts else None,
                "p95_drift_seconds": percentile(drifts, 95),
                "diarization_accuracy": speaker_accuracy,
                "diarization_label_mapping": speaker_mapping,
                "latency_seconds": latency_seconds,
                "cost_usd": result.cost_usd,
                "cost_usd_per_minute": cost_usd_per_minute,
                "cost_basis": result.cost_basis,
            })

        per_provider_report[provider_id] = {
            "model_id": provider.model_id,
            "vendor": provider.provider_id,
            "supports_word_timestamps": provider.supports_word_timestamps,
            "can_attempt_diarization": provider.can_attempt_diarization,
            "fixtures": fixture_reports,
            "summary": {
                "fixture_count": len(manifest_entries),
                "failure_count": failure_count,
                "mean_wer": mean_or_none(wer_values),
                "mean_keyterm_miss_rate": mean_or_none(miss_rate_values),
                "median_drift_seconds": statistics.median(all_drifts) if all_drifts else None,
                "p95_drift_seconds": percentile(all_drifts, 95),
                "mean_diarization_accuracy": mean_or_none(diarization_accuracies),
                "total_cost_usd": sum(cost_values),
                "mean_cost_usd_per_fixture": mean_or_none(cost_values),
                "mean_cost_usd_per_minute": mean_or_none(cost_per_minute_values),
            },
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture_count": len(manifest_entries),
        "providers": per_provider_report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help=(
            "Path to a JSON Lines manifest. Must resolve inside tests/integration/fixtures/. "
            "Each line's audio_path is resolved relative to the manifest file's own directory."
        ),
    )
    parser.add_argument("--output", required=True, type=Path, help="Path to write the aggregate JSON report to.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_arg_parser().parse_args(argv)

    try:
        manifest_path = ensure_within_fixtures(args.manifest, label="--manifest")
        if not manifest_path.exists():
            raise ValueError(f"manifest not found: {manifest_path}")
        manifest_entries = load_manifest(manifest_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    providers = ep.available_eval_providers(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        xai_api_key=os.getenv("XAI_API_KEY", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        azure_api_key=os.getenv("AZURE_SPEECH_KEY", ""),
        azure_region=os.getenv("AZURE_SPEECH_REGION", ""),
    )
    if not providers:
        print(
            "error: no providers configured (set OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY, "
            "and/or AZURE_SPEECH_KEY + AZURE_SPEECH_REGION)",
            file=sys.stderr,
        )
        return 1
    print(f"running {len(manifest_entries)} fixtures against: {', '.join(sorted(providers))}")

    report = run_benchmark(manifest_entries, providers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

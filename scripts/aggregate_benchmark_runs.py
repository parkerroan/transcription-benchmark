#!/usr/bin/env python3
"""Averages multiple scripts/benchmark_multimodal_transcription.py JSON reports (repeated runs
of the same manifest) into one report with the same schema, for more stable "final" charts.

Per (provider, fixture) pair, each numeric field is averaged across the runs where that fixture
succeeded; a run where a fixture errored contributes to that fixture's failure_count but is
skipped for the numeric average rather than treated as 0. Per-provider summary stats are
recomputed as the mean of each run's own summary stats (not re-derived from the averaged
fixtures), so total_cost_usd reports the mean cost of one full pass across all fixtures.

Usage:
    uv run python scripts/aggregate_benchmark_runs.py \\
        --reports /tmp/mt-run-1.json /tmp/mt-run-2.json ... \\
        --output /tmp/mt-aggregated.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Fields averaged across runs for each (provider, fixture) pair. Non-numeric fields
# (fixture, fixture_type, diarization_label_mapping) are carried over from the first
# successful run rather than averaged.
_FIXTURE_NUMERIC_FIELDS = (
    "duration_seconds",
    "wer",
    "keyterm_miss_rate",
    "matched_word_count",
    "reference_word_count",
    "median_drift_seconds",
    "p95_drift_seconds",
    "diarization_accuracy",
    "latency_seconds",
    "cost_usd",
    "cost_usd_per_minute",
)

_SUMMARY_NUMERIC_FIELDS = (
    "mean_wer",
    "mean_keyterm_miss_rate",
    "median_drift_seconds",
    "p95_drift_seconds",
    "mean_diarization_accuracy",
    "total_cost_usd",
    "mean_cost_usd_per_fixture",
    "mean_cost_usd_per_minute",
)


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _aggregate_fixtures(runs_fixtures: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """runs_fixtures[i] is the `fixtures` list from run i for one provider, in fixture order."""
    fixture_count = len(runs_fixtures[0])
    aggregated: list[dict[str, Any]] = []

    for idx in range(fixture_count):
        per_run_entries = [run[idx] for run in runs_fixtures if idx < len(run)]
        successful = [e for e in per_run_entries if "error_type" not in e]
        failure_count = len(per_run_entries) - len(successful)

        if not successful:
            aggregated.append({
                **per_run_entries[0],
                "run_count": len(per_run_entries),
                "run_failure_count": failure_count,
            })
            continue

        template = successful[0]
        entry: dict[str, Any] = {
            "fixture": template["fixture"],
            "fixture_type": template["fixture_type"],
            "diarization_label_mapping": template.get("diarization_label_mapping"),
            "cost_basis": template.get("cost_basis"),
            "run_count": len(per_run_entries),
            "run_success_count": len(successful),
            "run_failure_count": failure_count,
        }
        for field in _FIXTURE_NUMERIC_FIELDS:
            values = [e[field] for e in successful if e.get(field) is not None]
            entry[field] = _mean_or_none(values)
        aggregated.append(entry)

    return aggregated


def _aggregate_summary(runs_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "fixture_count": runs_summaries[0]["fixture_count"],
        "run_count": len(runs_summaries),
        "total_failure_count_across_runs": sum(s["failure_count"] for s in runs_summaries),
        "mean_failure_count_per_run": _mean_or_none([float(s["failure_count"]) for s in runs_summaries]),
    }
    for field in _SUMMARY_NUMERIC_FIELDS:
        values = [s[field] for s in runs_summaries if s.get(field) is not None]
        summary[field] = _mean_or_none(values)
    return summary


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("no reports to aggregate")

    provider_ids = list(reports[0]["providers"].keys())
    for i, report in enumerate(reports[1:], start=2):
        if list(report["providers"].keys()) != provider_ids:
            raise ValueError(
                f"report {i} has a different provider set than report 1 "
                f"({list(report['providers'].keys())} vs {provider_ids}) -- runs must share a manifest/config"
            )

    aggregated_providers: dict[str, Any] = {}
    for provider_id in provider_ids:
        runs_for_provider = [report["providers"][provider_id] for report in reports]
        aggregated_providers[provider_id] = {
            "model_id": runs_for_provider[0]["model_id"],
            "vendor": runs_for_provider[0]["vendor"],
            "supports_word_timestamps": runs_for_provider[0]["supports_word_timestamps"],
            "can_attempt_diarization": runs_for_provider[0]["can_attempt_diarization"],
            "fixtures": _aggregate_fixtures([r["fixtures"] for r in runs_for_provider]),
            "summary": _aggregate_summary([r["summary"] for r in runs_for_provider]),
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_count": len(reports),
        "source_generated_at": [r["generated_at"] for r in reports],
        "fixture_count": reports[0]["fixture_count"],
        "providers": aggregated_providers,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", required=True, nargs="+", type=Path, help="Paths to the individual run JSON reports.")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the aggregated JSON report to.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.reports]
    aggregated = aggregate_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
    print(f"aggregated {len(reports)} runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

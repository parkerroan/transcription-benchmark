#!/usr/bin/env python3
"""Renders PNG charts from a scripts/benchmark_multimodal_transcription.py JSON report, plus a
static controls-comparison table that doesn't need a report at all (it's sourced from vendor
docs/pricing pages, not from a live run -- see CONTROLS_TABLE below).

Usage:
    uv run python scripts/plot_multimodal_transcription_benchmark.py \\
        --report /tmp/recastr-multimodal-transcription-benchmark.json \\
        --out-dir /tmp/recastr-transcription-charts

    # Just the controls table, no report needed:
    uv run python scripts/plot_multimodal_transcription_benchmark.py --controls-table-only \\
        --out-dir /tmp/recastr-transcription-charts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Controls table -- model landscape verified live against each vendor's current model list on
# 2026-08-13 (several models originally assumed from training-data memory were already dead --
# see scripts/eval_providers.py's module docstring). $/min figures are real measured averages
# from tests/integration/fixtures/multimodal_eval/ (see /tmp/recastr-multimodal-transcription-
# benchmark.json), not vendor list-price estimates -- re-run scripts/benchmark_multimodal_
# transcription.py and regenerate this table if pricing or the fixture set changes.
# ---------------------------------------------------------------------------
CONTROLS_TABLE_COLUMNS = ("Model", "Vendor", "Word timestamps", "Native diarization", "Streaming", "Steerability", "Measured $/min")
CONTROLS_TABLE_ROWS = (
    ("Whisper-1", "OpenAI", "Yes", "No", "No (batch only)", "Vocabulary hint only", "$0.0060"),
    ("gpt-4o-transcribe", "OpenAI", "No", "No", "Yes", "Vocabulary hint only", "$0.0035"),
    ("gpt-4o-mini-transcribe", "OpenAI", "No", "No", "Yes", "Vocabulary hint only", "$0.0018"),
    ("gpt-4o-transcribe-diarize", "OpenAI", "No (segment-level)", "Yes (native)", "Yes", "Vocabulary hint only", "$0.0192"),
    ("gpt-audio", "OpenAI", "No (promptable, unreliable)", "Promptable only", "Text-token only", "Full NL prompt", "$0.0214"),
    ("Gemini 3.7 Flash", "Google", "No", "Promptable only", "Yes (separate Live API)", "Full NL prompt", "$0.0021"),
    ("Grok STT", "xAI", "Yes", "Yes (native)", "Yes", "Structured params only", "$0.0017"),
    ("Azure Fast Transcription", "Microsoft", "Yes", "Yes (native, up to 35 speakers)", "Yes (separate real-time tier)", "Structured params only", "$0.0060"),
)

_PALETTE = {
    "openai": "#10A37F",
    "google": "#4285F4",
    "xai": "#000000",
    "azure": "#0078D4",
    "fake": "#999999",
}


def _color_for(provider_id: str) -> str:
    return _PALETTE.get(provider_id, "#6B6352")


def plot_controls_table(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=CONTROLS_TABLE_ROWS,
        colLabels=CONTROLS_TABLE_COLUMNS,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(CONTROLS_TABLE_COLUMNS))))
    table.scale(1, 1.8)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#1A1A12")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F5F0E8" if row % 2 else "#FFFFFF")
    ax.set_title("Transcription model controls comparison (verified live 2026-08-13; $/min measured, not list price)", fontsize=11, pad=10)

    out_path = out_dir / "controls_comparison_table.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


# Every chart below splits into exactly these two buckets rather than the finer single_speaker/
# noisy/multi_speaker/real_world fixture_type breakdown -- "synthetic" folds the three
# TTS-generated fixture_types together since they share the thing that actually matters here
# (studio-clean audio, exact-by-construction ground truth), contrasted against the one
# real_world fixture (period audio, officially-transcribed ground truth, not studio-clean).
_SYNTHETIC_FIXTURE_TYPES = frozenset({"single_speaker", "noisy", "multi_speaker"})
_REAL_WORLD_FIXTURE_TYPES = frozenset({"real_world"})
_CATEGORY_LABELS = {"synthetic": "Synthetic (TTS-generated)", "real_world": "Real-world (historical audio)"}
_CATEGORY_MARKERS = {"synthetic": "o", "real_world": "^"}
_CATEGORY_BAR_COLORS = {"synthetic": "#009E60", "real_world": "#6B6352"}


def _category_for_fixture_type(fixture_type: str) -> str | None:
    if fixture_type in _SYNTHETIC_FIXTURE_TYPES:
        return "synthetic"
    if fixture_type in _REAL_WORLD_FIXTURE_TYPES:
        return "real_world"
    return None


def _scoped_stats(fixtures: list[dict[str, Any]], category: str) -> dict[str, float | int | None]:
    """Recompute per-category means directly from the fixtures list -- report["summary"] blends
    all fixture types together, which is exactly what these charts need to NOT do.
    """
    scoped = [f for f in fixtures if _category_for_fixture_type(f.get("fixture_type", "")) == category]

    def _mean(key: str) -> float | None:
        values = [f[key] for f in scoped if f.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "fixture_count": len(scoped),
        "mean_wer": _mean("wer"),
        "mean_cost_usd_per_minute": _mean("cost_usd_per_minute"),
        "mean_latency_seconds": _mean("latency_seconds"),
        "mean_diarization_accuracy": _mean("diarization_accuracy"),
    }


def _fixture_count_caption(report: dict[str, Any]) -> str:
    any_fixtures = next(iter(report["providers"].values()))["fixtures"]
    synthetic_n = int(_scoped_stats(any_fixtures, "synthetic")["fixture_count"] or 0)
    real_world_n = int(_scoped_stats(any_fixtures, "real_world")["fixture_count"] or 0)
    real_world_noun = "fixture" if real_world_n == 1 else "fixtures"
    caveat = (
        "a single real-world clip is a data point, not a statistically robust average"
        if real_world_n <= 1
        else f"still a small sample ({real_world_n} clips), not a statistically robust average"
    )
    return (
        f"Synthetic: n={synthetic_n} fixtures (exact-by-construction ground truth). "
        f"Real-world: n={real_world_n} {real_world_noun} (officially-transcribed historical "
        f"audio) -- {caveat}."
    )


def plot_cost_vs_wer(report: dict[str, Any], out_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(9, 7))
    any_points = False

    # Labels near points that cluster tightly (e.g. the priciest couple of models) would
    # otherwise stack directly on top of each other -- stagger vertically by plot order as a
    # cheap approximation of collision avoidance, not true overlap detection.
    label_offsets = [(8, 6), (8, -14), (10, 16), (10, -24), (-70, 6), (-70, -14)]

    for idx, (provider_id, data) in enumerate(report["providers"].items()):
        color = _color_for(data.get("vendor", ""))
        model_label = data.get("model_id", provider_id)
        points: dict[str, tuple[float, float]] = {}
        for category in ("synthetic", "real_world"):
            stats = _scoped_stats(data["fixtures"], category)
            if stats["mean_cost_usd_per_minute"] is not None and stats["mean_wer"] is not None:
                points[category] = (stats["mean_cost_usd_per_minute"], stats["mean_wer"])

        if len(points) == 2:
            (sx, sy), (rx, ry) = points["synthetic"], points["real_world"]
            ax.plot([sx, rx], [sy, ry], color=color, alpha=0.3, linewidth=1, zorder=1)

        for category, (x, y) in points.items():
            any_points = True
            ax.scatter(
                x, y, s=140, color=color, edgecolor="#1A1A12", linewidth=0.6, marker=_CATEGORY_MARKERS[category], zorder=3
            )

        label_point = points.get("real_world") or points.get("synthetic")
        if label_point:
            xytext = label_offsets[idx % len(label_offsets)]
            ax.annotate(model_label, label_point, textcoords="offset points", xytext=xytext, fontsize=8)

    if not any_points:
        plt.close(fig)
        return None

    from matplotlib.lines import Line2D  # noqa: PLC0415

    legend_handles = [
        Line2D(
            [0], [0], marker=_CATEGORY_MARKERS[c], color="w", markerfacecolor="#6B6352",
            markeredgecolor="#1A1A12", markersize=10, label=_CATEGORY_LABELS[c],
        )
        for c in ("synthetic", "real_world")
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Cost (USD per minute of audio, log scale)")
    ax.set_ylabel("Word error rate (lower is better)")
    ax.set_title("Cost vs. accuracy: synthetic vs. real-world audio")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.text(0.01, 0.01, _fixture_count_caption(report), fontsize=7.5, color="#6B6352")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = out_dir / "cost_vs_wer.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _grouped_bar_chart(
    report: dict[str, Any], out_dir: Path, *, metric_key: str, filename: str, ylabel: str, title: str, ylim: tuple[float, float] | None
) -> Path | None:
    labels: list[str] = []
    synthetic_vals: list[float] = []
    real_world_vals: list[float] = []
    for provider_id, data in report["providers"].items():
        synth = _scoped_stats(data["fixtures"], "synthetic")[metric_key]
        real = _scoped_stats(data["fixtures"], "real_world")[metric_key]
        if synth is None and real is None:
            continue
        labels.append(data.get("model_id", provider_id))
        synthetic_vals.append(synth if synth is not None else 0.0)
        real_world_vals.append(real if real is not None else 0.0)

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], synthetic_vals, width=width, label=_CATEGORY_LABELS["synthetic"], color=_CATEGORY_BAR_COLORS["synthetic"])
    ax.bar([i + width / 2 for i in x], real_world_vals, width=width, label=_CATEGORY_LABELS["real_world"], color=_CATEGORY_BAR_COLORS["real_world"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.text(0.01, 0.01, _fixture_count_caption(report), fontsize=7.5, color="#6B6352")
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_wer_synthetic_vs_real(report: dict[str, Any], out_dir: Path) -> Path | None:
    return _grouped_bar_chart(
        report, out_dir,
        metric_key="mean_wer", filename="wer_by_fixture_type.png",
        ylabel="Word error rate", title="WER: synthetic vs. real-world audio", ylim=None,
    )


def plot_latency_bar(report: dict[str, Any], out_dir: Path) -> Path | None:
    return _grouped_bar_chart(
        report, out_dir,
        metric_key="mean_latency_seconds", filename="latency_by_provider.png",
        ylabel="Mean latency (seconds)", title="Latency: synthetic vs. real-world audio", ylim=None,
    )


def plot_diarization_accuracy(report: dict[str, Any], out_dir: Path) -> Path | None:
    return _grouped_bar_chart(
        report, out_dir,
        metric_key="mean_diarization_accuracy", filename="diarization_accuracy.png",
        ylabel="Speaker-label accuracy (best permutation)", title="Diarization accuracy: synthetic vs. real-world audio",
        ylim=(0, 1.0),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, help="Path to a JSON report from scripts/benchmark_multimodal_transcription.py.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory to write PNG charts into.")
    parser.add_argument(
        "--controls-table-only",
        action="store_true",
        help="Render only the static controls-comparison table (no --report needed).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = [plot_controls_table(args.out_dir)]
    if not args.controls_table_only:
        if args.report is None:
            print("error: --report is required unless --controls-table-only is set")
            return 1
        report = json.loads(args.report.read_text(encoding="utf-8"))
        for plot_fn in (plot_cost_vs_wer, plot_latency_bar, plot_wer_synthetic_vs_real, plot_diarization_accuracy):
            path = plot_fn(report, args.out_dir)
            if path is not None:
                written.append(path)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

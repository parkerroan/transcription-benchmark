# Transcription benchmark: dedicated ASR vs. multimodal chat models

Companion benchmark for a blog post comparing dedicated speech-to-text models against general
multimodal chat models on cost, word error rate (WER), latency, and speaker diarization
accuracy.

> This work was completed to provide data for evaluating transcription services for
> [Recastr](https://recastr.io).

**[Read the visual benchmark writeup →](https://parkerroan.github.io/transcription-benchmark/)**
Pipeline, model landscape, fixture set, scoring mechanism, results, and category winners, with
downloadable CSVs.

## What's compared

**Dedicated ASR** (built specifically for transcription):
- `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize` (OpenAI)
- `grok-stt` (xAI)
- `azure-fast-transcription` (Microsoft)

**General multimodal chat models** (given audio + a "transcribe this" prompt):
- `gpt-audio` (OpenAI)
- `gemini-3.7-flash` (Google)

Every model id was verified live against each vendor's current model list on 2026-08-13 rather
than assumed from training data — several originally-planned ids (`gpt-4o-audio-preview`,
`gemini-2.5-flash`) were already retired by the time this ran. See the module docstring in
`scripts/eval_providers.py` for the full trail, including which pricing figures are confirmed
against vendor pricing pages vs. third-party aggregators.

## Test fixtures

Two categories, on purpose:

1. **Synthetic** (`tests/integration/fixtures/multimodal_eval/*.mp3` except the two real-world
   clips) — generated via `scripts/generate_transcription_fixtures.py` using OpenAI's `tts-1`.
   Ground truth is exact by construction: clean narration, a numbers/proper-nouns stress test,
   technical jargon, a synthetic-white-noise-overlaid clip, and 2- and 3-speaker scripted
   dialogues.
2. **Real-world** — two officially-transcribed clips, chosen to separate "real-world audio is
   hard" from "old/degraded recordings are hard" by covering different eras and recording
   technology — one historical and public-domain, one modern and professionally recorded:
   - A ~2-minute excerpt from the Library of Congress American Folklife Center's June 11, 1949
     interview with Fountain Hughes, a formerly enslaved person, conducted by Hermond Norwood
     ([loc.gov/item/afc1950037_afs09990a](https://www.loc.gov/item/afc1950037_afs09990a/),
     call number AFC 1950/037: AFS 09990A).
   - A ~4.9-minute excerpt from NASA's "Houston We Have a Podcast," episode 414 ("Science in
     Space," aired March 2026), a modern two-speaker studio interview with an official
     transcript published by NASA
     ([nasa.gov/podcasts/houston-we-have-a-podcast/science-in-space](https://www.nasa.gov/podcasts/houston-we-have-a-podcast/science-in-space)).

   (A third real-world clip -- a 1962 White House Dictabelt recording -- was tried and dropped:
   the recording quality was poor enough that one speaker was barely audible even to a human
   listener, making it a poor source of ground truth regardless of any model's performance on
   it. Not every real-world source is usable just because a transcript exists for it.)

   Both are included because every model's WER rises well above the synthetic fixtures —
   roughly 1.6x to 4x depending on the model, averaged over 10 runs — even on the *modern,
   clean* NASA podcast, not just the degraded 1949 interview, confirming this isn't purely an
   "old audio" effect.

## Results (2026-08-13, averaged over 10 runs)

See `results/multimodal-transcription-2026-08-13/` for the full aggregated JSON report
and PNG charts. Every metric there is a mean across 10 independent full passes through every
fixture and model (80 calls per model total), computed by `scripts/aggregate_benchmark_runs.py`,
to smooth out per-call noise in LLM outputs and API latency — see "Running it" below for how to
reproduce this. The cost-vs-WER scatter groups both real-world clips into one category (vs.
synthetic); the three bar charts (latency, WER, diarization accuracy) instead show **each
real-world clip as its own bar** rather than averaging them together — with the two clips
spanning very different eras and recording quality, a blended "real-world" bar would hide
exactly the per-clip variation these charts exist to surface.

Headline findings:
- `gpt-audio` (multimodal) actually posts the **lowest** synthetic-audio WER of any model
  (0.051, vs. 0.072 for `gpt-4o-transcribe` and 0.085 for `whisper-1`) while costing roughly 6x
  more per minute than `gpt-4o-transcribe` — multimodal isn't just "competitive" here, it's the
  most accurate option, but that edge comes at a real cost premium.
- Every model's WER rises going from synthetic to real-world audio, but the size of that jump
  varies more than a single run suggested: from ~1.6x for `gpt-4o-transcribe-diarize` up to
  ~4x for `gpt-audio` and `gpt-4o-transcribe`. This holds even for the modern, cleanly-recorded
  NASA podcast clip, not just the degraded 1949 interview.
- Diarization accuracy on real-world audio doesn't cleanly separate "dedicated" from
  "multimodal" once averaged over 10 runs: `azure-fast-transcription` is the most consistent
  native diarizer (0.92 on both real-world clips), while `grok-stt` swings the widest (0.74 on
  the 1949 interview vs. 0.92 on the NASA podcast) despite a strong overall average. A single
  run had suggested `gpt-4o-transcribe-diarize` was consistently among the worst on real audio —
  that didn't hold up under 10-run averaging, where it lands mid-pack and fairly consistent
  (~0.85 on both clips).

Re-running will still produce somewhat different numbers (models update, pricing changes, LLM
outputs aren't deterministic) — treat the committed report/charts as a snapshot, not a promise.

## Running it

```bash
cp .env.example .env   # fill in at least OPENAI_API_KEY
uv sync

make generate-fixtures   # optional -- fixtures are already committed; re-run to regenerate
make benchmark            # calls every configured provider's API for real (real $ cost)
make plot                 # writes PNGs to /tmp/multimodal-transcription-charts
```

Requires `ffmpeg` on PATH (via `pydub`). Each provider is skipped automatically if its
credential isn't set in `.env` — see `.env.example` for the full list.

To reproduce the committed 10-run-averaged report/charts (real $ cost, ~10x a single
`make benchmark` run):

```bash
for i in $(seq 1 10); do
  uv run python scripts/benchmark_multimodal_transcription.py \
    --manifest tests/integration/fixtures/multimodal_eval/manifest.jsonl \
    --output /tmp/mt-run-$i.json
done
uv run python scripts/aggregate_benchmark_runs.py \
  --reports /tmp/mt-run-{1..10}.json --output /tmp/mt-aggregated.json
uv run python scripts/plot_multimodal_transcription_benchmark.py \
  --report /tmp/mt-aggregated.json --out-dir /tmp/mt-charts
```

## Development

```bash
make lint       # ruff
make typecheck  # pyright
make test       # offline unit tests only -- no API calls, no keys required
```

## Layout

```
scripts/
  eval_providers.py                     # one transcribe() interface across 8 models/5 vendors
  transcription_alignment.py            # text-order word alignment + diarization scorer
  generate_transcription_fixtures.py    # synthesizes the TTS-generated fixtures
  benchmark_multimodal_transcription.py # runs the eval, writes a JSON report
  aggregate_benchmark_runs.py           # averages multiple JSON reports into one (same schema)
  plot_multimodal_transcription_benchmark.py  # renders charts from a report
tests/
  test_*.py                             # offline unit tests (no network/API keys)
  integration/fixtures/multimodal_eval/ # audio + manifest.jsonl consumed by the benchmark
results/                                # committed report.json + charts from past runs
```

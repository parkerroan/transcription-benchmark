# Transcription benchmark: dedicated ASR vs. multimodal chat models

Companion benchmark for a blog post comparing dedicated speech-to-text models against general
multimodal chat models on cost, word error rate (WER), latency, and speaker diarization
accuracy.

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
2. **Real-world** — two officially-transcribed public-domain historical clips, chosen to cover
   different eras, recording technology, and speaker counts:
   - A ~2-minute excerpt from the Library of Congress American Folklife Center's June 11, 1949
     interview with Fountain Hughes, a formerly enslaved person, conducted by Hermond Norwood
     ([loc.gov/item/afc1950037_000160](http://www.loc.gov/item/afc1950037_000160)).
   - A ~4.5-minute excerpt (Item 41.2) from a White House Dictabelt recording of President
     Kennedy briefing former President Eisenhower on October 28, 1962, the morning Khrushchev
     agreed to withdraw missiles from Cuba
     ([jfklibrary.org/asset-viewer/archives/jfkpof-tph-41](https://www.jfklibrary.org/asset-viewer/archives/jfkpof-tph-41)).
     **Rights note:** the JFK Library states that statements by U.S. government officials in
     the course of their duties are public domain, but Eisenhower was a private citizen in
     October 1962 and the library lists this belt's copyright status as unknown — his half of
     the call isn't cleanly public domain the way Kennedy's is. Included anyway as a data point
     because it's the best-known public two-voice historical audio with a matched official
     transcript; reconsider before any commercial use beyond this research/benchmarking context.

   Both are included because every model's error rate roughly **doubles to triples** on them
   relative to the synthetic fixtures — clean TTS audio understates how hard real period audio
   actually is, and `gpt-4o-transcribe-diarize` in particular goes from competitive on synthetic
   audio to nearly unusable (WER ~0.98) on the degraded Dictabelt recording.

## Results (2026-08-13 run)

See `docs/benchmarks/multimodal-transcription-2026-08-13/` for the full JSON report and PNG
charts (cost vs. WER, latency, WER by synthetic/real-world, diarization accuracy, and a static
controls-comparison table). Headline findings:
- `gpt-audio` (multimodal) matches `gpt-4o-transcribe`'s (dedicated) WER almost exactly on
  synthetic audio while costing roughly 6x more per minute — multimodal buys flexibility here,
  not accuracy.
- Every model's WER roughly doubles-to-triples going from synthetic to real-world audio, and
  the ranking shuffles: `gpt-4o-transcribe-diarize`, strong on synthetic multi-speaker audio,
  is the worst performer by far on the degraded 1962 Dictabelt recording.

Re-running will produce different numbers (models update, pricing changes, LLM outputs aren't
deterministic) — treat the committed report/charts as a snapshot, not a promise.

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
  plot_multimodal_transcription_benchmark.py  # renders charts from a report
tests/
  test_*.py                             # offline unit tests (no network/API keys)
  integration/fixtures/multimodal_eval/ # audio + manifest.jsonl consumed by the benchmark
docs/benchmarks/                        # committed report.json + charts from past runs
```

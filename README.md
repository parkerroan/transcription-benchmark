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
2. **Real-world** — three officially-transcribed clips, chosen to separate "real-world audio is
   hard" from "old/degraded recordings are hard" by covering different eras, recording
   technology, and speaker counts — two historical and public-domain, one modern and
   professionally recorded:
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
   - A ~4.9-minute excerpt from NASA's "Houston We Have a Podcast," episode 414 ("Science in
     Space," aired March 2026), a modern two-speaker studio interview with an official
     transcript published by NASA
     ([nasa.gov/podcasts/houston-we-have-a-podcast/science-in-space](https://www.nasa.gov/podcasts/houston-we-have-a-podcast/science-in-space)).

   All three are included because every model's WER roughly **triples to septuples** (3x-7x)
   relative to the synthetic fixtures — even the *modern, clean* NASA podcast, not just the
   two degraded historical recordings, confirming this isn't purely an "old audio" effect.
   `gpt-4o-transcribe-diarize` in particular is the worst or near-worst performer on all three
   real-world clips despite being competitive on synthetic multi-speaker audio.

## Results (2026-08-13 run)

See `docs/benchmarks/multimodal-transcription-2026-08-13/` for the full JSON report and PNG
charts (cost vs. WER, latency, WER by synthetic/real-world, diarization accuracy, and a static
controls-comparison table). Headline findings:
- `gpt-audio` (multimodal) is competitive with `gpt-4o-transcribe`'s (dedicated) WER on
  synthetic audio while costing roughly 6x more per minute — multimodal buys flexibility here,
  not accuracy.
- Every model's WER roughly triples-to-septuples (3x-7x) going from synthetic to real-world
  audio, and this holds even for the modern, cleanly-recorded NASA podcast clip, not just the
  two degraded historical recordings. The ranking shuffles too: `gpt-4o-transcribe-diarize`,
  strong on synthetic multi-speaker audio, is consistently among the worst performers on real
  audio.

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

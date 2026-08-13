"""Generates synthetic transcription-eval fixtures using the OpenAI TTS API directly.

Real-world audio with a verified transcript can't be sourced from every environment this repo
runs in (some dev sandboxes have no general internet access), so ground truth here is instead
*exact by construction*: every fixture is synthesized from hand-authored text at natural pace
(tts-1, no speed adjustment), so the reference transcript is guaranteed correct. Multi-speaker
fixtures are built by synthesizing each turn with a different OpenAI voice and concatenating
with a short silence gap, with the true speaker label recorded per turn.

Per-word start times in the manifest are an even-spacing approximation over each turn's
measured audio duration, not measured ASR-grade timing -- fine for WER/keyterm scoring (text
only) and diarization accuracy (matched by text order, not time -- see
scripts/transcription_alignment.py), but timestamp-drift numbers computed against these
fixtures should be read as rough, not authoritative.

Requires OPENAI_API_KEY (incurs a small tts-1 synthesis cost -- a few cents for the whole set).
Writes audio (mp3) + a manifest.jsonl into tests/integration/fixtures/multimodal_eval/,
consumed by scripts/benchmark_multimodal_transcription.py.

Usage:
    uv run python scripts/generate_transcription_fixtures.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "integration" / "fixtures" / "multimodal_eval"

_NOISE_GAIN_DB = -18.0  # synthetic white noise mixed in well under narration level
_TURN_GAP_MS = 300


@dataclass(frozen=True)
class SingleSpeakerFixture:
    fixture_id: str
    fixture_type: str  # "single_speaker" | "noisy"
    voice: str
    text: str
    keyterms: tuple[str, ...]
    add_noise: bool = False


@dataclass(frozen=True)
class DialogueTurn:
    speaker: str  # reference label ("A"/"B"/"C"), independent of the TTS voice used
    voice: str
    text: str


@dataclass(frozen=True)
class MultiSpeakerFixture:
    fixture_id: str
    turns: tuple[DialogueTurn, ...]
    keyterms: tuple[str, ...]


SINGLE_SPEAKER_FIXTURES: tuple[SingleSpeakerFixture, ...] = (
    SingleSpeakerFixture(
        fixture_id="clean-narration",
        fixture_type="single_speaker",
        voice="alloy",
        text=(
            "Autumn arrived quietly this year. The leaves turned gold and red almost overnight, "
            "and the streets filled with the smell of woodsmoke. Children walked to school in "
            "thicker jackets, and shopkeepers swept fallen leaves from their doorsteps every "
            "morning before opening."
        ),
        keyterms=("autumn", "woodsmoke", "doorsteps"),
    ),
    SingleSpeakerFixture(
        fixture_id="numbers-and-names",
        fixture_type="single_speaker",
        voice="echo",
        text=(
            "Doctor Elena Vasquez will present the quarterly results on March 3rd at 2:45 PM in "
            "Conference Room 12B. Revenue increased by 18.6 percent to reach 4.2 million dollars, "
            "and the team expects to onboard 37 new clients by the end of Q3."
        ),
        keyterms=("Elena Vasquez", "March 3rd", "2:45", "18.6 percent", "4.2 million", "Conference Room 12B"),
    ),
    SingleSpeakerFixture(
        fixture_id="technical-jargon",
        fixture_type="single_speaker",
        voice="nova",
        text=(
            "The transformer architecture relies on multi-head self-attention to weigh "
            "relationships between tokens in a sequence. Each attention head projects the input "
            "into separate query, key, and value subspaces before the outputs are concatenated "
            "and passed through a feed-forward layer."
        ),
        keyterms=("transformer", "self-attention", "query", "key", "value", "feed-forward"),
    ),
    SingleSpeakerFixture(
        fixture_id="noisy-narration",
        fixture_type="noisy",
        voice="sage",
        text=(
            "The old lighthouse keeper climbed the spiral staircase every evening before sunset. "
            "He checked the lamp, wiped the glass clean, and waited for the first ship to appear "
            "on the horizon."
        ),
        keyterms=("lighthouse", "staircase", "horizon"),
        add_noise=True,
    ),
)

MULTI_SPEAKER_FIXTURES: tuple[MultiSpeakerFixture, ...] = (
    MultiSpeakerFixture(
        fixture_id="two-speaker-dialogue",
        turns=(
            DialogueTurn("A", "alloy", "Hey, did you get a chance to look at the budget numbers I sent over?"),
            DialogueTurn("B", "onyx", "Yeah, I did. I think we're overspending on the marketing line by about ten percent."),
            DialogueTurn("A", "alloy", "That matches what I found too. Should we bring it up in tomorrow's meeting?"),
            DialogueTurn("B", "onyx", "Definitely. Let's put together a quick summary tonight so we're ready."),
            DialogueTurn("A", "alloy", "Sounds good. I'll draft the slide and send it to you by eight."),
            DialogueTurn("B", "onyx", "Perfect, thanks. I'll review it first thing in the morning."),
        ),
        keyterms=("budget", "marketing", "ten percent", "slide"),
    ),
    MultiSpeakerFixture(
        fixture_id="three-speaker-dialogue",
        turns=(
            DialogueTurn("A", "alloy", "Okay, let's go around quickly. Jordan, how's the launch prep going?"),
            DialogueTurn("B", "onyx", "On track. QA finishes tomorrow, and we're still targeting Friday."),
            DialogueTurn("C", "nova", "I can confirm marketing assets are ready on our end too."),
            DialogueTurn("A", "alloy", "Great. Any blockers I should know about before Friday?"),
            DialogueTurn("B", "onyx", "Just one, we're waiting on final sign-off from legal."),
            DialogueTurn("C", "nova", "I'll follow up with them this afternoon and loop you both in."),
            DialogueTurn("A", "alloy", "Perfect, thanks both. Let's regroup tomorrow at the same time."),
        ),
        keyterms=("launch", "QA", "Friday", "legal", "sign-off"),
    ),
)


def _synthesize_tts(client: OpenAI, voice: str, text: str) -> bytes:
    """tts-1 at natural pace (no speed param) so the reference transcript's word count/text
    stays exactly what was requested -- fixture ground truth depends on this.
    """
    response = client.audio.speech.create(model="tts-1", voice=voice, input=text)
    return response.content


def _white_noise_segment(duration_ms: int, frame_rate: int, channels: int, gain_db: float) -> AudioSegment:
    rng = np.random.default_rng(seed=42)  # fixed seed: reproducible fixture audio across runs
    num_samples = int(frame_rate * duration_ms / 1000) * channels
    samples = (rng.standard_normal(num_samples) * 3000).astype(np.int16)
    noise = AudioSegment(samples.tobytes(), frame_rate=frame_rate, sample_width=2, channels=channels)
    return noise.apply_gain(gain_db)


def _evenly_spaced_words(words: list[str], duration_s: float, speaker: str | None) -> list[dict]:
    if not words:
        return []
    per_word = duration_s / len(words)
    return [{"text": word, "start": round(idx * per_word, 3), "speaker": speaker} for idx, word in enumerate(words)]


def _synthesize_single_speaker(client: OpenAI, spec: SingleSpeakerFixture) -> tuple[AudioSegment, list[dict]]:
    audio_bytes = _synthesize_tts(client, spec.voice, spec.text)
    segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    if spec.add_noise:
        noise = _white_noise_segment(len(segment), segment.frame_rate, segment.channels, _NOISE_GAIN_DB)
        segment = segment.overlay(noise)

    reference_words = _evenly_spaced_words(spec.text.split(), len(segment) / 1000.0, speaker=None)
    return segment, reference_words


def _synthesize_multi_speaker(client: OpenAI, spec: MultiSpeakerFixture) -> tuple[AudioSegment, list[dict]]:
    combined = AudioSegment.empty()
    gap = AudioSegment.silent(duration=_TURN_GAP_MS)
    reference_words: list[dict] = []

    for turn in spec.turns:
        audio_bytes = _synthesize_tts(client, turn.voice, turn.text)
        turn_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        offset_s = len(combined) / 1000.0
        turn_words = _evenly_spaced_words(turn.text.split(), len(turn_segment) / 1000.0, turn.speaker)
        for word in turn_words:
            word["start"] = round(word["start"] + offset_s, 3)
        reference_words.extend(turn_words)
        combined += turn_segment + gap

    return combined, reference_words


def main() -> int:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict] = []

    for spec in SINGLE_SPEAKER_FIXTURES:
        print(f"synthesizing {spec.fixture_id}...")
        segment, reference_words = _synthesize_single_speaker(client, spec)
        audio_filename = f"{spec.fixture_id}.mp3"
        segment.export(FIXTURES_DIR / audio_filename, format="mp3")
        manifest_entries.append({
            "audio_path": audio_filename,
            "reference_text": spec.text,
            "language": "en",
            "reference_words": reference_words,
            "keyterms": list(spec.keyterms),
            "fixture_type": spec.fixture_type,
        })

    for spec in MULTI_SPEAKER_FIXTURES:
        print(f"synthesizing {spec.fixture_id}...")
        segment, reference_words = _synthesize_multi_speaker(client, spec)
        audio_filename = f"{spec.fixture_id}.mp3"
        segment.export(FIXTURES_DIR / audio_filename, format="mp3")
        manifest_entries.append({
            "audio_path": audio_filename,
            "reference_text": " ".join(turn.text for turn in spec.turns),
            "language": "en",
            "reference_words": reference_words,
            "keyterms": list(spec.keyterms),
            "fixture_type": "multi_speaker",
        })

    manifest_path = FIXTURES_DIR / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in manifest_entries:
            handle.write(json.dumps(entry) + "\n")

    print(f"wrote {len(manifest_entries)} fixtures + manifest to {FIXTURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

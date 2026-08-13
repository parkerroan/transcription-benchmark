"""Provider wrappers for scripts/benchmark_multimodal_transcription.py.

Wraps eight models across five vendors behind one `EvalProvider.transcribe()` shape so the
benchmark script can treat dedicated ASR and general multimodal chat models uniformly. Every
model id here was confirmed live (not from training-data memory) against each vendor's
current model list and pricing page on 2026-08-13 -- the model landscape moves fast enough
that several originally-planned ids were already dead or superseded by the time this ran:
  - whisper-1                 (OpenAI, dedicated /v1/audio/transcriptions)
  - gpt-4o-transcribe          (OpenAI, dedicated, no timestamps -- still current, not legacy)
  - gpt-4o-mini-transcribe     (OpenAI, dedicated, no timestamps -- still current, not legacy)
  - gpt-4o-transcribe-diarize  (OpenAI, dedicated, NATIVE diarization + segment-level timestamps,
                                 same per-token rate as gpt-4o-transcribe)
  - gpt-audio                  (OpenAI, chat completions, true multimodal -- replaces the now-dead
                                 "gpt-4o-audio-preview"; "gpt-realtime-2.1" has identical $32/$64
                                 pricing but requires the WebSocket Realtime API, not chat
                                 completions, so it's out of scope for this batch-style script)
  - gemini-3.7-flash           (Google, generateContent, true multimodal -- "gemini-2.5-flash" now
                                 404s "no longer available to new users"; 3.7 is the current
                                 latest stable flash-tier model, confirmed via live model list)
  - grok-stt                   (xAI, dedicated /v1/stt, native diarization)
  - azure-fast-transcription   (Azure AI Speech "Fast Transcription" REST API, dedicated, NATIVE
                                 word timestamps + native diarization up to 35 speakers. Its docs
                                 only show a custom-subdomain endpoint
                                 (https://{resource}.cognitiveservices.azure.com/...), but the
                                 older region-based endpoint (https://{region}.api.cognitive.
                                 microsoft.com/speechtotext/transcriptions:transcribe) was
                                 confirmed working live against AZURE_SPEECH_KEY/AZURE_SPEECH_REGION
                                 on 2026-08-13.

Multimodal chat models (gpt-audio, Gemini) return no structured timestamps or
speaker field at all -- when `diarize=True` they're instead prompted to label speaker turns
inline ("Speaker 1: ...") and that's parsed back into per-word speaker tags with no timing,
which is why EvalWord.start/end are optional. That gap (or lack of one) is itself a "controls"
finding for the comparison, not a bug to paper over.

PRICING NOTE: the USD_PER_* constants below were confirmed live against developers.openai.com/
api/docs/pricing and ai.google.dev/gemini-api/docs/pricing on 2026-08-13, except
XAI_STT_USD_PER_HOUR_BATCH and AZURE_STT_USD_PER_HOUR, both still from third-party aggregators
(x.ai's pricing page returned 403; Azure's public pricing page renders its dollar figures via
JS and came back blank on fetch, both azure.microsoft.com/.../speech-services/ and the Azure
pricing calculator) -- verify those two specifically before publishing. gemini-3.7-flash's
listed rate is a promotional price good only through 2026-12-31 (doubles 2027-01-01) -- re-check
before publishing if this runs after that date. Where an API response includes real usage/token
counts, cost is computed from that instead of these constants regardless (cost_basis field on
EvalTranscript tells you which happened); Azure and xAI bill by audio duration, not tokens, so
their cost is always the flat-rate constant.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

WHISPER_USD_PER_MINUTE = 0.006

GPT4O_MINI_TRANSCRIBE_USD_PER_1M_INPUT_AUDIO_TOKENS = 1.25
GPT4O_MINI_TRANSCRIBE_USD_PER_1M_OUTPUT_TOKENS = 5.00
GPT4O_MINI_TRANSCRIBE_USD_PER_MINUTE_FALLBACK = 0.003

GPT4O_TRANSCRIBE_USD_PER_1M_INPUT_AUDIO_TOKENS = 2.50
GPT4O_TRANSCRIBE_USD_PER_1M_OUTPUT_TOKENS = 10.00
GPT4O_TRANSCRIBE_USD_PER_MINUTE_FALLBACK = 0.006

# Pinned to a dated snapshot rather than the undated "gpt-audio" alias: the model this used to
# point at ("gpt-4o-audio-preview") was retired entirely, so a floating alias isn't safe to
# trust to keep meaning the same thing either -- a dated id at least fails loudly (404) instead
# of silently becoming a different model. Confirmed live against developers.openai.com/api/docs/pricing
# on 2026-08-13.
OPENAI_AUDIO_CHAT_MODEL = "gpt-audio-2025-08-28"
OPENAI_AUDIO_CHAT_USD_PER_1M_INPUT_AUDIO_TOKENS = 32.00
OPENAI_AUDIO_CHAT_USD_PER_1M_OUTPUT_TEXT_TOKENS = 10.00
# Only used when the API response doesn't break out audio_tokens separately (see
# _cost_from_openai_chat_audio_usage) -- OpenAI's own realtime-API docs cite ~50 audio
# tokens/sec as a rule of thumb at 24kHz.
OPENAI_AUDIO_CHAT_FALLBACK_AUDIO_TOKENS_PER_SECOND = 50

# "gemini-2.5-flash" returns 404 "no longer available to new users" as of 2026-08-13.
# gemini-3.7-flash is the current latest stable flash-tier model (confirmed against the live
# model list -- 3.5-flash and 3.6-flash both still work but are superseded). Unlike 2.5-flash,
# it bills all input modalities (text/audio/image/video) at one unified rate rather than a
# separate elevated audio rate, and bundles "thinking" tokens into the output rate (see
# _cost_from_gemini_usage). Rate below is the 2026-08-13 promotional price -- Google's own
# pricing page says it roughly doubles on 2027-01-01, so re-check before publishing after that.
GEMINI_MODEL = "gemini-3.7-flash"
GEMINI_USD_PER_1M_INPUT_AUDIO_TOKENS = 0.75
GEMINI_USD_PER_1M_OUTPUT_TEXT_TOKENS = 3.75
# Only used as a cost fallback when usage_metadata is missing from the response.
GEMINI_FALLBACK_AUDIO_TOKENS_PER_SECOND = 30

# docs.x.ai's own pricing table renders its dollar figures via JS (came back blank on fetch),
# and x.ai/api 403s. $0.10/hr batch is corroborated by four independent third-party sources as
# of 2026-08-13 and contradicted by none, but it's the one number in this file not confirmed
# directly against a vendor page -- worth a manual check before publishing.
XAI_STT_USD_PER_HOUR_BATCH = 0.10

# azure.microsoft.com's own pricing page also renders its dollar figures via JS (blank on
# fetch). $0.36/hr specifically for "fast transcription" (the endpoint this provider actually
# calls) is corroborated by one third-party aggregator as of 2026-08-13 -- the other Azure STT
# tiers found (batch $0.18/hr, real-time standard $1/hr) are NOT what this provider uses, don't
# substitute them. Diarization is assumed included at no extra cost on this endpoint (real-time
# standard transcription adds +$0.30/hr for it, but fast transcription's docs don't mention a
# diarization surcharge) -- worth a manual check before publishing.
AZURE_STT_USD_PER_HOUR = 0.36

_SPEAKER_LINE_RE = re.compile(r"^\s*Speaker\s+(\w+)\s*:\s*(.*)$", re.IGNORECASE)
_XAI_ENDPOINT = "https://api.x.ai/v1/stt"


@dataclass(frozen=True)
class EvalWord:
    text: str
    start: float | None
    end: float | None
    speaker: str | None


@dataclass(frozen=True)
class EvalTranscript:
    text: str
    words: tuple[EvalWord, ...]
    supports_word_timestamps: bool
    supports_diarization: bool
    cost_usd: float
    cost_basis: str  # "usage_tokens" | "flat_rate" | "duration_estimate"
    duration_seconds: float


class EvalProvider(Protocol):
    provider_id: str
    model_id: str
    # Static per-model capabilities (independent of any one call's `diarize` argument) --
    # used to label the report, not derived from run results.
    supports_word_timestamps: bool
    can_attempt_diarization: bool

    def transcribe(self, path: str, language: str | None, *, diarize: bool) -> EvalTranscript: ...


def probe_duration_seconds(path: str) -> float:
    from pydub import AudioSegment  # noqa: PLC0415

    return len(AudioSegment.from_file(path)) / 1000.0


def _parse_speaker_labeled_text(raw_text: str) -> tuple[str, tuple[EvalWord, ...]]:
    """Parse a "Speaker 1: ...\\nSpeaker 2: ..." transcript into per-word speaker tags.

    Best-effort: a multimodal model that doesn't follow the requested "Speaker N:" format
    exactly just yields fewer (or zero) speaker-tagged words rather than raising -- that
    degradation is itself worth reporting, not something to mask with a stricter parser.
    """
    words: list[EvalWord] = []
    plain_lines: list[str] = []
    for line in raw_text.splitlines():
        match = _SPEAKER_LINE_RE.match(line)
        if not match:
            if line.strip():
                plain_lines.append(line.strip())
            continue
        speaker_id, utterance = match.group(1), match.group(2)
        plain_lines.append(utterance)
        words.extend(EvalWord(token, None, None, speaker_id) for token in utterance.split())
    return " ".join(plain_lines), tuple(words)


def _audio_format_for_openai_chat(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix not in {"wav", "mp3"}:
        raise ValueError(f"gpt-audio only accepts wav/mp3 input, got {suffix!r}")
    return suffix


def _mime_type_for_gemini(path: str) -> str:
    mime_by_suffix = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4", "flac": "audio/flac"}
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix not in mime_by_suffix:
        raise ValueError(f"unsupported audio suffix {suffix!r} for Gemini")
    return mime_by_suffix[suffix]


def _cost_from_openai_transcribe_usage(model_id: str, usage: object, *, fallback_duration_seconds: float) -> tuple[float, str]:
    if model_id == "gpt-4o-mini-transcribe":
        input_rate = GPT4O_MINI_TRANSCRIBE_USD_PER_1M_INPUT_AUDIO_TOKENS
        output_rate = GPT4O_MINI_TRANSCRIBE_USD_PER_1M_OUTPUT_TOKENS
        per_minute_fallback = GPT4O_MINI_TRANSCRIBE_USD_PER_MINUTE_FALLBACK
    elif model_id in ("gpt-4o-transcribe", "gpt-4o-transcribe-diarize"):
        # gpt-4o-transcribe-diarize is billed at the same per-token rate as gpt-4o-transcribe --
        # diarization doesn't cost extra (confirmed developers.openai.com/api/docs/pricing 2026-08-13).
        input_rate = GPT4O_TRANSCRIBE_USD_PER_1M_INPUT_AUDIO_TOKENS
        output_rate = GPT4O_TRANSCRIBE_USD_PER_1M_OUTPUT_TOKENS
        per_minute_fallback = GPT4O_TRANSCRIBE_USD_PER_MINUTE_FALLBACK
    else:
        raise ValueError(f"unknown model_id {model_id!r}")

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is not None and output_tokens is not None:
        cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
        return cost, "usage_tokens"

    return (fallback_duration_seconds / 60.0) * per_minute_fallback, "duration_estimate"


def _cost_from_openai_chat_audio_usage(usage: object, *, fallback_duration_seconds: float) -> tuple[float, str]:
    prompt_details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
    audio_tokens = getattr(prompt_details, "audio_tokens", None) if prompt_details is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    if audio_tokens is not None and completion_tokens is not None:
        cost = (
            (audio_tokens / 1_000_000) * OPENAI_AUDIO_CHAT_USD_PER_1M_INPUT_AUDIO_TOKENS
            + (completion_tokens / 1_000_000) * OPENAI_AUDIO_CHAT_USD_PER_1M_OUTPUT_TEXT_TOKENS
        )
        return cost, "usage_tokens"

    estimated_audio_tokens = fallback_duration_seconds * OPENAI_AUDIO_CHAT_FALLBACK_AUDIO_TOKENS_PER_SECOND
    cost = (estimated_audio_tokens / 1_000_000) * OPENAI_AUDIO_CHAT_USD_PER_1M_INPUT_AUDIO_TOKENS
    return cost, "duration_estimate"


def _cost_from_gemini_usage(usage_metadata: object, *, fallback_duration_seconds: float) -> tuple[float, str]:
    prompt_tokens = getattr(usage_metadata, "prompt_token_count", None) if usage_metadata is not None else None
    output_tokens = getattr(usage_metadata, "candidates_token_count", None) if usage_metadata is not None else None
    if prompt_tokens is not None and output_tokens is not None:
        # Applies the elevated audio-input rate to the whole prompt (audio + short text
        # instruction) rather than splitting the few instruction tokens out at the cheaper
        # text rate -- a small, intentional overestimate.
        cost = (
            (prompt_tokens / 1_000_000) * GEMINI_USD_PER_1M_INPUT_AUDIO_TOKENS
            + (output_tokens / 1_000_000) * GEMINI_USD_PER_1M_OUTPUT_TEXT_TOKENS
        )
        return cost, "usage_tokens"

    estimated_audio_tokens = fallback_duration_seconds * GEMINI_FALLBACK_AUDIO_TOKENS_PER_SECOND
    cost = (estimated_audio_tokens / 1_000_000) * GEMINI_USD_PER_1M_INPUT_AUDIO_TOKENS
    return cost, "duration_estimate"


_PLAIN_TRANSCRIBE_INSTRUCTION = "Transcribe this audio verbatim. Output only the transcript text, no commentary."
_DIARIZE_INSTRUCTION = (
    "Transcribe this audio verbatim. Label every speaker turn on its own line as "
    "'Speaker 1:', 'Speaker 2:', etc., based on voice changes. Start a new labeled line every "
    "time the speaker changes, even mid-sentence. Output only the labeled transcript, no "
    "commentary."
)


class WhisperEvalProvider:
    """whisper-1 via /v1/audio/transcriptions -- native word + segment timestamps."""

    provider_id = "openai"
    model_id = "whisper-1"
    supports_word_timestamps = True
    can_attempt_diarization = False

    def __init__(self, openai_client) -> None:
        self._client = openai_client

    def transcribe(self, path: str, language: str | None, *, diarize: bool = False) -> EvalTranscript:
        with open(path, "rb") as audio:
            kwargs = {
                "model": self.model_id,
                "file": audio,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word", "segment"],
            }
            if language:
                kwargs["language"] = language
            result = self._client.audio.transcriptions.create(**kwargs)

        words = tuple(EvalWord(w.word.strip(), w.start, w.end, None) for w in (result.words or []))
        segment_ends = [segment.end for segment in (result.segments or [])]
        duration_seconds = max((w.end for w in words if w.end is not None), default=max(segment_ends, default=0.0))
        cost_usd = (duration_seconds / 60.0) * WHISPER_USD_PER_MINUTE
        return EvalTranscript(
            text=result.text or "",
            words=words,
            supports_word_timestamps=True,
            supports_diarization=False,
            cost_usd=cost_usd,
            cost_basis="flat_rate",
            duration_seconds=duration_seconds,
        )


class OpenAiDedicatedTranscribeEvalProvider:
    """gpt-4o-transcribe / gpt-4o-mini-transcribe -- dedicated endpoint, no timestamps."""

    provider_id = "openai"
    supports_word_timestamps = False
    can_attempt_diarization = False

    def __init__(self, openai_client, model_id: str) -> None:
        self._client = openai_client
        self.model_id = model_id

    def transcribe(self, path: str, language: str | None, *, diarize: bool = False) -> EvalTranscript:
        with open(path, "rb") as audio:
            kwargs = {"model": self.model_id, "file": audio, "response_format": "json"}
            if language:
                kwargs["language"] = language
            result = self._client.audio.transcriptions.create(**kwargs)

        duration_seconds = probe_duration_seconds(path)
        cost_usd, cost_basis = _cost_from_openai_transcribe_usage(
            self.model_id, getattr(result, "usage", None), fallback_duration_seconds=duration_seconds
        )
        return EvalTranscript(
            text=result.text or "",
            words=(),
            supports_word_timestamps=False,
            supports_diarization=False,
            cost_usd=cost_usd,
            cost_basis=cost_basis,
            duration_seconds=duration_seconds,
        )


def _words_from_diarized_segments(segments) -> tuple[EvalWord, ...]:
    """Evenly split each segment's duration across its words -- same approximation used for
    the multi-speaker fixtures themselves (see scripts/generate_transcription_fixtures.py).
    Each segment needs only .text/.start/.end/.speaker attributes.
    """
    words: list[EvalWord] = []
    for segment in segments:
        tokens = segment.text.split()
        if not tokens:
            continue
        per_word = (segment.end - segment.start) / len(tokens)
        words.extend(
            EvalWord(
                token,
                round(segment.start + i * per_word, 3),
                round(segment.start + (i + 1) * per_word, 3),
                segment.speaker,
            )
            for i, token in enumerate(tokens)
        )
    return tuple(words)


class OpenAiDiarizeTranscribeEvalProvider:
    """gpt-4o-transcribe-diarize -- dedicated endpoint, native diarization with segment-level
    (not word-level) timestamps. Billed at the same rate as gpt-4o-transcribe (see
    _cost_from_openai_transcribe_usage) -- diarization is not an extra cost.

    Word-level entries are synthesized by evenly splitting each returned segment's duration
    across its words, same approximation used for the multi-speaker fixtures themselves (see
    scripts/generate_transcription_fixtures.py) -- so timestamp-drift numbers computed against
    this provider are a rough signal, not the model's actual per-word precision.
    """

    provider_id = "openai"
    model_id = "gpt-4o-transcribe-diarize"
    supports_word_timestamps = False
    can_attempt_diarization = True

    def __init__(self, openai_client) -> None:
        self._client = openai_client

    def transcribe(self, path: str, language: str | None, *, diarize: bool = False) -> EvalTranscript:
        with open(path, "rb") as audio:
            # chunking_strategy is required for audio long enough to need internal chunking --
            # confirmed live: short synthetic clips (~30s) work without it, but a ~2min
            # real-world fixture 400s with "chunking_strategy is required for diarization
            # models" unless it's set. "auto" works for both lengths, so just always pass it.
            kwargs = {
                "model": self.model_id,
                "file": audio,
                "response_format": "diarized_json",
                "chunking_strategy": "auto",
            }
            if language:
                kwargs["language"] = language
            result = self._client.audio.transcriptions.create(**kwargs)

        words = _words_from_diarized_segments(result.segments)
        duration_seconds = result.duration
        cost_usd, cost_basis = _cost_from_openai_transcribe_usage(
            self.model_id, getattr(result, "usage", None), fallback_duration_seconds=duration_seconds
        )
        return EvalTranscript(
            text=result.text or "",
            words=words,
            supports_word_timestamps=False,
            supports_diarization=True,
            cost_usd=cost_usd,
            cost_basis=cost_basis,
            duration_seconds=duration_seconds,
        )


class OpenAiAudioChatEvalProvider:
    """gpt-audio via chat completions -- true multimodal, no structured timestamps."""

    provider_id = "openai"
    model_id = OPENAI_AUDIO_CHAT_MODEL
    supports_word_timestamps = False
    can_attempt_diarization = True  # promptable ("Speaker 1:" labels), not a structured field

    def __init__(self, openai_client) -> None:
        self._client = openai_client

    def transcribe(self, path: str, language: str | None, *, diarize: bool = False) -> EvalTranscript:
        audio_format = _audio_format_for_openai_chat(path)
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")

        instruction = _DIARIZE_INSTRUCTION if diarize else _PLAIN_TRANSCRIBE_INSTRUCTION
        response = self._client.chat.completions.create(
            model=self.model_id,
            modalities=["text"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}},
                ],
            }],
        )
        raw_text = response.choices[0].message.content or ""
        text, words = _parse_speaker_labeled_text(raw_text) if diarize else (raw_text, ())

        duration_seconds = probe_duration_seconds(path)
        cost_usd, cost_basis = _cost_from_openai_chat_audio_usage(
            getattr(response, "usage", None), fallback_duration_seconds=duration_seconds
        )
        return EvalTranscript(
            text=text,
            words=words,
            supports_word_timestamps=False,
            supports_diarization=diarize,
            cost_usd=cost_usd,
            cost_basis=cost_basis,
            duration_seconds=duration_seconds,
        )


class GeminiEvalProvider:
    """Gemini via generateContent -- true multimodal, no structured timestamps."""

    provider_id = "google"
    model_id = GEMINI_MODEL
    supports_word_timestamps = False
    can_attempt_diarization = True  # promptable ("Speaker 1:" labels), not a structured field

    def __init__(self, api_key: str) -> None:
        from google import genai  # noqa: PLC0415

        self._client = genai.Client(api_key=api_key)

    def transcribe(self, path: str, language: str | None, *, diarize: bool = False) -> EvalTranscript:
        from google.genai import types  # noqa: PLC0415

        with open(path, "rb") as f:
            audio_bytes = f.read()
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=_mime_type_for_gemini(path))

        instruction = _DIARIZE_INSTRUCTION if diarize else _PLAIN_TRANSCRIBE_INSTRUCTION
        response = self._client.models.generate_content(model=self.model_id, contents=[instruction, audio_part])
        raw_text = response.text or ""
        text, words = _parse_speaker_labeled_text(raw_text) if diarize else (raw_text, ())

        duration_seconds = probe_duration_seconds(path)
        cost_usd, cost_basis = _cost_from_gemini_usage(
            getattr(response, "usage_metadata", None), fallback_duration_seconds=duration_seconds
        )
        return EvalTranscript(
            text=text,
            words=words,
            supports_word_timestamps=False,
            supports_diarization=diarize,
            cost_usd=cost_usd,
            cost_basis=cost_basis,
            duration_seconds=duration_seconds,
        )


class XaiGrokEvalProvider:
    """grok-stt via /v1/stt -- dedicated endpoint, native word timestamps + diarization."""

    provider_id = "xai"
    model_id = "grok-stt"
    supports_word_timestamps = True
    can_attempt_diarization = True

    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        self._client = client if client is not None else httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=120.0
        )

    def transcribe(self, path: str, language: str | None, *, diarize: bool = False) -> EvalTranscript:
        import mimetypes  # noqa: PLC0415

        data = {"diarize": "true", "filler_words": "true"}
        if language:
            data.update({"language": language, "format": "true"})

        filename = Path(path).name
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(path, "rb") as audio:
            response = self._client.post(_XAI_ENDPOINT, data=data, files={"file": (filename, audio, content_type)})
            response.raise_for_status()
        payload = response.json()

        words = tuple(
            EvalWord(w["text"], w["start"], w["end"], str(w["speaker"]) if w.get("speaker") is not None else None)
            for w in payload.get("words", [])
        )
        duration_seconds = float(payload["duration"])
        cost_usd = (duration_seconds / 3600.0) * XAI_STT_USD_PER_HOUR_BATCH
        return EvalTranscript(
            text=payload["text"],
            words=words,
            supports_word_timestamps=True,
            supports_diarization=True,
            cost_usd=cost_usd,
            cost_basis="flat_rate",
            duration_seconds=duration_seconds,
        )


def _words_from_azure_phrases(phrases: list, *, diarize: bool) -> tuple[EvalWord, ...]:
    """Flatten Azure Fast Transcription's phrases[].words[] into EvalWords.

    Each phrase carries an optional integer `speaker` (only present when diarization was
    requested); its per-word offsets are milliseconds from the start of the audio. Expects
    plain dicts (the parsed JSON response), not objects, unlike the OpenAI/Gemini helpers.
    """
    words: list[EvalWord] = []
    for phrase in phrases:
        speaker = str(phrase["speaker"]) if diarize and "speaker" in phrase else None
        for word in phrase.get("words", []):
            start = word["offsetMilliseconds"] / 1000.0
            end = start + word["durationMilliseconds"] / 1000.0
            words.append(EvalWord(word["text"], start, end, speaker))
    return tuple(words)


_AZURE_DEFAULT_LOCALE_FOR_LANGUAGE = {"en": "en-US"}  # extend if fixtures ever add other languages


def _normalize_azure_locale(language: str) -> str:
    """Azure's `locales` field requires a full BCP-47 tag ("en-US"), not a bare ISO-639-1 code
    ("en") -- every other provider in this file accepts the bare code fine (confirmed live: a
    bare "en" 400s here with "InvalidLocale" but works everywhere else), so the manifest just
    uses that, and this maps it for the one vendor that's stricter.
    """
    if "-" in language:
        return language
    return _AZURE_DEFAULT_LOCALE_FOR_LANGUAGE.get(language, language)


class AzureSttEvalProvider:
    """Azure AI Speech Fast Transcription REST API -- dedicated endpoint, native word
    timestamps + native diarization (up to 35 speakers, though maxSpeakers is capped lower
    here to match this eval's fixtures). Billed per audio hour (flat rate) like xAI, not by
    token usage.
    """

    provider_id = "azure"
    model_id = "azure-fast-transcription"
    supports_word_timestamps = True
    can_attempt_diarization = True

    _ENDPOINT_TEMPLATE = "https://{region}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15"
    _MAX_SPEAKERS = 6  # headroom above this eval's largest fixture (3 speakers)

    def __init__(self, api_key: str, region: str, client: httpx.Client | None = None) -> None:
        self._url = self._ENDPOINT_TEMPLATE.format(region=region)
        self._headers = {"Ocp-Apim-Subscription-Key": api_key}
        self._client = client if client is not None else httpx.Client(timeout=120.0)

    def transcribe(self, path: str, language: str | None, *, diarize: bool = False) -> EvalTranscript:
        import mimetypes  # noqa: PLC0415

        definition: dict[str, object] = {}
        if language:
            definition["locales"] = [_normalize_azure_locale(language)]
        if diarize:
            definition["diarization"] = {"enabled": True, "maxSpeakers": self._MAX_SPEAKERS}

        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as audio:
            files = {
                "audio": (Path(path).name, audio, content_type),
                "definition": (None, json.dumps(definition), "application/json"),
            }
            response = self._client.post(self._url, headers=self._headers, files=files)
            response.raise_for_status()

        payload = response.json()
        words = _words_from_azure_phrases(payload.get("phrases", []), diarize=diarize)
        duration_seconds = payload["durationMilliseconds"] / 1000.0
        text = " ".join(combined["text"] for combined in payload.get("combinedPhrases", []))
        cost_usd = (duration_seconds / 3600.0) * AZURE_STT_USD_PER_HOUR
        return EvalTranscript(
            text=text,
            words=words,
            supports_word_timestamps=True,
            supports_diarization=diarize,
            cost_usd=cost_usd,
            cost_basis="flat_rate",
            duration_seconds=duration_seconds,
        )


def available_eval_providers(
    *,
    openai_api_key: str,
    xai_api_key: str,
    gemini_api_key: str,
    azure_api_key: str = "",
    azure_region: str = "",
) -> dict[str, EvalProvider]:
    """Construct one provider per credential currently configured, keyed by model id."""
    providers: dict[str, EvalProvider] = {}
    if openai_api_key:
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(api_key=openai_api_key)
        providers["whisper-1"] = WhisperEvalProvider(client)
        providers["gpt-4o-transcribe"] = OpenAiDedicatedTranscribeEvalProvider(client, "gpt-4o-transcribe")
        providers["gpt-4o-mini-transcribe"] = OpenAiDedicatedTranscribeEvalProvider(client, "gpt-4o-mini-transcribe")
        providers["gpt-4o-transcribe-diarize"] = OpenAiDiarizeTranscribeEvalProvider(client)
        providers["gpt-audio"] = OpenAiAudioChatEvalProvider(client)
    if xai_api_key:
        providers["grok-stt"] = XaiGrokEvalProvider(xai_api_key)
    if gemini_api_key:
        providers["gemini-3.7-flash"] = GeminiEvalProvider(gemini_api_key)
    if azure_api_key and azure_region:
        providers["azure-fast-transcription"] = AzureSttEvalProvider(azure_api_key, azure_region)
    return providers

"""Pure alignment/scoring helpers for scripts/benchmark_multimodal_transcription.py.

Deliberately free of any provider SDK import so it can be unit tested
(tests/test_transcription_alignment.py) without network access or API keys.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

_TOKEN_RE = re.compile(r"[^\w']", re.UNICODE)


def normalize_token(text: str) -> str:
    return _TOKEN_RE.sub("", text).lower()


class _Word(Protocol):
    @property
    def text(self) -> str: ...


@dataclass(frozen=True)
class AlignedPair:
    reference_speaker: str | None
    provider_speaker: str | None
    reference_start: float | None
    provider_start: float | None


def align_words(reference_words: Sequence[_Word], provider_words: Sequence[_Word]) -> list[AlignedPair]:
    """Greedy in-order text match between reference and provider words.

    Matches on text alone. Speaker and start-time fields are read via getattr with a None
    default so this works whether or not either side carries them: diarization-by-prompt
    providers (gpt-4o-audio-preview, Gemini) return a speaker label but no timestamp at all,
    while dedicated-transcription-model fixtures may have timestamps but no speaker. Callers
    pull out whichever of (reference_start, provider_start) / (reference_speaker,
    provider_speaker) they need -- drift scoring and diarization scoring share this one pass.

    Known limitation shared with scripts/benchmark_transcription.py's match_word_drift: this
    is a greedy first-match, not a real alignment/DTW algorithm, so repeated words or provider
    mis-transcriptions can produce skipped or mismatched pairs. Fine as a directional signal,
    not a precise metric.
    """
    normalized_provider = [(normalize_token(w.text), w) for w in provider_words]
    pairs: list[AlignedPair] = []
    cursor = 0
    for reference in reference_words:
        target = normalize_token(reference.text)
        if not target:
            continue
        for idx in range(cursor, len(normalized_provider)):
            token, provider_word = normalized_provider[idx]
            if token == target:
                pairs.append(AlignedPair(
                    reference_speaker=getattr(reference, "speaker", None),
                    provider_speaker=getattr(provider_word, "speaker", None),
                    reference_start=getattr(reference, "start", None),
                    provider_start=getattr(provider_word, "start", None),
                ))
                cursor = idx + 1
                break
    return pairs


def diarization_accuracy(pairs: list[AlignedPair]) -> tuple[float, dict[str, str]] | None:
    """Best-case speaker-label accuracy under the optimal reference<->provider label mapping.

    Provider speaker IDs are arbitrary (xAI returns 0/1/2 ints, prompted models return
    "Speaker 1"/"Speaker 2" strings) and have no reason to line up with our reference A/B/C
    labels, so this brute-forces the permutation that maximizes matches. Fine at the 2-3
    speaker counts these fixtures use; would need the Hungarian algorithm at larger counts.

    This is a simplified word-count accuracy, not a true (time-weighted) Diarization Error
    Rate -- some providers here return no timestamps at all, so DER isn't computable for
    all of them on equal footing.

    Returns None if no word was labeled with a speaker on both sides (nothing to score).
    """
    scored = [p for p in pairs if p.reference_speaker is not None and p.provider_speaker is not None]
    if not scored:
        return None

    reference_labels = sorted({p.reference_speaker for p in scored if p.reference_speaker is not None})
    provider_labels = sorted({p.provider_speaker for p in scored if p.provider_speaker is not None})

    if len(provider_labels) <= len(reference_labels):
        smaller, larger, provider_to_reference = provider_labels, reference_labels, True
    else:
        smaller, larger, provider_to_reference = reference_labels, provider_labels, False

    best_accuracy = 0.0
    best_mapping: dict[str, str] = {}
    for candidate in itertools.permutations(larger, len(smaller)):
        mapping = dict(zip(smaller, candidate, strict=True))
        correct = 0
        for pair in scored:
            assert pair.reference_speaker is not None and pair.provider_speaker is not None
            if provider_to_reference:
                correct += mapping.get(pair.provider_speaker) == pair.reference_speaker
            else:
                correct += mapping.get(pair.reference_speaker) == pair.provider_speaker
        accuracy = correct / len(scored)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_mapping = dict(mapping)

    return best_accuracy, best_mapping

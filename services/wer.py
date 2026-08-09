"""Word error rate: the standard ASR accuracy metric.

    WER = (substitutions + deletions + insertions) / reference_word_count

Computed by Levenshtein alignment over *word* sequences, which is what makes the
component counts meaningful -- a caller who sees WER 0.27 usually wants to know
whether it was three substitutions or seven deletions, because those point at very
different problems.

Note that WER is **unbounded above**. A hypothesis longer than the reference can
score above 1.0 through insertions alone, so treat it as a rate, not a percentage
of correctness.

Normalization, and why it stops where it does
---------------------------------------------
Before alignment both strings are lower-cased, stripped of punctuation, and have
hyphens and slashes turned into spaces (so ``contrast-enhanced`` and ``contrast
enhanced`` are the same two words, which they are).

What normalization deliberately does **not** do is expand numerals. A transcript
reading ``180 over 110`` against a reference of ``one hundred eighty over one
hundred ten`` scores six errors here, despite being semantically perfect -- and
that is the honest answer for a metric measuring transcription, not comprehension.
Silently normalizing numbers would hide one of the most common real behaviours of
modern ASR models. The fix belongs in a downstream text normalizer that the caller
can choose to apply, not in the metric.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

#: Turned into spaces: these join words rather than terminate them.
_WORD_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[-–—/\\_]+")

#: Dropped entirely. An explicit class rather than "everything non-alphanumeric",
#: which would strip the letters of any non-Latin script it did not recognize.
#:
#: The danda (U+0964) and double danda (U+0965) are included because they are how
#: Bengali and Devanagari end a sentence -- the exact equivalent of a full stop. Omit
#: them and every Bengali sentence-final word compares as a mismatch ("আছে।" against
#: "আছে"), inflating WER for the language this service is built for.
_PUNCTUATION: Final[re.Pattern[str]] = re.compile(
    r"[.,;:!?\"'`()\[\]{}<>*&^%$#@~|+=।॥…]+"
)

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class WordErrorRate:
    """A WER score together with the alignment it came from."""

    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_words: int
    hypothesis_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def is_exact_match(self) -> bool:
        return self.errors == 0


def normalize(text: str) -> list[str]:
    """Lower-case, de-punctuate and tokenize into comparable words."""
    # NFKC first: full-width and compatibility characters would otherwise survive
    # de-punctuation and count as distinct words.
    folded = unicodedata.normalize("NFKC", text).lower()
    folded = _WORD_SEPARATORS.sub(" ", folded)
    folded = _PUNCTUATION.sub("", folded)
    return [token for token in _WHITESPACE.split(folded.strip()) if token]


def compute(reference: str, hypothesis: str) -> WordErrorRate | None:
    """Score ``hypothesis`` against ``reference``.

    Returns ``None`` when the reference has no words: WER divides by the reference
    length, so it is genuinely undefined there. Reporting 0.0 would claim a perfect
    transcription of nothing, and 1.0 would claim total failure -- both are
    fabrications, and the caller can tell the difference from ``None``.
    """
    reference_words = normalize(reference)
    hypothesis_words = normalize(hypothesis)

    if not reference_words:
        return None

    substitutions, deletions, insertions, hits = _align(reference_words, hypothesis_words)
    errors = substitutions + deletions + insertions

    return WordErrorRate(
        wer=errors / len(reference_words),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        hits=hits,
        reference_words=len(reference_words),
        hypothesis_words=len(hypothesis_words),
    )


def _align(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int, int]:
    """Levenshtein alignment over words, returning (S, D, I, hits).

    Two rolling rows rather than a full matrix: the edit *distance* needs only the
    previous row, and the operation counts are carried alongside it. A full
    ``len(ref) x len(hyp)`` matrix would be wasted memory on long-form audio, where
    a transcript can run to thousands of words.
    """
    reference_length, hypothesis_length = len(reference), len(hypothesis)

    # Each cell holds (cost, substitutions, deletions, insertions, hits).
    previous: list[tuple[int, int, int, int, int]] = [
        (index, 0, 0, index, 0) for index in range(hypothesis_length + 1)
    ]

    for row in range(1, reference_length + 1):
        # Consuming a reference word with nothing to match it is a deletion.
        current: list[tuple[int, int, int, int, int]] = [(row, 0, row, 0, 0)]

        for column in range(1, hypothesis_length + 1):
            if reference[row - 1] == hypothesis[column - 1]:
                cost, subs, dels, ins, hits = previous[column - 1]
                current.append((cost, subs, dels, ins, hits + 1))
                continue

            substitute = previous[column - 1]
            delete = previous[column]
            insert = current[column - 1]

            # Ties are broken substitution > deletion > insertion. Any consistent
            # order yields the same total error count; fixing one keeps the
            # component breakdown reproducible across runs and platforms.
            best_cost = min(substitute[0], delete[0], insert[0]) + 1
            if substitute[0] + 1 == best_cost:
                cost, subs, dels, ins, hits = substitute
                current.append((best_cost, subs + 1, dels, ins, hits))
            elif delete[0] + 1 == best_cost:
                cost, subs, dels, ins, hits = delete
                current.append((best_cost, subs, dels + 1, ins, hits))
            else:
                cost, subs, dels, ins, hits = insert
                current.append((best_cost, subs, dels, ins + 1, hits))

        previous = current

    _cost, substitutions, deletions, insertions, hits = previous[hypothesis_length]
    return substitutions, deletions, insertions, hits

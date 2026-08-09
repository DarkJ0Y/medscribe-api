"""Word error rate: alignment, normalization, and the cases where it is undefined."""

from __future__ import annotations

import pytest

from services import wer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected", "why"),
    [
        # (wer, S, D, I)
        ("the cat sat on the mat", "the cat sat on the mat", (0.0, 0, 0, 0), "identical"),
        ("the cat sat on the mat", "the cat sat on the hat", (1 / 6, 1, 0, 0), "one substitution"),
        ("the cat sat on the mat", "the cat sat on mat", (1 / 6, 0, 1, 0), "one deletion"),
        (
            "the cat sat on the mat",
            "the cat sat down on the mat",
            (1 / 6, 0, 0, 1),
            "one insertion",
        ),
        ("one two three", "four five six", (1.0, 3, 0, 0), "everything substituted"),
        ("one two three", "", (1.0, 0, 3, 0), "empty hypothesis deletes the whole reference"),
        # WER is a rate, not a percentage: insertions can push it past 1.0.
        ("hello", "hello there my old friend", (4.0, 0, 0, 4), "unbounded above"),
        ("a b c", "a x b y c", (2 / 3, 0, 0, 2), "interleaved insertions"),
    ],
)
def test_scores_known_alignments(
    reference: str, hypothesis: str, expected: tuple[float, int, int, int], why: str
) -> None:
    score = wer.compute(reference, hypothesis)

    assert score is not None
    expected_wer, subs, dels, ins = expected
    assert score.wer == pytest.approx(expected_wer), why
    assert (score.substitutions, score.deletions, score.insertions) == (subs, dels, ins), why
    # The identity that must always hold.
    assert score.errors == subs + dels + ins
    assert score.wer == pytest.approx(score.errors / score.reference_words)


@pytest.mark.parametrize(
    ("reference", "hypothesis", "why"),
    [
        ("The Cat SAT", "the cat sat", "case is folded"),
        ("the cat, sat.", "the cat sat", "punctuation is stripped"),
        ("contrast-enhanced CT", "contrast enhanced CT", "a hyphen is a word separator"),
        ("blood/urine test", "blood urine test", "a slash is a word separator"),
        ("the   cat\tsat", "the cat sat", "whitespace runs collapse"),
        ("  the cat sat  ", "the cat sat", "surrounding whitespace is irrelevant"),
        ("(the) cat!", "the cat", "brackets and exclamation stripped"),
    ],
)
def test_normalization_does_not_invent_errors(reference: str, hypothesis: str, why: str) -> None:
    score = wer.compute(reference, hypothesis)

    assert score is not None
    assert score.is_exact_match is True, why
    assert score.wer == 0.0


def test_strips_bengali_sentence_punctuation() -> None:
    """The danda is Bengali's full stop.

    Without stripping it, every sentence-final word mismatches and WER is inflated
    for the language this service is built for.
    """
    score = wer.compute("রোগীর জ্বর আছে।", "রোগীর জ্বর আছে")

    assert score is not None
    assert score.is_exact_match is True
    assert score.reference_words == 3


def test_does_not_normalize_numerals_away() -> None:
    """Deliberate: WER measures transcription, not comprehension.

    "180" for "one hundred eighty" is semantically perfect and still wrong as a
    transcription. Silently equating them would hide one of the most common real
    behaviours of modern ASR models -- see the module docstring.
    """
    score = wer.compute("one hundred eighty over one hundred ten", "180 over 110")

    assert score is not None
    assert score.is_exact_match is False
    assert score.reference_words == 7
    assert score.substitutions == 2
    assert score.deletions == 4


@pytest.mark.parametrize("reference", ["", "   ", ".,;", "\t\n"])
def test_undefined_when_the_reference_has_no_words(reference: str) -> None:
    """WER divides by the reference length, so it is genuinely undefined here.

    Returning 0.0 would claim a perfect transcription of nothing and 1.0 would claim
    total failure. Both are fabrications; None is the truth.
    """
    assert wer.compute(reference, "anything at all") is None
    assert wer.compute(reference, "") is None


def test_hits_and_counts_are_self_consistent() -> None:
    score = wer.compute("alpha beta gamma delta", "alpha BETA gamma epsilon zeta")

    assert score is not None
    # Every reference word is either a hit, a substitution or a deletion.
    assert score.hits + score.substitutions + score.deletions == score.reference_words
    # Every hypothesis word is either a hit, a substitution or an insertion.
    assert score.hits + score.substitutions + score.insertions == score.hypothesis_words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("contrast-enhanced", ["contrast", "enhanced"]),
        ("", []),
        ("...", []),
        ("ESR CRP", ["esr", "crp"]),
        ("ｆｕｌｌｗｉｄｔｈ", ["fullwidth"]),  # NFKC folds compatibility forms
    ],
)
def test_normalize_tokenizes_as_documented(text: str, expected: list[str]) -> None:
    assert wer.normalize(text) == expected


def test_alignment_is_deterministic_under_ties() -> None:
    """A fixed tie-break keeps the S/D/I breakdown reproducible.

    The total error count is the same whichever operation wins a tie, but the
    breakdown is what callers read to decide what went wrong, so it must not drift
    between runs or platforms.
    """
    first = wer.compute("a b c d", "w x y z")
    second = wer.compute("a b c d", "w x y z")

    assert first == second
    assert first is not None
    assert (first.substitutions, first.deletions, first.insertions) == (4, 0, 0)

"""Tests for the aggregation and uncertainty functions in evaluate.py.

These matter because the reported conclusions rest entirely on them, and both
have already been wrong once: outcomes were macro-averaged with zero-denominator
shifts silently dropped (comparing different subsets of shifts between arms),
and the effect was inferred from whether two marginal intervals overlapped,
which is not a test.

Synthetic data with known answers, so a broken implementation cannot hide.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate import (  # noqa: E402
    bootstrap_ci,
    mean_of,
    paired_bootstrap_difference,
    pooled_ratio,
)


def shifts(pairs):
    """Build shift records from (numerator, denominator) pairs."""
    return [{"num": n, "den": d} for n, d in pairs]


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def test_pooled_ratio_is_event_weighted_not_shift_weighted():
    """One shift with many events must not count the same as one with few.

    (1/1 and 1/9) pools to 2/10 = 0.2. Averaging the per-shift ratios would
    give (1.0 + 0.111)/2 = 0.56 - nearly triple, and wrong.
    """
    data = shifts([(1, 1), (1, 9)])
    assert pooled_ratio(data, "num", "den") == pytest.approx(0.2)


def test_pooled_ratio_counts_zero_event_shifts_without_dropping_them():
    data = shifts([(0, 0), (3, 6)])
    assert pooled_ratio(data, "num", "den") == pytest.approx(0.5)


def test_pooled_ratio_is_none_when_nothing_happened():
    assert pooled_ratio(shifts([(0, 0)]), "num", "den") is None


def test_macro_average_drops_empty_shifts_which_is_why_pooling_is_used():
    """Documents the failure mode the pooled version exists to avoid."""
    records = [{"rate": None}, {"rate": 1.0}]
    assert mean_of(records, "rate") == pytest.approx(1.0)  # the None vanished


# ---------------------------------------------------------------------------
# Marginal interval
# ---------------------------------------------------------------------------

def test_marginal_interval_brackets_the_point_estimate():
    data = shifts([(1, 2)] * 40)
    point = pooled_ratio(data, "num", "den")
    lo, hi = bootstrap_ci(data, "num", "den", draws=500)
    assert lo <= point <= hi


def test_marginal_interval_collapses_when_there_is_no_variation():
    """Every shift identical: nothing to resample, so the interval is a point."""
    data = shifts([(1, 4)] * 30)
    lo, hi = bootstrap_ci(data, "num", "den", draws=300)
    assert lo == pytest.approx(0.25) and hi == pytest.approx(0.25)


def test_marginal_interval_is_wider_with_fewer_shifts():
    """Positive control on the machinery: less data, more uncertainty."""
    noisy = [(1, 1), (0, 1), (1, 1), (0, 1)]
    small = bootstrap_ci(shifts(noisy * 2), "num", "den", draws=800)
    large = bootstrap_ci(shifts(noisy * 20), "num", "den", draws=800)
    assert (small[1] - small[0]) > (large[1] - large[0])


# ---------------------------------------------------------------------------
# Paired contrast - the actual effect estimate
# ---------------------------------------------------------------------------

def test_paired_difference_recovers_a_known_effect():
    arm_a = shifts([(8, 10)] * 40)   # rate 0.8
    arm_b = shifts([(3, 10)] * 40)   # rate 0.3
    point, (lo, hi) = paired_bootstrap_difference(arm_a, arm_b, "num", "den", draws=500)
    assert point == pytest.approx(0.5)
    assert lo > 0, "a real, consistent effect must exclude zero"


def test_paired_difference_includes_zero_when_arms_are_identical():
    """Negative case, paired with the positive one above."""
    arm = shifts([(5, 10), (7, 10), (2, 10)] * 12)
    point, (lo, hi) = paired_bootstrap_difference(arm, list(arm), "num", "den", draws=500)
    assert point == pytest.approx(0.0)
    assert lo <= 0 <= hi


def test_paired_difference_beats_comparing_two_marginal_intervals():
    """The reason the paired contrast exists.

    Both arms are noisy shift to shift, but the treatment adds a consistent
    amount within every matched pair. The marginal intervals overlap heavily;
    the paired difference still excludes zero. Reading overlap as "no effect"
    would have thrown this away.
    """
    base = [(1, 10), (9, 10), (2, 10), (8, 10), (3, 10), (7, 10)] * 6
    arm_b = shifts(base)
    arm_a = shifts([(n + 1, d) for n, d in base])  # +0.1 in every single shift

    lo_a, hi_a = bootstrap_ci(arm_a, "num", "den", draws=800)
    lo_b, hi_b = bootstrap_ci(arm_b, "num", "den", draws=800)
    assert lo_a < hi_b, "positive control: the marginal intervals must overlap"

    point, (lo, hi) = paired_bootstrap_difference(arm_a, arm_b, "num", "den", draws=800)
    assert point == pytest.approx(0.1, abs=1e-9)
    assert lo > 0, "the paired contrast should still detect a consistent effect"


def test_paired_difference_requires_matched_arms():
    assert paired_bootstrap_difference(
        shifts([(1, 2)]), shifts([(1, 2), (1, 2)]), "num", "den"
    ) is None


def test_paired_difference_is_deterministic_for_a_fixed_seed():
    arm_a = shifts([(6, 10), (4, 10)] * 15)
    arm_b = shifts([(3, 10), (5, 10)] * 15)
    first = paired_bootstrap_difference(arm_a, arm_b, "num", "den", draws=400)
    second = paired_bootstrap_difference(arm_a, arm_b, "num", "den", draws=400)
    assert first == second

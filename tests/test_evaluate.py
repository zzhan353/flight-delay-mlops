"""The gate must actually block bad models — including in the ways that matter."""

from __future__ import annotations

from flight_delay.evaluate import check


def candidate(pr_auc=0.42, ece=0.005, decile=1.75, base_pr=0.30, base_decile=1.55):
    return {
        "model": {"pr_auc": pr_auc, "ece": ece, "top_decile_lift": decile},
        "baseline": {"pr_auc": base_pr, "top_decile_lift": base_decile},
    }


GOOD = candidate()
CHAMPION = {"model": {"pr_auc": 0.41, "ece": 0.006, "top_decile_lift": 1.70}}
LIMITS = dict(
    min_decile_lift=1.08,
    min_pr_auc_floor=1.00,
    max_pr_auc_regression=0.01,
    max_ece=0.02,
)


def passed(results):
    return all(ok for ok, _ in results)


def test_good_model_passes():
    assert passed(check(GOOD, CHAMPION, **LIMITS))


def test_model_no_better_than_baseline_is_blocked():
    assert not passed(check(candidate(decile=1.58), None, **LIMITS))


def test_model_ranking_worse_than_baseline_is_blocked_by_the_floor():
    """Good top-decile lift must not excuse worse overall ranking."""
    assert not passed(check(candidate(pr_auc=0.28, decile=1.90), None, **LIMITS))


def test_regression_against_champion_is_blocked():
    assert not passed(check(candidate(pr_auc=0.38), CHAMPION, **LIMITS))


def test_small_regression_within_tolerance_is_allowed():
    """Holdout noise must not fail the build, or the gate gets disabled."""
    assert passed(check(candidate(pr_auc=0.405), CHAMPION, **LIMITS))


def test_miscalibrated_model_is_blocked_even_when_ranking_is_good():
    assert not passed(check(candidate(pr_auc=0.50, ece=0.09), CHAMPION, **LIMITS))


def test_first_ever_run_passes_without_a_champion():
    results = check(GOOD, None, **LIMITS)
    assert passed(results)
    assert any("champion" in msg for _, msg in results)

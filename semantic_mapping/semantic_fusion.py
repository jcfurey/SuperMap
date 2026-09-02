"""Bayesian semantic label fusion (Sec. IV-B.4, Eq. 10).

Open-vocabulary detectors are intermittent and view-dependent: the same
physical object may be reported as "chair" from one angle and "seat" (or
misclassified entirely) from another. Each object instance therefore keeps a
categorical belief over labels and fuses new detections recursively via
Bayes' rule, which suppresses transient misclassifications instead of
letting the last detection silently overwrite the label.
"""
from __future__ import annotations

LabelBelief = dict[str, float]


def new_belief(label: str, score: float, prior: float = 1.0) -> LabelBelief:
    """Initialize the categorical belief for a freshly spawned instance."""
    return {label: prior * max(score, 1e-3)}


def bayesian_label_update(
    belief: LabelBelief,
    observed_label: str,
    observed_score: float,
    p_self: float = 0.85,
    p_other: float = 0.05,
    new_label_prior: float = 1e-3,
) -> LabelBelief:
    """Fuse one detector observation z_t into the instance's label posterior.

    Implements P(L_j = c | z_1:t) = eta * P(z_t | L_j = c) * P(L_j = c | z_1:t-1)
    (Eq. 10). The detector confusion matrix P(z_t | L_j = c) is approximated
    as a two-level model: probability ``p_self`` (scaled by the detector's
    reported confidence) that the observation matches the true label, and a
    uniform ``p_other`` confusion probability spread over every other label
    currently under consideration for this instance. Labels never before
    seen for this instance are admitted with a small prior mass so repeated
    corroborating detections can still promote them.
    """
    if not belief:
        return new_belief(observed_label, observed_score)

    working = dict(belief)
    if observed_label not in working:
        working[observed_label] = new_label_prior

    effective_p_self = p_other + float(observed_score) * (p_self - p_other)

    unnormalized: LabelBelief = {}
    for label, prior_prob in working.items():
        likelihood = effective_p_self if label == observed_label else p_other
        unnormalized[label] = prior_prob * likelihood

    total = sum(unnormalized.values())
    if total <= 0.0:
        return working

    return {label: value / total for label, value in unnormalized.items()}


def prune_low_confidence_labels(belief: LabelBelief, min_prob: float = 1e-3) -> LabelBelief:
    """Drop negligible-probability label candidates and renormalize."""
    kept = {label: prob for label, prob in belief.items() if prob >= min_prob}
    if not kept:
        return belief
    total = sum(kept.values())
    return {label: prob / total for label, prob in kept.items()}


def best_label(belief: LabelBelief) -> tuple[str, float]:
    if not belief:
        return "unknown", 0.0
    label, prob = max(belief.items(), key=lambda kv: kv[1])
    return label, prob


def should_discard_instance(belief: LabelBelief, min_confidence: float, min_observations: int,
                             observation_count: int) -> bool:
    """Instance-level counterpart of "remove content whose posterior belief is
    too small" (Sec. IV-B.4): once an instance has had enough opportunities
    to settle on a label, drop it if it never became confident, which
    suppresses one-off false-positive detections from polluting the map.
    """
    if observation_count < min_observations:
        return False
    _label, confidence = best_label(belief)
    return confidence < min_confidence

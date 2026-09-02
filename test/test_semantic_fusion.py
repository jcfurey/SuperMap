from semantic_mapping import semantic_fusion as sf


def test_new_belief_is_single_label():
    belief = sf.new_belief("chair", 0.9)
    assert belief == {"chair": 0.9}


def test_repeated_corroborating_detections_increase_confidence():
    belief = sf.new_belief("chair", 0.6)
    for _ in range(5):
        belief = sf.bayesian_label_update(belief, "chair", 0.9)
    label, confidence = sf.best_label(belief)
    assert label == "chair"
    assert confidence > 0.95


def test_conflicting_detection_reduces_but_does_not_immediately_flip_label():
    belief = sf.new_belief("chair", 0.9)
    for _ in range(3):
        belief = sf.bayesian_label_update(belief, "chair", 0.9)
    before_label, before_confidence = sf.best_label(belief)

    updated = sf.bayesian_label_update(belief, "stool", 0.5)
    after_label, after_confidence = sf.best_label(updated)

    assert before_label == "chair"
    assert after_label == "chair"  # one weak conflicting observation shouldn't flip a confident belief
    assert after_confidence < before_confidence


def test_repeated_conflicting_detections_can_flip_the_label():
    belief = sf.new_belief("chair", 0.5)
    for _ in range(8):
        belief = sf.bayesian_label_update(belief, "stool", 0.95)
    label, _confidence = sf.best_label(belief)
    assert label == "stool"


def test_prune_low_confidence_labels_renormalizes():
    belief = {"chair": 0.98, "stool": 0.019, "table": 0.001}
    pruned = sf.prune_low_confidence_labels(belief, min_prob=0.01)
    assert set(pruned) == {"chair", "stool"}
    assert abs(sum(pruned.values()) - 1.0) < 1e-9


def test_should_discard_instance_requires_enough_observations_first():
    belief = sf.new_belief("chair", 0.1)
    assert sf.should_discard_instance(belief, min_confidence=0.5, min_observations=5, observation_count=2) is False
    assert sf.should_discard_instance(belief, min_confidence=0.5, min_observations=5, observation_count=5) is True


def test_should_discard_instance_keeps_confident_instance():
    belief = sf.new_belief("chair", 0.95)
    assert sf.should_discard_instance(belief, min_confidence=0.5, min_observations=1, observation_count=10) is False

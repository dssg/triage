from triage.component.catwalk.metrics import fpr
from triage.component.catwalk.evaluation import ModelEvaluator


def test_metric_directionality():
    """All metrics must be wrapped using the @Metric decorator available
    in catwalk.metrics to provide an `greater_is_better` attribute which must
    be one of True or False.
    """
    for met in ModelEvaluator.available_metrics.values():
        assert hasattr(met, "greater_is_better")
        assert met.greater_is_better in (True, False)


def test_fpr():
    predictions_binary = [1, 1, 1, 0, 0, 0, 0, 0]
    labels = [1, 1, 0, 1, 0, 0, 0, 1]

    result = fpr([], predictions_binary, labels, [])
    # false positives = 1
    # total negatives = 4
    assert result == 0.25

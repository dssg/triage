__all__ = ["BaselineFeatureNotInMatrix"]


class BaselineFeatureNotInMatrix(KeyError):
    """ This error is used to allow feature mixing and baseline classes to be
    included in the same experiment.

    Without error handling, the baseline classes would cause the experiment to
    end prematurely when they received a matrix without the required feature
    (if, for example, leave-one-out feature mixing is enabled). Raising this
    error will cause the model to be skipped elegantly.
    """

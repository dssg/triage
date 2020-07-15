import inspect
import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from numpy import exp, log, average

from .metric_directionality import greater_is_better, best_in_series, idxbest


def random_model_group(df, train_end_time, n=1):
    """Pick a random model group (as a baseline)

    Arguments:
        train_end_time (Timestamp): current train end time
        df (pandas.DataFrame): dataframe containing the columns
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                below_best
        n (int): the number of model group ids to return
    Returns: (int) the model group id to select, with highest current raw metric value
    """
    return df["model_group_id"].drop_duplicates().sample(frac=1).tolist()[:n]


def _mg_best_avg_by(df, value_col, metric, n=1):
    """Best model group in dataframe by average of some column

    Args:
        df (pandas.DataFrame)
        value_col (str): The column which contains the value to be averaged
        metric (str): the name of the column
        n (int): the number of model group ids to return
    """
    if n == 1:
        return [
            getattr(
                df.groupby(["model_group_id"])[value_col].mean().sample(frac=1),
                idxbest(metric),
            )()
        ]
    else:
        if greater_is_better(metric):
            return (
                df.groupby(["model_group_id"])[value_col]
                .mean()
                .nlargest(n)
                .index.tolist()
            )
        else:
            return (
                df.groupby(["model_group_id"])[value_col]
                .mean()
                .nsmallest(n)
                .index.tolist()
            )


def best_current_value(df, train_end_time, metric, parameter, n=1):
    """Pick the model group with the best current metric value

    Arguments:
        metric (string): model evaluation metric, such as 'precision@'
        parameter (string): model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp): current train end time
        df (pandas.DataFrame): dataframe containing the columns:
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                dist_from_best_case
        n (int): the number of model group ids to return
    Returns: (int) the model group id to select, with highest current raw metric value
    """
    curr_df = df.loc[
        (df["train_end_time"] == train_end_time)
        & (df["metric"] == metric)
        & (df["parameter"] == parameter)
    ]
    # sample(frac=1) to shuffle rows so we don't accidentally introduce bias in breaking ties
    best_raw_value = getattr(curr_df["raw_value"], best_in_series(metric))()
    if n <= 1:
        return (
            curr_df.loc[curr_df["raw_value"] == best_raw_value, "model_group_id"]
            .sample(frac=1)
            .tolist()
        )
    else:
        if greater_is_better(metric):
            result = curr_df.nlargest(n, "raw_value")["model_group_id"].tolist()
            return result
        else:
            result = curr_df.nsmallest(n, "raw_value")["model_group_id"].tolist()
            return result


def best_average_value(df, train_end_time, metric, parameter, n=1):
    """Pick the model with the highest average metric value so far

    Arguments:
        metric (string): model evaluation metric, such as 'precision@'
        parameter (string): model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp): current train end time
        df (pandas.DataFrame): dataframe containing the columns
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                dist_from_best_case
        n (int): the number of model group ids to return
    Returns: (int) the model group id to select, with highest mean raw metric value
    """
    met_df = df.loc[(df["metric"] == metric) & (df["parameter"] == parameter)]
    return _mg_best_avg_by(met_df, "raw_value", metric, n)


def lowest_metric_variance(df, train_end_time, metric, parameter, n=1):
    """Pick the model with the lowest metric variance so far

    Arguments:
        metric (string): model evaluation metric, such as 'precision@'
        parameter (string): model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp): current train end time
        df (pandas.DataFrame): dataframe containing the columns
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                below_best
        n (int): the number of model group ids to return
    Returns: (int) the model group id to select, with highest mean raw metric value
    """

    met_df = (
        df.loc[(df["metric"] == metric) & (df["parameter"] == parameter)]
        .groupby(["model_group_id"])["raw_value"]
        .std()
    )

    if met_df.isnull().sum() == met_df.shape[0]:
        # variance will be undefined in first time window since we only have one obseravtion
        # per model group
        logger.debug(
            "Null metric variances for %s %s at %s; picking at random",
            metric,
            parameter,
            train_end_time,
        )
        return df["model_group_id"].drop_duplicates().sample(n=n).tolist()
    elif met_df.isnull().sum() > 0:
        # the variances should be all null or no nulls, a mix shouldn't be possible
        # since we should have the same number of observations for every model group
        raise ValueError(
            "Mix of null and non-null metric variances for or {} {} at {}".format(
                metric, parameter, train_end_time
            )
        )
    if n == 1:
        # sample(frac=1) to shuffle rows so we don't accidentally introduce bias in breaking ties
        return [met_df.sample(frac=1).idxmin()]
    else:
        return met_df.nsmallest(n).index.tolist()


def most_frequent_best_dist(
    df, train_end_time, metric, parameter, dist_from_best_case, n=1
):
    """Pick the model that is most frequently within `dist_from_best_case` from the
    best-performing model group across test sets so far

    Arguments:
        dist_from_best_case (float): distance from the best performing model
        metric (string): model evaluation metric, such as 'precision@'
        parameter (string): model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp): current train end time
        df (pandas.DataFrame): dataframe containing the columns
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                below_best
        n (int): the number of model group ids to return
    Returns: (int) the model group id to select, with highest mean raw metric value
    """

    met_df = df.loc[(df["metric"] == metric) & (df["parameter"] == parameter)]
    met_df["within_dist"] = (df["dist_from_best_case"] <= dist_from_best_case).astype(
        "int"
    )
    if n == 1:
        # sample(frac=1) to shuffle rows so we don't accidentally introduce bias in breaking ties
        return [
            met_df.groupby(["model_group_id"])["within_dist"]
            .mean()
            .sample(frac=1)
            .idxmax()
        ]
    else:
        return (
            met_df.groupby(["model_group_id"])["within_dist"]
            .mean()
            .nlargest(n)
            .index.tolist()
        )


def best_average_two_metrics(
    df,
    train_end_time,
    metric1,
    parameter1,
    metric2,
    parameter2,
    metric1_weight=0.5,
    n=1,
):
    """Pick the model with the highest average combined value to date
    of two metrics weighted together using `metric1_weight`

    Arguments:
        metric1_weight (float): relative weight of metric1, between 0 and 1
        metric1 (string): model evaluation metric, such as 'precision@'
        parameter1 (string): model evaluation metric parameter,
            such as '300_abs'
        metric2 (string): model evaluation metric, such as 'precision@'
        parameter2 (string): model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp): current train end time
        df (pandas.DataFrame): dataframe containing the columns
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                below_best
        n (int): the number of model group ids to return
    Returns: (int) the model group id to select, with highest mean raw metric value
    """

    if metric1_weight < 0 or metric1_weight > 1:
        raise ValueError("Metric weight must be between 0 and 1")

    metric1_dir = greater_is_better(metric1)
    metric2_dir = greater_is_better(metric2)
    if metric1_dir != metric2_dir:
        raise ValueError("Metric directionalities must be the same")

    met_df = df.loc[
        ((df["metric"] == metric1) & (df["parameter"] == parameter1))
        | ((df["metric"] == metric2) & (df["parameter"] == parameter2))
    ]

    met_df.loc[
        (met_df["metric"] == metric1) & (met_df["parameter"] == parameter1),
        "weighted_raw",
    ] = (
        met_df.loc[
            (met_df["metric"] == metric1) & (met_df["parameter"] == parameter1),
            "raw_value",
        ]
        * metric1_weight
    )

    met_df.loc[
        (met_df["metric"] == metric2) & (met_df["parameter"] == parameter2),
        "weighted_raw",
    ] = met_df.loc[
        (met_df["metric"] == metric2) & (met_df["parameter"] == parameter2), "raw_value"
    ] * (
        1.0 - metric1_weight
    )

    met_df_wt = met_df.groupby(
        ["model_group_id", "train_end_time"], as_index=False
    ).sum()

    # sample(frac=1) to shuffle rows so we don't accidentally introduce bias in breaking ties
    return _mg_best_avg_by(met_df_wt, "weighted_raw", metric1, n)


def best_avg_var_penalized(df, train_end_time, metric, parameter, stdev_penalty, n=1):
    """Pick the model with the highest
     average metric value so far, placing less weight in older
     results. You need to specify two parameters: the shape of how the
     weight affects points (decay_type, linear or exponential) and the relative
     weight of the most recent point (curr_weight).

    Arguments:
        stdev_penalty (float): penalty for instability
        metric (string): model evaluation metric, such as 'precision@'
        parameter (string): model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp): current train end time
        df (pandas.DataFrame): dataframe containing the columns
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                below_best
        n (int): the number of model group ids to return
    Returns: (int) the model group id to select, with highest mean raw metric value
    """

    # for metrics where smaller values are better, the penalty for instability should
    # add to the mean, so introduce a factor of -1
    stdev_penalty = stdev_penalty if greater_is_better(metric) else -1.0 * stdev_penalty

    met_df = df.loc[(df["metric"] == metric) & (df["parameter"] == parameter)]
    met_df_grp = met_df.groupby(["model_group_id"]).aggregate(
        {"raw_value": ["mean", "std"]}
    )
    met_df_grp.columns = met_df_grp.columns.droplevel(0)
    met_df_grp.columns = ["raw_avg", "raw_stdev"]
    if met_df_grp["raw_stdev"].isnull().sum() == met_df_grp.shape[0]:
        # variance will be undefined in first time window since we only have one obseravtion
        # per model group
        logger.debug(
            "Null metric variances for %s %s at %s; just using mean",
            metric,
            parameter,
            train_end_time,
        )
        return [getattr(met_df_grp["raw_avg"].sample(frac=1), idxbest(metric))()]
    elif met_df_grp["raw_stdev"].isnull().sum() > 0:
        # the variances should be all null or no nulls, a mix shouldn't be possible
        # since we should have the same number of observations for every model group
        raise ValueError(
            "Mix of null and non-null metric variances for or {} {} at {}".format(
                metric, parameter, train_end_time
            )
        )

    min_stdev = met_df_grp["raw_stdev"].min()
    met_df_grp["penalized_avg"] = met_df_grp["raw_avg"] - stdev_penalty * (
        met_df_grp["raw_stdev"] - min_stdev
    )

    if n == 1:
        # sample(frac=1) to shuffle rows so we don't accidentally introduce bias in breaking ties
        return [getattr(met_df_grp["penalized_avg"].sample(frac=1), idxbest(metric))()]
    else:
        if greater_is_better(metric):
            return met_df_grp["penalized_avg"].nlargest(n).index.tolist()
        else:
            return met_df_grp["penalized_avg"].nsmallest(n).index.tolist()


def best_avg_recency_weight(
    df, train_end_time, metric, parameter, curr_weight, decay_type, n=1
):
    """Pick the model with the highest average metric value so far, penalized
    for relative variance as:
        avg_value - (stdev_penalty) * (stdev - min_stdev)
    where min_stdev is the minimum standard deviation of the metric across all
    model groups

    Arguments:
        decay_type (string): either 'linear' or 'exponential'; the shape of
            how the weights fall off between the current and first point
        curr_weight (float): amount of weight to put on the most recent point,
            relative to the first point (e.g., a value of 5.0 would mean the
            current data is weighted 5 times as much as the first one)
        metric (string): model evaluation metric, such as 'precision@'
        parameter (string): model evaluation metric parameter,
            such as '300_abs'
        train_end_time (Timestamp): current train end time
        df (pandas.DataFrame): dataframe containing the columns
                model_group_id,
                model_id,
                train_end_time,
                metric,
                parameter,
                raw_value,
                below_best
        n (int): the number of model group ids to return
    Returns: (int) the model group id to select, with highest mean raw metric value
    """

    # curr_weight is amount of weight to put on current point, relative to the first point
    # (e.g., if the first point has a weight of 1.0)
    # decay type is linear or exponetial

    first_date = df["train_end_time"].min()
    df["days_out"] = (df["train_end_time"] - first_date).apply(lambda x: float(x.days))
    tmax = df["days_out"].max()

    if tmax == 0:
        # only one date (must be on first time point), so everything gets a weight of 1
        df["weight"] = 1.0
    elif decay_type == "linear":
        # weight = (curr_weight - 1.0) * (t/tmax) + 1.0
        df["weight"] = (curr_weight - 1.0) * (df["days_out"] / tmax) + 1.0
    elif decay_type == "exponential":
        # weight = exp(ln(curr_weight)*t/tmax)
        df["weight"] = exp(log(curr_weight) * df["days_out"] / tmax)
    else:
        raise ValueError("Must specify linear or exponential decay type")

    def wm(x):
        return average(x, weights=df.loc[x.index, "weight"])

    met_df = df.loc[(df["metric"] == metric) & (df["parameter"] == parameter)]
    if n == 1:
        # sample(frac=1) to shuffle rows so we don't accidentally introduce bias in breaking ties
        result = getattr(
            met_df.groupby(["model_group_id"])
            .aggregate({"raw_value": wm})
            .sample(frac=1),
            idxbest(metric),
        )()
        return result.tolist()
    else:
        met_df_grp = met_df.groupby(["model_group_id"]).aggregate({"raw_value": wm})
        if greater_is_better(metric):

            return met_df_grp["raw_value"].nlargest(n).index.tolist()
        else:
            return met_df_grp["raw_value"].nsmallest(n).index.tolist()


SELECTION_RULES = {
    "random_model_group": random_model_group,
    "best_current_value": best_current_value,
    "best_average_value": best_average_value,
    "lowest_metric_variance": lowest_metric_variance,
    "most_frequent_best_dist": most_frequent_best_dist,
    "best_average_two_metrics": best_average_two_metrics,
    "best_avg_var_penalized": best_avg_var_penalized,
    "best_avg_recency_weight": best_avg_recency_weight,
}


class BoundSelectionRule:
    """A selection rule bound with a set of arguments

    Args:
        args (dict): A set of keyword arguments, that should be sufficient
            to call the function when a dataframe and train_end_time is added
        function_name (string, optional): The name of a function in SELECTION_RULES
        descriptive_name (string, optional): A descriptive name, used in charts
            If none is given it will be automatically constructed
        function (function, optional): A function
    """

    def __init__(self, args, function_name=None, descriptive_name=None, function=None):
        if not function_name and not function:
            raise ValueError("Need either function_name or function")

        if not descriptive_name and not function_name:
            raise ValueError("Need either descriptive_name or function_name")

        self.args = args
        self.function_name = function_name
        self._function = function
        self._descriptive_name = descriptive_name

    @property
    def function(self):
        if not self._function:
            self._function = SELECTION_RULES[self.function_name]
        return self._function

    @property
    def descriptive_name(self):
        if not self._descriptive_name:
            self._descriptive_name = self._build_descriptive_name()

        return self._descriptive_name

    def __str__(self):
        return self.descriptive_name

    def _build_descriptive_name(self):
        """Build a descriptive name for the bound selection rule

        Constructed using the function name and arguments.
        """
        argspec = inspect.getfullargspec(self.function)
        args = [arg for arg in argspec.args if arg not in ["df", "train_end_time", "n"]]
        return "_".join([self.function_name] + [str(self.args[key]) for key in args])

    def pick(self, dataframe, train_end_time):
        """Run the selection rule for a given time on a dataframe

        Args:
            dataframe (pandas.DataFrame)
            train_end_time (timestamp) Current train end time

        Returns: (int) a model group id
        """
        return self.function(dataframe, train_end_time, **(self.args))

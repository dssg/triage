import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from .metric_directionality import is_better_operator
import pandas as pd
from datetime import datetime


def _past_threshold(df, metric_filter):
    return df[
        is_better_operator(metric_filter["metric"])(
            df["raw_value"], metric_filter["threshold_value"]
        )
    ]


def _close_to_best_case(df, metric_filter):
    return df[df["dist_from_best_case"] < metric_filter["max_from_best"]]


def _of_metric(df, metric_filter):
    return df[
        (df["metric"] == metric_filter["metric"])
        & (df["parameter"] == metric_filter["parameter"])
    ]


def model_groups_filter(
    train_end_times, initial_model_group_ids, models_table, db_engine
):
    """Filter the models which related train end times don't contain the user-input
    train_end_times

    Before creating the distance table, we want to make sure the model_group_ids
    and train_end_times are reasonable:
        1. the input train_end_times should exisit in the database
        2. every model group should have the same train_end_times
    to prevent incomparable model groups from being populated to the distance table.

    Args:
        train_end_times (list) The set of train_end_times to consider during
                               the iteration
        initial_model_group_ids (list) The initial list of model group ids to
                                       narrow down
        models_table (string) The name of the results schema
        db_engine (sqlalchemy.engine) A database engine with access to results
                                      schema of a completed modeling run
    """

    if isinstance(train_end_times, str) or not hasattr(train_end_times, "__iter__"):
        raise TypeError("train_end_times should be a list of str or timestamp")

    if not bool(train_end_times):
        raise ValueError("train_end_times shouldn't be an empty list")

    end_times_sql = "ARRAY[{}]".format(
        ", ".join(
            "'{}'".format(
                end_time.strftime("%Y-%m-%d")
                if isinstance(end_time, datetime)
                else end_time
            )
            for end_time in train_end_times
        )
    )
    logger.debug("Checking if all model groups have the same train end times")
    logger.debug(f"Found {len(initial_model_group_ids)} total model groups")
    query = f"""
        SELECT model_group_id
        FROM (
            SELECT
                model_group_id,
                array_agg(distinct train_end_time::date::text) as train_end_time_list
            FROM triage_metadata.{models_table}
            WHERE model_group_id in ({','.join([str(m) for m in initial_model_group_ids])})
            GROUP BY model_group_id
        ) as t
        WHERE train_end_time_list @> {end_times_sql}
        """
    model_group_ids = {row['model_group_id'] for row in db_engine.execute(query)}

    if not model_group_ids:
        raise ValueError("The train_end_times passed in is not a subset of train end times of any model group. Please double check that all the model groups have the specified train end times.")

    dropped_model_groups = len(initial_model_group_ids) - len(model_group_ids)
    logger.debug(
        f"Dropped {dropped_model_groups} model groups which don't match the train end times"
    )
    logger.debug(f"Found {len(model_group_ids)} total model groups past the checker")


    return model_group_ids


class ModelGroupThresholder:
    def __init__(
        self,
        distance_from_best_table,
        train_end_times,
        initial_model_group_ids,
        initial_metric_filters,
    ):
        """Iteratively narrow down a list of model groups by changing thresholds
        for max below best model and minimum absolute value with respect to
        different metrics

        Args:
            distance_from_best_table (audition.DistanceFromBestTable)
                A pre-populated distance-from-best database table
            train_end_times (list) The set of train end times to consider during iteration
            initial_model_group_ids (list) The initial list of model group ids to
                narrow down

        """
        self.distance_from_best_table = distance_from_best_table
        self.train_end_times = train_end_times
        self._initial_model_group_ids = initial_model_group_ids
        self._metric_filters = initial_metric_filters

    def _filter_model_groups(self, df, filter_func):
        """Filter model groups by ensuring each of their metrics meets the given
            filtering function.

        Args:
            df (pandas.DataFrame): A set of rows in the format given by
                audition.DistanceFromBestTable.as_dataframe
            filter_func (function): A function that takes a dataframe and a
                metric filter and returns a subset of the dataframe

        Returns: (set) The model group ids that pass filtering
        """
        passing = set(self._initial_model_group_ids)
        for metric_filter in self._metric_filters:
            passing &= set(
                filter_func(_of_metric(df, metric_filter), metric_filter)[
                    "model_group_id"
                ]
            )
        return passing

    def model_groups_past_threshold(self, df):
        """Return the model groups in the dataframe that are above the
            currently-configured minimum value

        Args:
            df (pandas.DataFrame): A set of rows in the format given by
                audition.DistanceFromBestTable.as_dataframe
        Returns: (set) The model group ids above the minimum value for each metric
        """
        return self._filter_model_groups(df, _past_threshold)

    def model_groups_close_to_best_case(self, df):
        """Return the model groups in the dataframe that are close enough to
            the best value according to current metric filter configuration

        Args:
            df (pandas.DataFrame): A set of rows in the format given by
                audition.DistanceFromBestTable.as_dataframe
        Returns: (set) The model group ids close to the best value for each metric
        """
        return self._filter_model_groups(df, _close_to_best_case)

    def model_groups_passing_rules(self):
        """Return the model groups passing both the close-to-best and
            above-min checks based on the current filters.

        Works by ensuring that a model group passes the close-to-best check
        for at least one train end time, and that is passes the above-min
        check for *all* train end times. Model groups must pass both
        of these checks.

        Returns: (set) The passing model group ids
        """
        past_threshold_model_groups = set(self._initial_model_group_ids)
        close_to_best_model_groups = set()
        for train_end_time in self.train_end_times:
            df_as_of = self.distance_from_best_table.dataframe_as_of(
                model_group_ids=self._initial_model_group_ids,
                train_end_time=train_end_time,
            )
            close_to_best = self.model_groups_close_to_best_case(df_as_of)
            logger.debug(
                f"Found {len(close_to_best)} model groups close to best for {train_end_time}",
                            )
            close_to_best_model_groups |= close_to_best

            past_threshold = self.model_groups_past_threshold(df_as_of)
            logger.debug(
                f"Found {len(past_threshold)} model groups above min for {train_end_time}",
            )
            past_threshold_model_groups &= past_threshold

        total_model_groups = close_to_best_model_groups & past_threshold_model_groups
        logger.debug(
            "Found {len(total_model_groups)} total model groups past threshold"
        )
        return total_model_groups

    def update_filters(self, new_metric_filters):
        """Update the saved metric filters.

        Args: new_metric_filters (list) A list of metrics to filter model
            groups on, and how to filter them. Each entry should be a dict
            with the keys:

                metric (string) -- model evaluation metric, such as 'precision@'
                parameter (string) -- model evaluation metric parameter,
                    such as '300_abs'
                max_below_best (float) The maximum value that the given metric
                    can be below the best for a given train end time
                min_value (float) The minimum value that the given metric can be
        """
        if new_metric_filters != self._metric_filters:
            self._metric_filters = new_metric_filters

    @property
    def model_group_ids(self):
        return self.model_groups_passing_rules()

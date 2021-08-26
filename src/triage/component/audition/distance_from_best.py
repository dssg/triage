import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

import os
import numpy as np
import pandas as pd

from .utils import str_in_sql
from .metric_directionality import sql_rank_order, value_agg_funcs
from .plotting import plot_cats, plot_bounds, category_colordict, category_styledict


class DistanceFromBestTable:
    def __init__(self, db_engine, models_table, distance_table, agg_type):
        """A database table that stores the distance from models and the
        best model for that train end time for a variety of chosen metrics

        Args:
            db_engine (sqlalchemy.engine)
            models_table (string) The name of a models table in the database, pre-populated
            distance_table (string) The desired name of the distance table to be
                produced by this class
            agg_type (string) Method for aggregating metric values (for instance, if there
                are multiple models at a given train_end_time with different random seeds)
        """
        self.db_engine = db_engine
        self.models_table = models_table
        self.distance_table = distance_table
        self.agg_type = agg_type

    def _delete(self):
        """Delete the distance-from-best table if it exists"""
        self.db_engine.execute("drop table if exists {}".format(self.distance_table))

    def _create(self):
        """Create the distance-from-best table"""
        self.db_engine.execute(
            """create table {} (
            model_group_id int,
            train_end_time timestamp,
            metric text,
            parameter text,
            raw_value float,
            best_case float,
            dist_from_best_case float,
            raw_value_next_time float,
            dist_from_best_case_next_time float
        )""".format(
                self.distance_table
            )
        )

    def _populate(self, model_group_ids, train_end_times, metrics):
        """Populate the distance table with the given model groups, times, and metrics

        Args:
            model_group_ids (list) Model group ids to include in the distance table
            train_end_times (list) Train end times to include in the table
            metrics (list) Metrics and metric params to include in the table. Each
                row should be a dict with keys:
                        'metric' (e.g. 'precision@')
                        'parameter' (e.g. '100_abs')
                All models should have the test_results.evaluations table populated
                for all given model group ids, train end times, and metric/param combos
        """
        logger.debug("Populating data to distance table")
        for metric in metrics:
            self.db_engine.execute(
                """
                insert into {new_table}
                WITH first_evals AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY model_id
                        ORDER BY evaluation_start_time ASC, evaluation_end_time ASC
                        ) AS eval_rn
                    FROM test_results.evaluations
                    WHERE metric='{metric}' AND parameter='{parameter}' AND subset_hash=''
                ),
                metric_values AS (
                    SELECT
                        m.model_group_id,
                        m.train_end_time,
                        {metric_agg_fcn}(ev.stochastic_value) as value
                  FROM first_evals ev
                  JOIN triage_metadata.{models_table} m USING(model_id)
                  JOIN triage_metadata.model_groups mg USING(model_group_id)
                  WHERE m.model_group_id IN ({model_group_ids})
                        AND train_end_time in ({train_end_times})
                        AND ev.eval_rn = 1
                  GROUP BY model_group_id, train_end_time
                ),
                model_ranks AS (
                    SELECT
                        model_group_id,
                        train_end_time,
                        value,
                        row_number() OVER (
                            PARTITION BY train_end_time
                            ORDER BY value {metric_value_order}, RANDOM()
                        ) AS rank
                  FROM metric_values
                ),
                model_tols AS (
                  SELECT train_end_time, model_group_id,
                         rank,
                         value,
                         first_value(value) over (
                            partition by train_end_time
                            order by rank ASC
                        ) AS best_val
                  FROM model_ranks
                ),
                current_best_vals as (
                    SELECT
                        model_group_id,
                        train_end_time,
                        '{metric}',
                        '{parameter}',
                        value as raw_value,
                        best_val as best_case,
                        abs(value - best_val) dist_from_best_case
                    FROM model_tols
                )
                select
                    current_best_vals.*,
                    first_value(raw_value) over (
                        partition by model_group_id
                        order by train_end_time asc
                        rows between 1 following and unbounded following
                    ) raw_value_next_time,
                    first_value(dist_from_best_case) over (
                        partition by model_group_id
                        order by train_end_time asc
                        rows between 1 following and unbounded following
                    ) dist_from_best_case_next_time
                from current_best_vals
                order by train_end_time
            """.format(
                    model_group_ids=str_in_sql(model_group_ids),
                    train_end_times=str_in_sql(train_end_times),
                    models_table=self.models_table,
                    metric=metric["metric"],
                    parameter=metric["parameter"],
                    metric_value_order=sql_rank_order(metric["metric"]),
                    new_table=self.distance_table,
                    metric_agg_fcn=value_agg_funcs(metric["metric"])[self.agg_type],
                )
            )

    @property
    def observed_bounds(self):
        query = """
            SELECT
                metric,
                parameter,
                min(raw_value),
                max(raw_value)
            FROM {distance_table} dist
            GROUP BY metric, parameter
        """.format(
            distance_table=self.distance_table
        )
        return dict(
            ((metric, parameter), (minimum, maximum))
            for metric, parameter, minimum, maximum in self.db_engine.execute(query)
        )

    def create_and_populate(
        self, model_group_ids, train_end_times, metrics, delete=True
    ):
        """Creates and populates the distance table with the
            given model groups, times, and metrics

        Args:
            model_group_ids (list) Model group ids to include in the distance table
            train_end_times (list) Train end times to include in the table
            metrics (list) Metrics and parameters to include in the table. Each
                row should be a dict with keys:
                        'metric' (e.g. 'precision@')
                        'parameter' (e.g. '100_abs')
                All models should have the test_results.evaluations table populated
                for all given model group ids, train end times, and metric/param combos
            delete (boolean, optional) Delete any previous version of the
                distance table if it exists
        """
        if delete:
            self._delete()
        self._create()
        self._populate(model_group_ids, train_end_times, metrics)

    def as_dataframe(self, model_group_ids):
        """Return model-group-id subset of table as dataframe

        Args:
            model_group_ids (list) the desired model group ids

        Returns: (pandas.DataFrame) The data from the table corresponding
            to those model group ids
        """
        return pd.read_sql(
            "select * from {} where model_group_id in ({})".format(
                self.distance_table, str_in_sql(model_group_ids)
            ),
            self.db_engine,
        )

    def dataframe_as_of(self, model_group_ids, train_end_time):
        """Return model group id/train end time subset of table as dataframe

        Args:
            model_group_ids (list) the desired model group ids
            train_end_date (string) the desired train end time

        Returns: (pandas.DataFrame) The data from the table corresponding
            to those model group ids and train end time
        """
        base_df = self.as_dataframe(model_group_ids)
        return base_df[base_df["train_end_time"] == train_end_time]


class BestDistancePlotter:
    def __init__(self, distance_from_best_table, directory=None):
        """Generate a plot illustrating the effect of different below-best maximum
        thresholds across the dataset.

        Args:
            distance_from_best_table (audition.DistanceFromBestTable)
                A pre-populated distance-from-best database table
        """
        self.distance_from_best_table = distance_from_best_table
        self.directory = directory

        self.colordict = None
        self.styledict = None
        self.cmap_name = "tab10"

    def plot_bounds(self, metric, parameter):
        observed_min, observed_max = self.distance_from_best_table.observed_bounds[
            (metric, parameter)
        ]
        return plot_bounds(observed_min, observed_max)

    def plot_tick_dist(self, plot_min, plot_max):
        dist = plot_max - plot_min
        return dist / 100.0

    def generate_plot_data(self, metric, parameter, model_group_ids, train_end_times):
        """Fetch data necessary for producing the plot from the distance table

        Arguments:
            metric (string) -- model evaluation metric, such as 'precision@'
            parameter (string) -- model evaluation metric parameter,
                such as '300_abs'
            model_group_ids (list) - Model group ids to include in the dataset
            train_end_times (list) - Train end times to include in the dataset

        Returns: (pandas.DataFrame) The relevant models and the percentage of time
            each was within various thresholds of the best model at that time
        """
        model_group_union_sql = " union all ".join(
            [
                "(select {} as model_group_id)".format(model_group_id)
                for model_group_id in model_group_ids
            ]
        )
        plot_min, plot_max = self.plot_bounds(metric, parameter)
        plot_tick_dist = self.plot_tick_dist(plot_min, plot_max)
        sel_params = {
            "metric": metric,
            "parameter": parameter,
            "model_group_union_sql": model_group_union_sql,
            "distance_table": self.distance_from_best_table.distance_table,
            "model_group_str": str_in_sql(model_group_ids),
            "train_end_str": str_in_sql(train_end_times),
            "series_start": plot_min,
            "series_end": plot_max,
            "series_tick": plot_tick_dist,
        }
        sel = """\
            with model_group_ids as ({model_group_union_sql}),
            x_vals AS (
                SELECT m.model_group_id, s.distance
                FROM (SELECT GENERATE_SERIES(
                {series_start}, {series_end}, {series_tick}
                ) AS distance) s
                CROSS JOIN
                (
                SELECT DISTINCT model_group_id FROM model_group_ids
                ) m
            )
            SELECT dist.model_group_id, distance, mg.model_type,
                    COUNT(*) AS num_models,
                    AVG(CASE WHEN dist_from_best_case <= distance THEN 1 ELSE 0 END) AS pct_of_time
            FROM {distance_table} dist
            JOIN x_vals USING(model_group_id)
            JOIN triage_metadata.model_groups mg using (model_group_id)
            WHERE
                dist.metric='{metric}'
                AND dist.parameter='{parameter}'
                and model_group_id in ({model_group_str})
                and train_end_time in ({train_end_str})
            GROUP BY 1,2,3
        """.format(
            **sel_params
        )

        return pd.read_sql(sel, self.distance_from_best_table.db_engine).sort_values(
            ["model_group_id", "distance"]
        )

    def plot_all_best_dist(self, metric_filters, model_group_ids, train_end_times):
        """For each metric, plot the percentage of time that a model group is
        within X percentage points of the best-performing model group using that
        metric.

        Arguments:
            metric_filters (list) The metrics to plot. Each element should be
                a dict with the following keys:

                metric (string) -- model evaluation metric, such as 'precision@'
                parameter (string) -- model evaluation metric parameter,
                    such as '300_abs'
            model_group_ids (list) - Model group ids to include in the plot
            train_end_times (list) - Train end times to include in the plot

        """
        for metric_filter in metric_filters:
            logger.debug(
                "Building best distance plot for %s and %s",
                metric_filter,
                train_end_times,
            )
            df = self.generate_plot_data(
                metric=metric_filter["metric"],
                parameter=metric_filter["parameter"],
                model_group_ids=model_group_ids,
                train_end_times=train_end_times,
            )

            # set stable colors/styles by model type
            categories = np.unique(df['model_type'])
            if not self.colordict:
                self.colordict = category_colordict(self.cmap_name, categories, None)
            if not self.styledict:
                self.styledict = category_styledict(self.colordict, None)

            plot_best_dist(
                metric=metric_filter["metric"],
                parameter=metric_filter["parameter"],
                df_best_dist=df,
                directory=self.directory,
                colordict=self.colordict,
                styledict=self.styledict,
            )


def plot_best_dist(metric, parameter, df_best_dist, directory=None, **plt_format_args):
    """Generates a plot of the percentage of time that a model group is
    within X percentage points of the best-performing model group using a
    given metric. At each point in time that a set of model groups is
    evaluated, the performance of the best model is calculated and the
    difference in performace for all other models found relative to this.

    An (x,y) point for a given model group on the plot generated by this
    method means that across all of those tets sets, the model
    from that model group performed within X percentage points of the best
    model in y% of the test sets.

    The plot will contain a line for each given model group representing
    the cumulative percent of the time that the group is within Xpp of the
    best group for each value of X between 0 and 100. All groups ultimately
    reach (1,1) on this graph (as every model group must be within 100pp of
    the best model 100% of the time), and a model specification that always
    dominated the others in the experiment would start at (0,1) and remain
    at y=1 across the graph.

    Arguments:
        metric (string) -- model evaluation metric, such as 'precision@'
        parameter (string) -- model evaluation metric parameter, such as '300_abs'
        df_best_dist (pandas.DataFrame)
        **plt_format_args -- formatting arguments passed through to plot_cats()
    """

    cat_col = "model_type"
    plt_title = "Fraction of models X pp worse than best {} {}".format(
        metric, parameter
    )

    if directory:
        path_to_save = os.path.join(
            directory, f"distance_from_best_{metric}{parameter}.png"
        )
    else:
        path_to_save = None

    plot_cats(
        frame=df_best_dist,
        x_col="distance",
        y_col="pct_of_time",
        cat_col=cat_col,
        title=plt_title,
        x_label="distance from best {}".format(metric),
        y_label="fraction of models",
        x_ticks=np.arange(0, 1.1, 0.1),
        path_to_save=path_to_save,
        **plt_format_args,
    )

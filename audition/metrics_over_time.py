from audition.utils import str_in_sql
from audition.plotting import plot_cats
import pandas as pd


class MetricOverTimePlotter(object):
    def __init__(self, distance_from_best_table):
        """Generate a plot illustrating the value of a metric over time

        Args:
            distance_from_best_table (audition.DistanceFromBestTable)
                A pre-populated distance-from-best database table
        """
        self.distance_from_best_table = distance_from_best_table

    def plot_all(self, metric_filters, model_group_ids, train_end_times):
        """For each metric, plot the value of that metric over time

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
            df = self.generate_plot_data(
                metric=metric_filter['metric'],
                parameter=metric_filter['parameter'],
                model_group_ids=model_group_ids,
                train_end_times=train_end_times
            )
            self.plot(
                metric=metric_filter['metric'],
                parameter=metric_filter['parameter'],
                df_metric=df
            )

    def generate_plot_data(self, metric, parameter, model_group_ids, train_end_times):
        """Fetch data necessary for producing the plot from the distance table

        Arguments:
            metric (string) -- model evaluation metric, such as 'precision@'
            parameter (string) -- model evaluation metric parameter,
                such as '300_abs'
            model_group_ids (list) - Model group ids to include in the dataset
            train_end_times (list) - Train end times to include in the dataset

        Returns: (pandas.DataFrame) The relevant models and their performance
        on the given metric over time
        """

        base_df = pd.read_sql(
            '''select
    model_group_id,
    metric,
    parameter,
    train_end_time,
    raw_value,
    mg.model_type
from {dist_table} dist
join results.model_groups mg using (model_group_id)
where model_group_id in ({model_group_ids})
union
select
    0 model_group_id,
    metric,
    parameter,
    train_end_time,
    best_case,
    'best case' model_type
from {dist_table}
group by 1, 2, 3, 4, 5, 6
            '''.format(
                dist_table=self.distance_from_best_table.distance_table,
                model_group_ids=str_in_sql(model_group_ids)
            ),
            self.distance_from_best_table.db_engine
        )
        df = base_df[
            (base_df['train_end_time'].isin(train_end_times)) &
            (base_df['metric'] == metric) &
            (base_df['parameter'] == parameter)
        ]
        return df

    def plot(self, metric, parameter, df_metric, **plt_format_args):
        """Draw the plot representing the given data

        Arguments:
            metric (string) -- model evaluation metric, such as 'precision@'
            parameter (string) -- model evaluation metric parameter, such as '300_abs'
            df_metric (pandas.DataFrame)
            **plt_format_args -- formatting arguments passed through to plot_cats()
        """
        cat_col = 'model_type'
        plt_title = '{} {} over time'.format(metric, parameter)

        plot_cats(
            frame=df_metric,
            x_col='train_end_time',
            y_col='raw_value',
            cat_col=cat_col,
            highlight_grp='best case',
            title=plt_title,
            x_label='train end time',
            y_label='value of {}'.format(metric),
            x_ticks=df_metric['train_end_time'].unique(),
            **plt_format_args
        )

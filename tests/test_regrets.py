from audition.regrets import RegretCalculator, SelectionRulePlotter, BoundSelectionRule
import testing.postgresql
from sqlalchemy import create_engine
from tests.utils import create_sample_distance_table
from audition.selection_rules import best_current_value, best_average_value
import numpy
from unittest.mock import patch


def test_regret_calculator():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        distance_table, model_groups = create_sample_distance_table(engine)

        def pick_spiky(df, train_end_time):
            return model_groups['spiky'].model_group_id

        regret_calculator = RegretCalculator(
            distance_from_best_table=distance_table
        )

        regrets = regret_calculator.regrets_for_rule(
            bound_selection_rule=BoundSelectionRule(
                descriptive_name='spiky',
                function=pick_spiky,
                args={}
            ),
            model_group_ids=[mg.model_group_id for mg in model_groups.values()],
            train_end_times=['2014-01-01', '2015-01-01', '2016-01-01'],
            regret_metric='precision@',
            regret_parameter='100_abs',
        )
        assert regrets == [0.19, 0.3, 0.12]


def test_regret_calculator_with_args():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        distance_table, model_groups = create_sample_distance_table(engine)

        def pick_highest_avg(df, train_end_time, metric, parameter):
            assert len(df['train_end_time'].unique()) == 2
            subsetted = df[(df['metric'] == metric) & (df['parameter'] == parameter)]
            mean = subsetted.groupby(['model_group_id'])['raw_value'].mean()
            return mean.nlargest(1).index[0]

        regret_calculator = RegretCalculator(
            distance_from_best_table=distance_table
        )
        regrets = regret_calculator.regrets_for_rule(
            bound_selection_rule=BoundSelectionRule(
                descriptive_name='pick_highest_avg',
                function=pick_highest_avg,
                args={'metric': 'recall@', 'parameter': '100_abs'},
            ),
            model_group_ids=[mg.model_group_id for mg in model_groups.values()],
            train_end_times=['2015-01-01'],
            regret_metric='precision@',
            regret_parameter='100_abs',
        )
        # picking the highest avg recall will pick 'spiky' for this time
        assert regrets == [0.3]


def test_SelectionPlotter_create_plot_dataframe():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        distance_table, model_groups = create_sample_distance_table(engine)
        plotter = SelectionRulePlotter(
            regret_calculator=RegretCalculator(distance_table)
        )
        plot_df = plotter.create_plot_dataframe(
            bound_selection_rules=[
                BoundSelectionRule(
                    descriptive_name='best_current_precision',
                    function=best_current_value,
                    args={'metric': 'precision@', 'parameter': '100_abs'}
                ),
                BoundSelectionRule(
                    descriptive_name='best_avg_precision',
                    function=best_average_value,
                    args={'metric': 'precision@', 'parameter': '100_abs'}
                ),
            ],
            model_group_ids=[mg.model_group_id for mg in model_groups.values()],
            train_end_times=['2014-01-01', '2015-01-01', '2016-01-01'],
            regret_metric='precision@',
            regret_parameter='100_abs',
        )
        # assert that we have the right # of columns and a row for each % diff value
        assert plot_df.shape == (100 * 2, 3)

        # both selection rules have a regret lower than 70
        for value in plot_df[plot_df['regret'] == 0.70]['pct_of_time'].values:
            assert numpy.isclose(value, 1.0)

        # best avg precision rule should be within 0.14 1/3 of the time
        for value in plot_df[
            (plot_df['regret'] == 0.14) &
            (plot_df['selection_rule'] == 'best_avg_precision')
        ]['pct_of_time'].values:
            assert numpy.isclose(value, 1.0/3)


def test_SelectionPlotter_plot():
    with patch('audition.regrets.plot_cats') as plot_patch:
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            distance_table, model_groups = create_sample_distance_table(engine)
            plotter = SelectionRulePlotter(
                regret_calculator=RegretCalculator(distance_table)
            )
            plotter.plot_all_selection_rules(
                bound_selection_rules=[
                    BoundSelectionRule(
                        descriptive_name='best_current_precision',
                        function=best_current_value,
                        args={'metric': 'precision@', 'parameter': '100_abs'}
                    ),
                    BoundSelectionRule(
                        descriptive_name='best_avg_precision',
                        function=best_average_value,
                        args={'metric': 'precision@', 'parameter': '100_abs'}
                    ),
                ],
                model_group_ids=[mg.model_group_id for mg in model_groups.values()],
                train_end_times=['2014-01-01', '2015-01-01', '2016-01-01'],
                regret_metric='precision@',
                regret_parameter='100_abs',
            )
        assert plot_patch.called
        args, kwargs = plot_patch.call_args
        assert 'regret' in kwargs['frame']
        assert 'pct_of_time' in kwargs['frame']
        assert kwargs['x_col'] == 'regret'
        assert kwargs['y_col'] == 'pct_of_time'

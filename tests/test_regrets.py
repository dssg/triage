from audition.regrets import RegretCalculator
import testing.postgresql
from sqlalchemy import create_engine
from tests.utils import create_sample_distance_table


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
            selection_rule=pick_spiky,
            model_group_ids=[mg.model_group_id for mg in model_groups.values()],
            train_end_times=['2014-01-01', '2015-01-01', '2016-01-01'],
            metric='precision@',
            parameter='100_abs',
            selection_rule_args={}
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
            selection_rule=pick_highest_avg,
            model_group_ids=[mg.model_group_id for mg in model_groups.values()],
            train_end_times=['2015-01-01'],
            metric='precision@',
            parameter='100_abs',
            selection_rule_args={'metric': 'recall@', 'parameter': '100_abs'}
        )
        # picking the highest avg recall will pick 'spiky' for this time
        assert regrets == [0.3]

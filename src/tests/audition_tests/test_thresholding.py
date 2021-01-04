from unittest import TestCase
from datetime import datetime
import testing.postgresql
from sqlalchemy import create_engine

from triage.component.audition.distance_from_best import DistanceFromBestTable
from triage.component.audition.thresholding import (
    model_groups_filter,
    ModelGroupThresholder,
)
from triage.component.catwalk.db import ensure_db

from tests.results_tests.factories import (
    ModelFactory,
    ModelGroupFactory,
    init_engine,
    session,
)


class ModelGroupFilterTest(TestCase):
    def filter_train_end_times(self, engine, train_end_times):
        ensure_db(engine)
        init_engine(engine)
        mg1 = ModelGroupFactory(model_group_id=1, model_type="modelType1")
        mg2 = ModelGroupFactory(model_group_id=2, model_type="modelType2")
        mg3 = ModelGroupFactory(model_group_id=3, model_type="modelType3")
        mg4 = ModelGroupFactory(model_group_id=4, model_type="modelType4")
        mg5 = ModelGroupFactory(model_group_id=5, model_type="modelType5")
        # model group 1
        ModelFactory(model_group_rel=mg1, train_end_time=datetime(2014, 1, 1))
        ModelFactory(model_group_rel=mg1, train_end_time=datetime(2015, 1, 1))
        ModelFactory(model_group_rel=mg1, train_end_time=datetime(2016, 1, 1))
        ModelFactory(model_group_rel=mg1, train_end_time=datetime(2017, 1, 1))
        # model group 2 only has one timestamps
        ModelFactory(model_group_rel=mg2, train_end_time=datetime(2014, 1, 1))
        # model group 3
        ModelFactory(model_group_rel=mg3, train_end_time=datetime(2014, 1, 1))
        ModelFactory(model_group_rel=mg3, train_end_time=datetime(2015, 1, 1))
        ModelFactory(model_group_rel=mg3, train_end_time=datetime(2016, 1, 1))
        ModelFactory(model_group_rel=mg3, train_end_time=datetime(2017, 1, 1))
        # model group 4 only has two timestamps
        ModelFactory(model_group_rel=mg4, train_end_time=datetime(2015, 1, 1))
        ModelFactory(model_group_rel=mg4, train_end_time=datetime(2016, 1, 1))
        # model group 5 only has three timestamps
        ModelFactory(model_group_rel=mg5, train_end_time=datetime(2014, 1, 1))
        ModelFactory(model_group_rel=mg5, train_end_time=datetime(2015, 1, 1))
        ModelFactory(model_group_rel=mg5, train_end_time=datetime(2016, 1, 1))



        session.commit()
        model_groups = [1, 2, 3, 4, 5]
        model_group_ids = model_groups_filter(
            train_end_times=train_end_times,
            initial_model_group_ids=model_groups,
            models_table="models",
            db_engine=engine,
        )

        return model_group_ids

    def test_have_same_train_end_times(self):
        with testing.postgresql.Postgresql() as postgresql:
            custom_train_end_times = ["2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"]
            engine = create_engine(postgresql.url())
            # The filter will only let those models pass only if the model's train end times
            # contain the custom train end times
            pass_model_groups = self.filter_train_end_times(engine, custom_train_end_times)
            assert pass_model_groups == {1, 3}

    def test_have_partial_train_end_times(self):
        with testing.postgresql.Postgresql() as postgresql:
            custom_train_end_times = ["2014-01-01", "2015-01-01", "2016-01-01"]
            engine = create_engine(postgresql.url())
            pass_model_groups = self.filter_train_end_times(engine, custom_train_end_times)
            assert pass_model_groups == {1, 3, 5}

    def test_have_unmatched_train_end_times(self):
        with testing.postgresql.Postgresql() as postgresql:
            custom_train_end_times = ["2014-01-01", "2019-01-01"]
            engine = create_engine(postgresql.url())
            self.assertRaises(ValueError, lambda: self.filter_train_end_times(engine, custom_train_end_times))

class ModelGroupThresholderTest(TestCase):

    metric_filters = [
        {
            "metric": "precision@",
            "parameter": "100_abs",
            "max_from_best": 0.2,
            "threshold_value": 0.4,
        },
        {
            "metric": "recall@",
            "parameter": "100_abs",
            "max_from_best": 0.2,
            "threshold_value": 0.4,
        },
        {
            "metric": "false positives@",
            "parameter": "100_abs",
            "max_from_best": 30,
            "threshold_value": 50,
        },
    ]

    def setup_data(self, engine):
        ensure_db(engine)
        init_engine(engine)
        ModelGroupFactory(model_group_id=1, model_type="modelType1")
        ModelGroupFactory(model_group_id=2, model_type="modelType2")
        ModelGroupFactory(model_group_id=3, model_type="modelType3")
        ModelGroupFactory(model_group_id=4, model_type="modelType4")
        ModelGroupFactory(model_group_id=5, model_type="modelType5")
        session.commit()
        distance_table = DistanceFromBestTable(
            db_engine=engine, models_table="models", distance_table="dist_table", agg_type="worst"
        )
        distance_table._create()
        distance_rows = [
            # 2014: model group 1 should pass both close and min checks
            (1, "2014-01-01", "precision@", "100_abs", 0.5, 0.5, 0.0, 0.38),
            (1, "2014-01-01", "recall@", "100_abs", 0.5, 0.5, 0.0, 0.38),
            (1, "2014-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            # 2015: model group 1 should not pass close check
            (1, "2015-01-01", "precision@", "100_abs", 0.5, 0.88, 0.38, 0.0),
            (1, "2015-01-01", "recall@", "100_abs", 0.5, 0.88, 0.38, 0.0),
            (1, "2015-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            (1, "2016-01-01", "precision@", "100_abs", 0.46, 0.46, 0.0, 0.11),
            (1, "2016-01-01", "recall@", "100_abs", 0.46, 0.46, 0.0, 0.11),
            (1, "2016-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            # 2014: model group 2 should not pass min check
            (2, "2014-01-01", "precision@", "100_abs", 0.39, 0.5, 0.11, 0.5),
            (2, "2014-01-01", "recall@", "100_abs", 0.5, 0.5, 0.0, 0.38),
            (2, "2014-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            # 2015: model group 2 should pass both checks
            (2, "2015-01-01", "precision@", "100_abs", 0.69, 0.88, 0.19, 0.12),
            (2, "2015-01-01", "recall@", "100_abs", 0.69, 0.88, 0.19, 0.0),
            (2, "2015-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            (2, "2016-01-01", "precision@", "100_abs", 0.34, 0.46, 0.12, 0.11),
            (2, "2016-01-01", "recall@", "100_abs", 0.46, 0.46, 0.0, 0.11),
            (2, "2016-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            # model group 3 not included in this round
            (3, "2014-01-01", "precision@", "100_abs", 0.28, 0.5, 0.22, 0.0),
            (3, "2014-01-01", "recall@", "100_abs", 0.5, 0.5, 0.0, 0.38),
            (3, "2014-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            (3, "2015-01-01", "precision@", "100_abs", 0.88, 0.88, 0.0, 0.02),
            (3, "2015-01-01", "recall@", "100_abs", 0.5, 0.88, 0.38, 0.0),
            (3, "2015-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            (3, "2016-01-01", "precision@", "100_abs", 0.44, 0.46, 0.02, 0.11),
            (3, "2016-01-01", "recall@", "100_abs", 0.46, 0.46, 0.0, 0.11),
            (3, "2016-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            # 2014: model group 4 should not pass any checks
            (4, "2014-01-01", "precision@", "100_abs", 0.29, 0.5, 0.21, 0.21),
            (4, "2014-01-01", "recall@", "100_abs", 0.5, 0.5, 0.0, 0.38),
            (4, "2014-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            # 2015: model group 4 should not pass close check
            (4, "2015-01-01", "precision@", "100_abs", 0.67, 0.88, 0.21, 0.21),
            (4, "2015-01-01", "recall@", "100_abs", 0.5, 0.88, 0.38, 0.0),
            (4, "2015-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            (4, "2016-01-01", "precision@", "100_abs", 0.25, 0.46, 0.21, 0.21),
            (4, "2016-01-01", "recall@", "100_abs", 0.46, 0.46, 0.0, 0.11),
            (4, "2016-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            # 2014: model group 5 should not pass because precision is good but not recall
            (5, "2014-01-01", "precision@", "100_abs", 0.5, 0.38, 0.0, 0.38),
            (5, "2014-01-01", "recall@", "100_abs", 0.3, 0.5, 0.2, 0.38),
            (5, "2014-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            # 2015: model group 5 should not pass because precision is good but not recall
            (5, "2015-01-01", "precision@", "100_abs", 0.5, 0.88, 0.38, 0.0),
            (5, "2015-01-01", "recall@", "100_abs", 0.3, 0.88, 0.58, 0.0),
            (5, "2015-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            (5, "2016-01-01", "precision@", "100_abs", 0.46, 0.46, 0.0, 0.11),
            (5, "2016-01-01", "recall@", "100_abs", 0.3, 0.46, 0.16, 0.11),
            (5, "2016-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
            # 2014: model group 6 is failed by false positives
            (6, "2014-01-01", "precision@", "100_abs", 0.5, 0.5, 0.0, 0.38),
            (6, "2014-01-01", "recall@", "100_abs", 0.5, 0.5, 0.0, 0.38),
            (6, "2014-01-01", "false positives@", "100_abs", 60, 30, 30, 10),
            # 2015: model group 6 is failed by false positives
            (6, "2015-01-01", "precision@", "100_abs", 0.5, 0.88, 0.38, 0.0),
            (6, "2015-01-01", "recall@", "100_abs", 0.5, 0.38, 0.0, 0.38),
            (6, "2015-01-01", "false positives@", "100_abs", 60, 30, 30, 10),
            (6, "2016-01-01", "precision@", "100_abs", 0.46, 0.46, 0.0, 0.11),
            (6, "2016-01-01", "recall@", "100_abs", 0.5, 0.5, 0.0, 0.38),
            (6, "2016-01-01", "false positives@", "100_abs", 40, 30, 10, 10),
        ]
        for dist_row in distance_rows:
            engine.execute(
                "insert into dist_table values (%s, %s, %s, %s, %s, %s, %s, %s)",
                dist_row,
            )
        thresholder = ModelGroupThresholder(
            distance_from_best_table=distance_table,
            train_end_times=["2014-01-01", "2015-01-01"],
            initial_model_group_ids=[1, 2, 4, 5, 6],
            initial_metric_filters=self.metric_filters,
        )
        return thresholder

    def dataframe_as_of(self, thresholder, train_end_time):
        return thresholder.distance_from_best_table.dataframe_as_of(
            model_group_ids=thresholder._initial_model_group_ids,
            train_end_time=train_end_time,
        )

    def test_thresholder_2014_close(self):
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            thresholder = self.setup_data(engine)

            assert thresholder.model_groups_close_to_best_case(
                self.dataframe_as_of(thresholder, "2014-01-01")
            ) == set([1, 2])

    def test_thresholder_2015_close(self):
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            thresholder = self.setup_data(engine)
            assert thresholder.model_groups_close_to_best_case(
                self.dataframe_as_of(thresholder, "2015-01-01")
            ) == set([2])

    def test_thresholder_2014_threshold(self):
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            thresholder = self.setup_data(engine)
            assert thresholder.model_groups_past_threshold(
                self.dataframe_as_of(thresholder, "2014-01-01")
            ) == set([1])

    def test_thresholder_2015_threshold(self):
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            thresholder = self.setup_data(engine)
            assert thresholder.model_groups_past_threshold(
                self.dataframe_as_of(thresholder, "2015-01-01")
            ) == set([1, 2, 4])

    def test_thresholder_all_rules(self):
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            thresholder = self.setup_data(engine)
            # The multi-date version of this function should have
            # the mins ANDed together and the closes ORed together
            assert thresholder.model_groups_passing_rules() == set([1])

    def test_update_filters(self):
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            thresholder = self.setup_data(engine)
            assert thresholder.model_group_ids == set([1])
            thresholder.update_filters([])
            assert thresholder.model_group_ids == set([1, 2, 4, 5, 6])

from datetime import datetime, timedelta
from unittest.mock import patch

import factory
import numpy as np
import testing.postgresql
from sqlalchemy import create_engine

from triage.component.audition.distance_from_best import (
    DistanceFromBestTable,
    BestDistancePlotter,
)
from triage.component.catwalk.db import ensure_db

from tests.results_tests.factories import (
    EvaluationFactory,
    ModelFactory,
    ModelGroupFactory,
    init_engine,
    session,
)

from .utils import create_sample_distance_table


def _sql_add_days(sql_date, days):
    return datetime.strftime(
        datetime.strptime(sql_date, "%Y-%m-%d") + timedelta(days=days), "%Y-%m-%d"
    )


def test_DistanceFromBestTable():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        init_engine(engine)
        model_groups = {
            "stable": ModelGroupFactory(model_type="myStableClassifier"),
            "bad": ModelGroupFactory(model_type="myBadClassifier"),
            "spiky": ModelGroupFactory(model_type="mySpikeClassifier"),
        }

        class StableModelFactory(ModelFactory):
            model_group_rel = model_groups["stable"]

        class BadModelFactory(ModelFactory):
            model_group_rel = model_groups["bad"]

        class SpikyModelFactory(ModelFactory):
            model_group_rel = model_groups["spiky"]

        models = {
            "stable_3y_ago": StableModelFactory(train_end_time="2014-01-01"),
            "stable_2y_ago": StableModelFactory(train_end_time="2015-01-01"),
            "stable_1y_ago": StableModelFactory(train_end_time="2016-01-01"),
            "bad_3y_ago": BadModelFactory(train_end_time="2014-01-01"),
            "bad_2y_ago": BadModelFactory(train_end_time="2015-01-01"),
            "bad_1y_ago": BadModelFactory(train_end_time="2016-01-01"),
            "spiky_3y_ago": SpikyModelFactory(train_end_time="2014-01-01"),
            "spiky_2y_ago": SpikyModelFactory(train_end_time="2015-01-01"),
            "spiky_1y_ago": SpikyModelFactory(train_end_time="2016-01-01"),
        }

        class ImmediateEvalFactory(EvaluationFactory):
            evaluation_start_time = factory.LazyAttribute(
                lambda o: o.model_rel.train_end_time
            )
            evaluation_end_time = factory.LazyAttribute(
                lambda o: _sql_add_days(o.model_rel.train_end_time, 1)
            )

        class MonthOutEvalFactory(EvaluationFactory):
            evaluation_start_time = factory.LazyAttribute(
                lambda o: _sql_add_days(o.model_rel.train_end_time, 31)
            )
            evaluation_end_time = factory.LazyAttribute(
                lambda o: _sql_add_days(o.model_rel.train_end_time, 32)
            )

        class Precision100Factory(ImmediateEvalFactory):
            metric = "precision@"
            parameter = "100_abs"

        class Precision100FactoryMonthOut(MonthOutEvalFactory):
            metric = "precision@"
            parameter = "100_abs"

        class Recall100Factory(ImmediateEvalFactory):
            metric = "recall@"
            parameter = "100_abs"

        class Recall100FactoryMonthOut(MonthOutEvalFactory):
            metric = "recall@"
            parameter = "100_abs"

        for (add_val, PrecFac, RecFac) in (
            (0, Precision100Factory, Recall100Factory),
            (-0.15, Precision100FactoryMonthOut, Recall100FactoryMonthOut),
        ):
            PrecFac(model_rel=models["stable_3y_ago"], stochastic_value=0.6 + add_val)
            PrecFac(model_rel=models["stable_2y_ago"], stochastic_value=0.57 + add_val)
            PrecFac(model_rel=models["stable_1y_ago"], stochastic_value=0.59 + add_val)
            PrecFac(model_rel=models["bad_3y_ago"], stochastic_value=0.4 + add_val)
            PrecFac(model_rel=models["bad_2y_ago"], stochastic_value=0.39 + add_val)
            PrecFac(model_rel=models["bad_1y_ago"], stochastic_value=0.43 + add_val)
            PrecFac(model_rel=models["spiky_3y_ago"], stochastic_value=0.8 + add_val)
            PrecFac(model_rel=models["spiky_2y_ago"], stochastic_value=0.4 + add_val)
            PrecFac(model_rel=models["spiky_1y_ago"], stochastic_value=0.4 + add_val)
            RecFac(model_rel=models["stable_3y_ago"], stochastic_value=0.55 + add_val)
            RecFac(model_rel=models["stable_2y_ago"], stochastic_value=0.56 + add_val)
            RecFac(model_rel=models["stable_1y_ago"], stochastic_value=0.55 + add_val)
            RecFac(model_rel=models["bad_3y_ago"], stochastic_value=0.35 + add_val)
            RecFac(model_rel=models["bad_2y_ago"], stochastic_value=0.34 + add_val)
            RecFac(model_rel=models["bad_1y_ago"], stochastic_value=0.36 + add_val)
            RecFac(model_rel=models["spiky_3y_ago"], stochastic_value=0.35 + add_val)
            RecFac(model_rel=models["spiky_2y_ago"], stochastic_value=0.8 + add_val)
            RecFac(model_rel=models["spiky_1y_ago"], stochastic_value=0.36 + add_val)
        session.commit()
        distance_table = DistanceFromBestTable(
            db_engine=engine, models_table="models", distance_table="dist_table"
        )
        metrics = [
            {"metric": "precision@", "parameter": "100_abs"},
            {"metric": "recall@", "parameter": "100_abs"},
        ]
        model_group_ids = [mg.model_group_id for mg in model_groups.values()]
        distance_table.create_and_populate(
            model_group_ids, ["2014-01-01", "2015-01-01", "2016-01-01"], metrics
        )

        # get an ordered list of the models/groups for a particular metric/time
        query = """
            select model_id, raw_value, dist_from_best_case, dist_from_best_case_next_time
            from dist_table where metric = %s and parameter = %s and train_end_time = %s
            order by dist_from_best_case
        """

        prec_3y_ago = engine.execute(query, ("precision@", "100_abs", "2014-01-01"))
        assert [row for row in prec_3y_ago] == [
            (models["spiky_3y_ago"].model_id, 0.8, 0, 0.17),
            (models["stable_3y_ago"].model_id, 0.6, 0.2, 0),
            (models["bad_3y_ago"].model_id, 0.4, 0.4, 0.18),
        ]

        recall_2y_ago = engine.execute(query, ("recall@", "100_abs", "2015-01-01"))
        assert [row for row in recall_2y_ago] == [
            (models["spiky_2y_ago"].model_id, 0.8, 0, 0.19),
            (models["stable_2y_ago"].model_id, 0.56, 0.24, 0),
            (models["bad_2y_ago"].model_id, 0.34, 0.46, 0.19),
        ]

        assert distance_table.observed_bounds == {
            ("precision@", "100_abs"): (0.39, 0.8),
            ("recall@", "100_abs"): (0.34, 0.8),
        }


def test_BestDistancePlotter():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        distance_table, model_groups = create_sample_distance_table(engine)
        plotter = BestDistancePlotter(distance_table)
        df_dist = plotter.generate_plot_data(
            metric="precision@",
            parameter="100_abs",
            model_group_ids=[1, 2],
            train_end_times=["2014-01-01", "2015-01-01"],
        )
        # assert that we have the right # of columns and a row for each % diff value
        # 202 row because 101 percentiles (0-100 inclusive), 2 model groups
        assert df_dist.shape == (101 * 2, 5)

        # all of the model groups are within .34 of the best, so pick
        # a number higher than that and all should qualify
        for value in df_dist[df_dist["distance"] == 0.35]["pct_of_time"].values:
            assert np.isclose(value, 1.0)

        # model group 1 (stable) should be within 0.11 1/2 of the time
        # if we included 2016 in the train_end_times, this would be 1/3!
        for value in df_dist[
            (df_dist["distance"] == 0.11) & (df_dist["model_group_id"] == 1)
        ]["pct_of_time"].values:
            assert np.isclose(value, 0.5)


def test_BestDistancePlotter_plot():
    with patch("triage.component.audition.distance_from_best.plot_cats") as plot_patch:
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            distance_table, model_groups = create_sample_distance_table(engine)
            plotter = BestDistancePlotter(distance_table)
            plotter.plot_all_best_dist(
                [{"metric": "precision@", "parameter": "100_abs"}],
                model_group_ids=[1, 2],
                train_end_times=["2014-01-01", "2015-01-01"],
            )
        assert plot_patch.called
        args, kwargs = plot_patch.call_args
        assert "distance" in kwargs["frame"]
        assert "pct_of_time" in kwargs["frame"]
        assert kwargs["x_col"] == "distance"
        assert kwargs["y_col"] == "pct_of_time"


def test_BestDistancePlotter_plot_bounds():
    class FakeDistanceTable:
        @property
        def observed_bounds(self):
            return {
                ("precision@", "100_abs"): (0.02, 0.87),
                ("recall@", "100_abs"): (0.0, 1.0),
                ("false positives@", "300_abs"): (2, 162),
            }

    plotter = BestDistancePlotter(FakeDistanceTable())
    assert plotter.plot_bounds("precision@", "100_abs") == (0.0, 1.0)
    assert plotter.plot_bounds("recall@", "100_abs") == (0.0, 1.0)
    assert plotter.plot_bounds("false positives@", "300_abs") == (2, 178)

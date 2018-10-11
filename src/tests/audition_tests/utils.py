from triage.component.audition.distance_from_best import DistanceFromBestTable
from triage.component.catwalk.db import ensure_db

from tests.results_tests.factories import (
    ModelFactory,
    ModelGroupFactory,
    init_engine,
    session,
)


def create_sample_distance_table(engine):
    ensure_db(engine)
    init_engine(engine)
    model_groups = {
        "stable": ModelGroupFactory(model_type="myStableClassifier"),
        "spiky": ModelGroupFactory(model_type="mySpikeClassifier"),
    }

    class StableModelFactory(ModelFactory):
        model_group_rel = model_groups["stable"]

    class SpikyModelFactory(ModelFactory):
        model_group_rel = model_groups["spiky"]

    models = {
        "stable_3y_ago": StableModelFactory(train_end_time="2014-01-01"),
        "stable_2y_ago": StableModelFactory(train_end_time="2015-01-01"),
        "stable_1y_ago": StableModelFactory(train_end_time="2016-01-01"),
        "spiky_3y_ago": SpikyModelFactory(train_end_time="2014-01-01"),
        "spiky_2y_ago": SpikyModelFactory(train_end_time="2015-01-01"),
        "spiky_1y_ago": SpikyModelFactory(train_end_time="2016-01-01"),
    }
    session.commit()
    distance_table = DistanceFromBestTable(
        db_engine=engine, models_table="models", distance_table="dist_table"
    )
    distance_table._create()
    stable_grp = model_groups["stable"].model_group_id
    spiky_grp = model_groups["spiky"].model_group_id
    stable_3y_id = models["stable_3y_ago"].model_id
    stable_3y_end = models["stable_3y_ago"].train_end_time
    stable_2y_id = models["stable_2y_ago"].model_id
    stable_2y_end = models["stable_2y_ago"].train_end_time
    stable_1y_id = models["stable_1y_ago"].model_id
    stable_1y_end = models["stable_1y_ago"].train_end_time
    spiky_3y_id = models["spiky_3y_ago"].model_id
    spiky_3y_end = models["spiky_3y_ago"].train_end_time
    spiky_2y_id = models["spiky_2y_ago"].model_id
    spiky_2y_end = models["spiky_2y_ago"].train_end_time
    spiky_1y_id = models["spiky_1y_ago"].model_id
    spiky_1y_end = models["spiky_1y_ago"].train_end_time
    distance_rows = [
        (
            stable_grp,
            stable_3y_id,
            stable_3y_end,
            "precision@",
            "100_abs",
            0.5,
            0.6,
            0.1,
            0.5,
            0.15,
        ),
        (
            stable_grp,
            stable_2y_id,
            stable_2y_end,
            "precision@",
            "100_abs",
            0.5,
            0.84,
            0.34,
            0.5,
            0.18,
        ),
        (
            stable_grp,
            stable_1y_id,
            stable_1y_end,
            "precision@",
            "100_abs",
            0.46,
            0.67,
            0.21,
            0.5,
            0.11,
        ),
        (
            spiky_grp,
            spiky_3y_id,
            spiky_3y_end,
            "precision@",
            "100_abs",
            0.45,
            0.6,
            0.15,
            0.5,
            0.19,
        ),
        (
            spiky_grp,
            spiky_2y_id,
            spiky_2y_end,
            "precision@",
            "100_abs",
            0.84,
            0.84,
            0.0,
            0.5,
            0.3,
        ),
        (
            spiky_grp,
            spiky_1y_id,
            spiky_1y_end,
            "precision@",
            "100_abs",
            0.45,
            0.67,
            0.22,
            0.5,
            0.12,
        ),
        (
            stable_grp,
            stable_3y_id,
            stable_3y_end,
            "recall@",
            "100_abs",
            0.4,
            0.4,
            0.0,
            0.4,
            0.0,
        ),
        (
            stable_grp,
            stable_2y_id,
            stable_2y_end,
            "recall@",
            "100_abs",
            0.5,
            0.5,
            0.0,
            0.5,
            0.0,
        ),
        (
            stable_grp,
            stable_1y_id,
            stable_1y_end,
            "recall@",
            "100_abs",
            0.6,
            0.6,
            0.0,
            0.6,
            0.0,
        ),
        (
            spiky_grp,
            spiky_3y_id,
            spiky_3y_end,
            "recall@",
            "100_abs",
            0.65,
            0.65,
            0.0,
            0.65,
            0.0,
        ),
        (
            spiky_grp,
            spiky_2y_id,
            spiky_2y_end,
            "recall@",
            "100_abs",
            0.55,
            0.55,
            0.0,
            0.55,
            0.0,
        ),
        (
            spiky_grp,
            spiky_1y_id,
            spiky_1y_end,
            "recall@",
            "100_abs",
            0.45,
            0.45,
            0.0,
            0.45,
            0.0,
        ),
    ]
    for dist_row in distance_rows:
        engine.execute(
            "insert into dist_table values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            dist_row,
        )
    return distance_table, model_groups

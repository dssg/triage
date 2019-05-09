from triage.component.results_schema.utils import (
    save_experiment_and_get_hash,
    associate_models_with_experiment,
    associate_matrices_with_experiment,
    missing_model_hashes,
    missing_matrix_uuids,
)

from triage.component.results_schema.schema import Matrix, Model
from triage.component.catwalk.db import ensure_db
from sqlalchemy import create_engine
import testing.postgresql


def test_save_experiment_and_get_hash():
    # no reason to make assertions on the config itself, use a basic dict
    experiment_config = {"one": "two"}
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        exp_hash = save_experiment_and_get_hash(experiment_config, engine)
        assert isinstance(exp_hash, str)
        new_hash = save_experiment_and_get_hash(experiment_config, engine)
        assert new_hash == exp_hash


def test_missing_model_hashes():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)

        experiment_hash = save_experiment_and_get_hash({}, db_engine)
        model_hashes = ['abcd', 'bcde', 'cdef']

        # if we associate model hashes with an experiment but don't actually train the models
        # they should show up as missing
        associate_models_with_experiment(experiment_hash, model_hashes, db_engine)
        assert missing_model_hashes(experiment_hash, db_engine) == model_hashes

        # if we insert a model row they should no longer be considered missing
        db_engine.execute(
            f"insert into {Model.__table__.fullname} (model_hash) values (%s)",
            model_hashes[0]
        )
        assert missing_model_hashes(experiment_hash, db_engine) == model_hashes[1:]


def test_missing_matrix_uuids():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)

        experiment_hash = save_experiment_and_get_hash({}, db_engine)
        matrix_uuids = ['abcd', 'bcde', 'cdef']

        # if we associate matrix uuids with an experiment but don't actually build the matrices
        # they should show up as missing
        associate_matrices_with_experiment(experiment_hash, matrix_uuids, db_engine)
        assert missing_matrix_uuids(experiment_hash, db_engine) == matrix_uuids

        # if we insert a matrix row they should no longer be considered missing
        db_engine.execute(
            f"insert into {Matrix.__table__.fullname} (matrix_uuid) values (%s)",
            matrix_uuids[0]
        )
        assert missing_matrix_uuids(experiment_hash, db_engine) == matrix_uuids[1:]

from triage.component.catwalk.utils import (
    filename_friendly_hash,
    save_experiment_and_get_hash,
    associate_models_with_experiment,
    associate_matrices_with_experiment,
    missing_model_hashes,
    missing_matrix_uuids,
    sort_predictions_and_labels,
)
from triage.component.results_schema.schema import Matrix, Model
from triage.component.catwalk.db import ensure_db
from sqlalchemy import create_engine
import testing.postgresql
import datetime
import re
import numpy as np
from numpy.testing import assert_array_equal
import pytest


def test_filename_friendly_hash():
    data = {
        "stuff": "stuff",
        "other_stuff": "more_stuff",
        "a_datetime": datetime.datetime(2015, 1, 1),
        "a_date": datetime.date(2016, 1, 1),
        "a_number": 5.0,
    }
    output = filename_friendly_hash(data)
    assert isinstance(output, str)
    assert re.match(r"^[\w]+$", output) is not None

    # make sure ordering keys differently doesn't change the hash
    new_output = filename_friendly_hash(
        {
            "other_stuff": "more_stuff",
            "stuff": "stuff",
            "a_datetime": datetime.datetime(2015, 1, 1),
            "a_date": datetime.date(2016, 1, 1),
            "a_number": 5.0,
        }
    )
    assert new_output == output

    # make sure new data hashes to something different
    new_output = filename_friendly_hash({"stuff": "stuff", "a_number": 5.0})
    assert new_output != output


def test_filename_friendly_hash_stability():
    nested_data = {"one": "two", "three": {"four": "five", "six": "seven"}}
    output = filename_friendly_hash(nested_data)
    # 1. we want to make sure this is stable across different runs
    # so hardcode an expected value
    assert output == "9a844a7ebbfd821010b1c2c13f7391e6"
    other_nested_data = {"one": "two", "three": {"six": "seven", "four": "five"}}
    new_output = filename_friendly_hash(other_nested_data)
    assert output == new_output


def test_save_experiment_and_get_hash():
    # no reason to make assertions on the config itself, use a basic dict
    experiment_config = {"one": "two"}
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        exp_hash = save_experiment_and_get_hash(experiment_config, 1234, engine)
        assert isinstance(exp_hash, str)
        new_hash = save_experiment_and_get_hash(experiment_config, 1234, engine)
        assert new_hash == exp_hash


def test_missing_model_hashes():
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        ensure_db(db_engine)

        experiment_hash = save_experiment_and_get_hash({}, 1234, db_engine)
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

        experiment_hash = save_experiment_and_get_hash({}, 1234, db_engine)
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


def test_sort_predictions_and_labels():
    predictions = np.array([0.5, 0.4, 0.6, 0.5, 0.6])
    entities = np.array(range(6))
    labels = np.array([0, 0, 1, 1, None])

    # best sort
    sorted_predictions, sorted_labels = sort_predictions_and_labels(
        predictions, labels, tiebreaker='best'
    )
    assert_array_equal(sorted_predictions, np.array([0.6, 0.6, 0.5, 0.5, 0.4]))
    assert_array_equal(sorted_labels, np.array([1, None, 1, 0, 0]))

    # worst wort
    sorted_predictions, sorted_labels = sort_predictions_and_labels(
        predictions, labels, tiebreaker='worst'
    )
    assert_array_equal(sorted_predictions, np.array([0.6, 0.6, 0.5, 0.5, 0.4]))
    assert_array_equal(sorted_labels, np.array([None, 1, 0, 1, 0]))

    # random tiebreaker needs a seed
    with pytest.raises(ValueError):
        sort_predictions_and_labels(predictions, labels, tiebreaker='random')

    # random tiebreaker respects the seed
    sorted_predictions, sorted_labels = sort_predictions_and_labels(
        predictions,
        labels,
        tiebreaker='random',
        sort_seed=1234
    )
    assert_array_equal(sorted_predictions, np.array([0.6, 0.6, 0.5, 0.5, 0.4]))
    assert_array_equal(sorted_labels, np.array([None, 1, 1, 0, 0]))


    sorted_predictions, sorted_labels = sort_predictions_and_labels(
        predictions,
        labels,
        tiebreaker='random',
        sort_seed=24376234
    )
    assert_array_equal(sorted_predictions, np.array([0.6, 0.6, 0.5, 0.5, 0.4]))
    assert_array_equal(sorted_labels, np.array([None, 1, 0, 1, 0]))

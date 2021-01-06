import testing.postgresql
import datetime
from copy import copy

from sqlalchemy import create_engine
from triage.component.catwalk.db import ensure_db

from triage.component.catwalk.model_grouping import ModelGrouper
from .utils import sample_metadata


def test_model_grouping_default_config(sample_metadata):
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        model_grouper = ModelGrouper()
        # get the basic first model group with our default matrix
        assert (
            model_grouper.get_model_group_id(
                "module.Classifier", {"param1": "val1"}, sample_metadata, engine
            )
            == 1
        )

        # the end time is not by default a model group key so changing it
        # should still get us the same group
        metadata_new_end_time = copy(sample_metadata)
        metadata_new_end_time["end_time"] = datetime.date(2017, 3, 20)
        assert (
            model_grouper.get_model_group_id(
                "module.Classifier", {"param1": "val1"}, metadata_new_end_time, engine
            )
            == 1
        )

        # max_training_history is a default key,
        # so it should trigger a new group
        metadata_train_history = copy(sample_metadata)
        metadata_train_history["max_training_history"] = "3y"
        assert (
            model_grouper.get_model_group_id(
                "module.Classifier", {"param1": "val1"}, metadata_train_history, engine
            )
            == 2
        )

        # classifier is of course a default key as well
        assert (
            model_grouper.get_model_group_id(
                "module.OtherClassifier", {"param1": "val1"}, sample_metadata, engine
            )
            == 3
        )


def test_model_grouping_custom_config(sample_metadata):
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        model_grouper = ModelGrouper(
            model_group_keys=["feature_names", "as_of_date_frequency"]
        )
        # get the basic first model group with our default matrix
        assert (
            model_grouper.get_model_group_id(
                "module.Classifier", {"param1": "val1"}, sample_metadata, engine
            )
            == 1
        )

        # classifier is now not a key, so changing it should not get a new id
        assert (
            model_grouper.get_model_group_id(
                "module.OtherClassifier", {"param1": "val1"}, sample_metadata, engine
            )
            == 1
        )

        # as_of_date_frequency is a key,
        # so it should trigger a new group
        metadata_frequency = copy(sample_metadata)
        metadata_frequency["as_of_date_frequency"] = "2w"
        assert (
            model_grouper.get_model_group_id(
                "module.Classifier", {"param1": "val1"}, metadata_frequency, engine
            )
            == 2
        )

        # testing feature names may seem redundant but it is on a separate
        # code path so make sure its logic works
        metadata_features = copy(sample_metadata)
        metadata_features["feature_names"] = ["ft1", "ft3"]
        assert (
            model_grouper.get_model_group_id(
                "module.Classifier", {"param1": "val1"}, metadata_features, engine
            )
            == 3
        )

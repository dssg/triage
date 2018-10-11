"""Tests for Metta IO"""
import copy
import datetime
import os
import unittest
from dateutil.relativedelta import relativedelta
from shutil import rmtree
from tempfile import mkdtemp

import pandas as pd

from triage.component.metta import metta_io


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

example_data_csv = os.path.join(DATA, "titanic.csv")
example_data_h5 = os.path.join(DATA, "titanic.h5")

dict_test_config = {
    "feature_start_time": datetime.date(2016, 1, 1),
    "end_time": datetime.date(2016, 12, 31),
    "matrix_id": "testing_matrix",
    "label": "testing_data",
    "label_name": "Survived",
    "label_timespan": "1yr",
    "feature_names": ["break_last_3y", "soil", "pressure_zone"],
}


class TestMettaIO(unittest.TestCase):
    """Tests Metta IO functionality"""

    def setUp(self):
        self.temp_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.temp_dir)

    def temp_file(self, fname):
        return os.path.join(self.temp_dir, fname)

    def test_config(self):

        metta_io.check_config_types(dict_test_config)

    def test_uuid(self):
        fake_uuid = "d1193048d65fca97ed10d494c49e2d1e"

        assert fake_uuid == metta_io.generate_uuid(dict_test_config)

    def test_store_matrix(self):
        df_data = pd.read_csv(example_data_csv)

        metta_io._store_matrix(
            dict_test_config, df_data, "test_titanic", self.temp_dir, format="csv"
        )
        metta_io._store_matrix(
            dict_test_config, df_data, "test_titanich5", self.temp_dir, format="hd5"
        )

        # check it wrote to files
        assert os.path.isfile(self.temp_file("test_titanic.csv"))
        assert os.path.isfile(self.temp_file("test_titanich5.h5"))
        assert os.path.isfile(self.temp_file("test_titanic.yaml"))
        assert os.path.isfile(self.temp_file("test_titanich5.yaml"))

    def test_archive_matrix(self):
        df_data = pd.read_csv(example_data_csv)

        train_uuid = metta_io.archive_matrix(
            dict_test_config, example_data_csv, directory=self.temp_dir, format="csv"
        )

        train_uuid = metta_io.archive_matrix(
            dict_test_config, example_data_h5, directory=self.temp_dir, format="csv"
        )

        train_uuid = metta_io.archive_matrix(
            dict_test_config, df_data, directory=self.temp_dir, format="csv"
        )

        # check it wrote to files
        assert os.path.isfile(self.temp_file("{}.csv".format(train_uuid)))
        assert os.path.isfile(self.temp_file("{}.yaml".format(train_uuid)))

        def store_new_split(years):
            new_test_config = copy.deepcopy(dict_test_config)
            new_test_config["feature_start_time"] += relativedelta(years=years)
            new_test_config["end_time"] += relativedelta(years=years)
            return metta_io.archive_matrix(
                new_test_config,
                df_data,
                directory=self.temp_dir,
                format="csv",
                train_uuid=train_uuid,
            )

        test_uuids = [store_new_split(years) for years in range(1, 5)]

        for test_uuid in test_uuids:
            assert os.path.isfile(self.temp_file("{}.csv".format(test_uuid)))
            assert os.path.isfile(self.temp_file("{}.yaml".format(test_uuid)))

    def test_archive_train_test(self):
        df_data = pd.read_csv(example_data_csv)

        metta_io.archive_train_test(
            dict_test_config,
            df_data,
            dict_test_config,
            df_data,
            directory=self.temp_dir,
            format="hd5",
            overwrite=False,
        )

        # check that you don't write to a file again
        metta_io.archive_train_test(
            dict_test_config,
            df_data,
            dict_test_config,
            df_data,
            directory=self.temp_dir,
            format="hd5",
            overwrite=False,
        )

        assert os.path.isfile(self.temp_file("d1193048d65fca97ed10d494c49e2d1e.h5"))

        assert os.path.isfile(self.temp_file("d1193048d65fca97ed10d494c49e2d1e.yaml"))

        prior_creation_time = os.path.getmtime(
            self.temp_file("d1193048d65fca97ed10d494c49e2d1e.h5")
        )

        metta_io.archive_train_test(
            dict_test_config,
            df_data,
            dict_test_config,
            df_data,
            directory=self.temp_dir,
            format="hd5",
            overwrite=True,
        )

        later_creation_time = os.path.getmtime(
            self.temp_file("d1193048d65fca97ed10d494c49e2d1e.h5")
        )

        assert (later_creation_time - prior_creation_time) > 0

        assert len(os.listdir(self.temp_dir)) == 2

    def test_recover(self):
        df_data = pd.read_csv(example_data_csv)
        fake_uuid = "d1193048d65fca97ed10d494c49e2d1e"
        metta_io.archive_train_test(
            dict_test_config,
            df_data,
            dict_test_config,
            df_data,
            directory=self.temp_dir,
        )

        assert metta_io.recover_matrix("garbageuuid", directory=self.temp_dir) is None

        df_uuid = metta_io.recover_matrix(dict_test_config, directory=self.temp_dir)
        assert df_data.equals(df_uuid)

        df_config = metta_io.recover_matrix(fake_uuid, directory=self.temp_dir)
        assert df_data.equals(df_config)

"""
Tests for Metta IO
"""
import datetime
from dateutil.relativedelta import relativedelta
import metta.metta_io
import pandas as pd
import os
import unittest
from metta.datafiles import (example_uuid_fname,
                             example_data_csv,
                             example_data_h5)

from tempfile import mkdtemp
from shutil import rmtree
import copy


dict_test_config = {'start_time': datetime.date(2016, 1, 1),
                    'end_time': datetime.date(2016, 12, 31),
                    'matrix_id': 'testing_matrix',
                    'label': 'testing_data',
                    'label_name': 'SexCode',
                    'prediction_window': '1yr',
                    'feature_names': ['break_last_3y', 'soil',
                                      'pressure_zone']}


class TestMettaIO(unittest.TestCase):
    """Tests Metta IO functionality"""

    def setUp(self):
        self.temp_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.temp_dir)

    def temp_file(self, fname):
        return os.path.join(self.temp_dir, fname)

    def test_config(self):

        metta.metta_io.check_config_types(dict_test_config)

    def test_uuid(self):
        fake_uuid = 'f6187a0cfc4fc3af0f5febd040e9e07e'
        assert fake_uuid == metta.metta_io.generate_uuid(
            dict_test_config)

    def test_load_uuids(self):
        assert len(metta.metta_io.load_uuids(example_uuid_fname)) > 8
        assert len(metta.metta_io.load_uuids('notafile')) == 0

    def test_store_matrix(self):
        df_data = pd.read_csv(example_data_csv)

        metta.metta_io._store_matrix(
            dict_test_config,
            df_data,
            'test_titanic',
            self.temp_dir,
            format='csv'
        )
        metta.metta_io._store_matrix(
            dict_test_config,
            df_data,
            'test_titanich5',
            self.temp_dir,
            format='hd5'
        )

        # check it wrote to files
        assert os.path.isfile(self.temp_file('test_titanic.csv'))
        assert os.path.isfile(self.temp_file('test_titanich5.h5'))
        assert os.path.isfile(self.temp_file('test_titanic.yaml'))
        assert os.path.isfile(self.temp_file('test_titanich5.yaml'))

    def test_archive_matrix(self):
        df_data = pd.read_csv(example_data_csv)

        train_uuid = metta.metta_io.archive_matrix(
            dict_test_config,
            example_data_csv,
            directory=self.temp_dir,
            format='csv')

        train_uuid = metta.metta_io.archive_matrix(
            dict_test_config, example_data_h5,
            directory=self.temp_dir, format='csv')

        train_uuid = metta.metta_io.archive_matrix(
            dict_test_config, df_data,
            directory=self.temp_dir, format='csv')

        # check it wrote to files
        assert os.path.isfile(self.temp_file('{}.csv'.format(train_uuid)))
        assert os.path.isfile(self.temp_file('{}.yaml'.format(train_uuid)))

        def store_new_split(years):
            new_test_config = copy.deepcopy(dict_test_config)
            new_test_config['start_time'] += relativedelta(years=years)
            new_test_config['end_time'] += relativedelta(years=years)
            return metta.metta_io.archive_matrix(
                new_test_config,
                df_data,
                directory=self.temp_dir,
                format='csv',
                train_uuid=train_uuid
            )

        test_uuids = [store_new_split(years) for years in range(1, 5)]

        for test_uuid in test_uuids:
            assert os.path.isfile(self.temp_file('{}.csv'.format(test_uuid)))
            assert os.path.isfile(self.temp_file('{}.yaml'.format(test_uuid)))

    def test_archive_train_test(self):
        df_data = pd.read_csv(example_data_csv)

        metta.metta_io.archive_train_test(dict_test_config,
                                          df_data,
                                          dict_test_config,
                                          df_data,
                                          directory=self.temp_dir,
                                          format='hd5',
                                          overwrite=False)

        # check that you don't write to a file again
        metta.metta_io.archive_train_test(dict_test_config,
                                          df_data,
                                          dict_test_config,
                                          df_data,
                                          directory=self.temp_dir,
                                          format='hd5',
                                          overwrite=False)

        assert os.path.isfile(
            self.temp_file('f6187a0cfc4fc3af0f5febd040e9e07e.h5')
        )

        assert os.path.isfile(
            self.temp_file('f6187a0cfc4fc3af0f5febd040e9e07e.yaml')
        )

        prior_creation_time = os.path.getmtime(
            self.temp_file('f6187a0cfc4fc3af0f5febd040e9e07e.h5'))

        metta.metta_io.archive_train_test(dict_test_config,
                                          df_data,
                                          dict_test_config,
                                          df_data,
                                          directory=self.temp_dir,
                                          format='hd5',
                                          overwrite=True)

        later_creation_time = os.path.getmtime(
            self.temp_file('f6187a0cfc4fc3af0f5febd040e9e07e.h5'))

        assert (later_creation_time - prior_creation_time) > 0

        assert len(os.listdir(self.temp_dir)) == 2

    def test_recover(self):
        df_data = pd.read_csv(example_data_csv)
        fake_uuid = 'f6187a0cfc4fc3af0f5febd040e9e07e'
        metta.metta_io.archive_train_test(dict_test_config, df_data,
                                          dict_test_config, df_data,
                                          directory=self.temp_dir)

        assert metta.metta_io.recover_matrix(
            'garbageuuid', directory=self.temp_dir) is None

        df_uuid = metta.metta_io.recover_matrix(dict_test_config,
                                                directory=self.temp_dir)
        assert df_data.equals(df_uuid)

        df_config = metta.metta_io.recover_matrix(fake_uuid,
                                                  directory=self.temp_dir)
        assert df_data.equals(df_config)

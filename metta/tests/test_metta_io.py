"""
Tests for Metta IO
"""
import datetime
import metta.metta_io
import pandas as pd
import os
from metta.datafiles import example_uuid_fname, example_data

dict_test_config = {'start_time': datetime.date(2016, 1, 1),
                    'end_time': datetime.date(2016, 12, 31),
                    'matrix_id': 'testing_matrix',
                    'label': 'testing_data',
                    'label_name': 'SexCode',
                    'prediction_window': 1,
                    'feature_names': ['break_last_3y', 'soil',
                                      'pressure_zone']}


class TestMettaIO():
    """Tests Metta IO functionality"""

    def test_config(self):

        metta.metta_io.check_config_types(dict_test_config)

    def test_uuid(self):
        fake_uuid = '74f1ee3f-2a87-3f0c-a838-7fc806b355f5'
        assert fake_uuid == metta.metta_io.generate_uuid(
            dict_test_config)

    def test_load_uuids(self):
        assert len(metta.metta_io.load_uuids(example_uuid_fname)) > 8
        assert len(metta.metta_io.load_uuids('notafile')) == 0

    def test_store_matrix(self):
        df_data = pd.read_csv(example_data)

        metta.metta_io._store_matrix(
            dict_test_config, df_data, 'test_titanic', '/tmp/', format='csv')
        metta.metta_io._store_matrix(
            dict_test_config, df_data, 'test_titanich5', '/tmp/', format='hd5')
        metta.metta_io._store_matrix(
            dict_test_config, df_data, 'test_titanich5', '/tmp/', format='csv')
        # check it wrote to files
        os.path.isfile('/tmp/test_titanic.csv')
        os.path.isfile('/tmp/test_titanich5.h5')
        os.path.isfile('/tmp/test_titanic.yaml')
        os.path.isfile('/tmp/test_titanich5.yaml')
        os.path.isfile('/tmp/.matrix_uuids')

    def test_archive_train_test(self):
        df_data = pd.read_csv(example_data)

        metta.metta_io.archive_train_test(dict_test_config, df_data,
                                          dict_test_config, df_data,
                                          directory='/tmp/')

        # check that you don't write to a file again
        metta.metta_io.archive_train_test(dict_test_config, df_data,
                                          dict_test_config, df_data,
                                          directory='/tmp/')

        os.path.isfile('74f1ee3f-2a87-3f0c-a838-7fc806b355f5.h5')
        os.path.isfile('74f1ee3f-2a87-3f0c-a838-7fc806b355f5.yaml')
        os.path.isfile('acfda439-3fc1-3c01-9308-f397109973c6.h5')
        os.path.isfile('acfda439-3fc1-3c01-9308-f397109973c6.yaml')

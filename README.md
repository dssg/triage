# metta-data
Train Matrix and Test Matrix Storage
[![Build Status](https://travis-ci.org/dssg/metta-data.svg?branch=master)](https://travis-ci.org/dssg/metta-data)
[![codecov](https://codecov.io/gh/dssg/metta-data/branch/master/graph/badge.svg)](https://codecov.io/gh/dssg/metta-data)

## Short Description
Store and recall matrices.

## Long Description

Python library for storing meta data, pandas of training and
testing sets.

## How-to

`metta` expects you to hand it a dictionary for each dataframe with the following keys:
- `start_time` (date.datetime): The earliest time that enters your covariate calculations.
- `end_time` (date.dateime): The last time that enters your covariate calculations.
- `prediction_window` (int): The length of the prediction window you are using in this matrix (in months).
- `label_name` (str): The outcome variable's column name. This column must be in the last position in your dataframe.
- `matrix_id` (str): Human readable id for the dataset

```
import metta


train_config = {'start_time': datetime.date(2012, 12, 20),
                'end_time': datetime.date(2016, 12, 20),
                'prediction_window': 29,
                'label_name': 'inspection_1yr',
                'label_type': 'binary',
                'prediction_window': 3,
                'matrix_id': 'CDPH_2012',
                'feature_names': ['break_last_3yr', 'soil', 'pressure_zone']}


test_config = {'start_time': datetime.date(2015, 12, 20),
               'end_time': datetime.date(2016, 12, 21),
               'prediction_window': 29,
               'label_name': 'inspection_1yr',
               'label_type': 'binary',
               'prediction_window': 3,
               'matrix_id': 'CDPH_2015',
               'feature_names': ['break_last_3yr', 'soil', 'pressure_zone']}


metta.archive_train_test(train_config, X_train,
                         test_config, X_test,
                         directory='./old_matrices')


dict_config = yaml.load(open('aws_keys.yaml'))

metta.upload_to_s3(access_key_id=dict_config['AWSAccessKey'],
                   secret_access_key=dict_config['AWSSecretKey'],
                   bucket=dict_config['Bucket'],
                   folder=dict_config['Folder'],
                   directory='./old_matrices')

```

## Installation
```
pip install git+git://github.com/dssg/metta-data.git
```

## License
Licensed under MIT License

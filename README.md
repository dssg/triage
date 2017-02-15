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
- `prediction_window` (str): The length of the prediction window you are using in this matrix eg: '1y', '6m'
- `label_name` (str): The outcome variable's column name. This column must be in the last position in your dataframe.
- `matrix_id` (str): Human readable id for the dataset

### Storing a train and test pair
```
import metta


train_config = {'start_time': datetime.date(2012, 12, 20),
                'end_time': datetime.date(2016, 12, 20),
                'prediction_window': 29,
                'label_name': 'inspection_1yr',
                'label_type': 'binary',
                'prediction_window': '3m',
                'matrix_id': 'CDPH_2012',
                'feature_names': ['break_last_3yr', 'soil', 'pressure_zone']}


test_config = {'start_time': datetime.date(2015, 12, 20),
               'end_time': datetime.date(2016, 12, 21),
               'prediction_window': 29,
               'label_name': 'inspection_1yr',
               'label_type': 'binary',
               'prediction_window': '3m',
               'matrix_id': 'CDPH_2015',
               'feature_names': ['break_last_3yr', 'soil', 'pressure_zone']}


metta.archive_train_test(train_config, X_train,
                         test_config, X_test,
                         directory='./old_matrices')
```

### Storing a train and multiple test sets
```
import metta


train_config = {'start_time': datetime.date(2012, 12, 20),
                'end_time': datetime.date(2016, 12, 20),
                'prediction_window': '3m',
                'label_name': 'inspection_1yr',
                'label_type': 'binary',
                'matrix_id': 'CDPH_2012',
                'feature_names': ['break_last_3yr', 'soil', 'pressure_zone']}


base_test_config = {'start_time': datetime.date(2015, 12, 20),
               'end_time': datetime.date(2016, 12, 21),
               'prediction_window': '3m',
               'label_name': 'inspection_1yr',
               'label_type': 'binary',
               'matrix_id': 'CDPH_2015',
               'feature_names': ['break_last_3yr', 'soil', 'pressure_zone']}

train_uuid = metta.archive_matrix(train_config, X_train, directory='./matrices')

test_uuids = []

for years in range(1, 5):
	test_config = base_test_config.copy()
	test_config['start_time'] += relativedelta(years=years)
	test_config['end_time'] += relativedelta(years=years)
	test_config['matrix_id'] = 'CDPH_{}'.format(test_config['end_time'].year)
	test_uuids.append(metta.archive_matrix(
		test_config,
		df_data,
		'./matrices',
		format='csv',
		train_uuid=train_uuid
	))

```


### Uploading to S3
```
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

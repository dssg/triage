# metta-data
Train Matrix and Test Matrix Storage

[![Build Status](https://travis-ci.org/dssg/metta-data.svg?branch=master)](https://travis-ci.org/dssg/metta-data)
[![codecov](https://codecov.io/gh/dssg/metta-data/branch/master/graph/badge.svg)](https://codecov.io/gh/dssg/metta-data)


##  Description

Python library for storing and recalling meta data, and DataFrames of training and
testing sets.

## Installation
To get the latest stable version:
```
pip install metta-data
```

To get the current master branch:
```
pip install git+git://github.com/dssg/metta-data.git
```


## How-to

`metta` expects you to hand it a dictionary for each dataframe with the following keys:
- `feature_start_time` (date.datetime): The earliest time that enters your covariate calculations.
- `end_time` (date.dateime): The last time that enters your covariate calculations.
- `label_timespan` (str): The length of the labeling window you are using in this matrix eg: '1y', '6m'
- `label_name` (str): The outcome variable's column name. This column must be in the last position in your dataframe.
- `matrix_id` (str): Human readable id for the dataset

### Storing a train and test pair
```
import metta


train_config = {'feature_start_time': datetime.date(2012, 12, 20),
                'end_time': datetime.date(2016, 12, 20),
                'label_timespan': '3m',
                'label_name': 'inspection_1yr',
                'label_type': 'binary',
                'matrix_id': 'CDPH_2012',
                'feature_names': ['break_last_3yr', 'soil', 'pressure_zone'],
                'indices': ['entity_id', 'as_of_date'] }


test_config = {'feature_start_time': datetime.date(2015, 12, 20),
               'end_time': datetime.date(2016, 12, 21),
               'label_timespan': '3m',
               'label_name': 'inspection_1yr',
               'label_type': 'binary'
               'matrix_id': 'CDPH_2015',
               'feature_names': ['break_last_3yr', 'soil', 'pressure_zone'],
               'inidces': ['entity_id', 'as_of_date'] }


metta.archive_train_test(train_config,
                         X_train,
                         test_config,
                         X_test,
                         directory='./old_matrices',
                         format='hd5',
                         overwrite=False)
```

### Storing a train and multiple test sets
```
import metta


train_config = {'feature_start_time': datetime.date(2012, 12, 20),
                'end_time': datetime.date(2016, 12, 20),
                'label_timespan': '3m',
                'label_name': 'inspection_1yr',
                'label_type': 'binary',
                'matrix_id': 'CDPH_2012',
                'feature_names': ['break_last_3yr', 'soil', 'pressure_zone'],
                'indices': ['entity_id', 'as_of_date'] }


base_test_config = {'feature_start_time': datetime.date(2015, 12, 20),
               'end_time': datetime.date(2016, 12, 21),
               'label_timespan': '3m',
               'label_name': 'inspection_1yr',
               'label_type': 'binary',
               'matrix_id': 'CDPH_2015',
               'feature_names': ['break_last_3yr', 'soil', 'pressure_zone'],
               'indices': ['entity_id', 'as_of_date']}

train_uuid = metta.archive_matrix(train_config, X_train, directory='./matrices')

test_uuids = []

for years in range(1, 5):
	test_config = base_test_config.copy()
	test_config['feature_start_time'] += relativedelta(years=years)
	test_config['end_time'] += relativedelta(years=years)
	test_config['matrix_id'] = 'CDPH_{}'.format(test_config['end_time'].year)
	test_uuids.append(metta.archive_matrix(
		test_config,
		df_data,
		directory='./matrices',
        overwrite=False,
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

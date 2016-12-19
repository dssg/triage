# metta-data
Train Matrix and Test Matrix Storage

## Short Description
Store and recall matrices.

## Long Description

Python library for storing meta data, pandas of training and
testing sets.

## How-to

`metta` expects you to hand it a dictionary for each dataframe with the following keys:
- `start_time` (pandas.tslib.Timestamp): The earliest time that enters your covariate calculations.
- `end_time` (pandas.tslib.Timestamp): The last time that enters your covariate calculations.
- `prediction_start` (pandas.tslib.Timestamp): The beginning of the time window from which you are drawing outcomes.
- `prediction_end` (pandas.tslib.Timestamp): The end of the time window from which you are drawing outcomes.
- `label_name` (str): The outcome variable's column name. This column must be in the last position in your dataframe.

## Installation

## Dependencies

## Developers

## License
Licensed under MIT License

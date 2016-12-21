# metta-data
Train Matrix and Test Matrix Storage

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

## Installation

## Dependencies

## Developers

## License
Licensed under MIT License

# timechop
Generate temporal cross-validation time windows for matrix creation


[![Build Status](https://travis-ci.org/dssg/timechop.svg?branch=master)](https://travis-ci.org/dssg/timechop)

[![codecov](https://codecov.io/gh/dssg/timechop/branch/master/graph/badge.svg)](https://codecov.io/gh/dssg/timechop)

[![codeclimate](https://codeclimate.com/github/dssg/timechop.png)](https://codeclimate.com/github/dssg/timechop)

In predictive analytics, temporal cross-validation can be complicated. There are a variety of questions to balance: How frequently to retrain model? Should the time between rows for the same entity in the train and test matrices be different? Keeping track of how to create matrix time windows that successfully answer all of these questions is difficult. 

That's why we created timechop. Timechop takes in high-level time configuration (e.g. lists of train label windows, update windows) and returns all matrix time definitions. Here's an example:

```
chopper = Timechop(
	beginning_of_time = datetime.datetime(1990, 1, 1, 0, 0),
	modeling_start_time = datetime.datetime(2010, 1, 1, 0, 0),
	modeling_end_time = datetime.datetime(2010, 1, 16, 0, 0),
	update_window = '5 days',
	train_example_frequency = '1 days',
	test_example_frequency = '3 days',
	train_durations = ['5 days'],
	test_durations = ['5 days'],
	train_label_windows=['1 day'],
	test_label_windows=['3 months']
)
result = chopper.chop_time()
print(result)
```

```
[
            {
                'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
                'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 1, 0, 0),
                        datetime.datetime(2010, 1, 2, 0, 0),
                        datetime.datetime(2010, 1, 3, 0, 0),
                        datetime.datetime(2010, 1, 4, 0, 0),
                        datetime.datetime(2010, 1, 5, 0, 0)
                    ],
                    'label_window': '1 day',
                    'example_frequency': '1 days',
                    'train_duration': '5 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                    ],
                    'label_window': '3 months',
                    'example_frequency': '3 days'
                }]
            },
            {
                'beginning_of_time': datetime.datetime(1990, 1, 1, 0, 0),
                'modeling_start_time': datetime.datetime(2010, 1, 1, 0, 0),
                'modeling_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                'train_matrix': {
                    'matrix_start_time': datetime.datetime(2010, 1, 6, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 6, 0, 0),
                        datetime.datetime(2010, 1, 7, 0, 0),
                        datetime.datetime(2010, 1, 8, 0, 0),
                        datetime.datetime(2010, 1, 9, 0, 0),
                        datetime.datetime(2010, 1, 10, 0, 0)
                    ],
                    'label_window': '1 day',
                    'example_frequency': '1 days',
                    'train_duration': '5 days'
                },
                'test_matrices': [{
                    'matrix_start_time': datetime.datetime(2010, 1, 11, 0, 0),
                    'matrix_end_time': datetime.datetime(2010, 1, 16, 0, 0),
                    'as_of_times': [
                        datetime.datetime(2010, 1, 11, 0, 0),
                        datetime.datetime(2010, 1, 14, 0, 0)
                    ],
                    'label_window': '3 months',
                    'example_frequency': '3 days'
                }]
            }
        ]
```

Timechop currently works with the following:

- `beginning_of_time` - earliest date included in features
- `modeling_start_time` - earliest date in any model
- `modeling_end_time` - all dates in any model are < this date
- `update_window` - how frequently to retrain models
- `train_example_frequency` - time between rows for same entity in train matrix
- `test_example_frequency` - time between rows for same entity in test matrix
- `train_durations` - keep creating rows in train matrix for this duration
- `test_durations` - keep creating rows in test matrix for this duration
- `train_label_windows` - how much time is included in a label in the train matrix
- `test_label_windows` - how much time is included in a label in the test matrix

The output of Timechop works as input to the [architect.Planner](https://github.com/dssg/architect/blob/master/architect/planner.py).

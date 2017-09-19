from . import utils
from dateutil.relativedelta import relativedelta
import warnings
import logging
import itertools

class Timechop(object):
    def __init__(
        self,
        beginning_of_time,
        modeling_start_time,
        modeling_end_time,
        update_window,
        train_example_frequency,
        test_example_frequency,
        train_durations,
        test_durations,
        train_label_windows,
        test_label_windows
    ):
        self.beginning_of_time = beginning_of_time # earliest date at which features are calculated
        self.modeling_start_time = modeling_start_time # earliest date in any model
        self.modeling_end_time = modeling_end_time # all dates in any model are < this date
        self.update_window = update_window # how frequently to retrain models
        self.train_example_frequency = train_example_frequency # time between rows for same entity in train matrix
        self.test_example_frequency = test_example_frequency # time between rows for same entity in test matrix
        self.train_durations = train_durations # keep creating rows in train matrix for this duration
        self.test_durations = test_durations # keep creating rows in test matrix for this duration
        self.train_label_windows = train_label_windows # how much time is included in a label in the train matrix
        self.test_label_windows = test_label_windows # how much time is included in a label in the test matrix
        if beginning_of_time > modeling_start_time:
            raise ValueError('Beginning of time is later than modeling start time.')

    def chop_time(self):
        matrix_set_definitions = []
        for train_duration, train_label_window, test_label_window in itertools.product(
                self.train_durations,
                self.train_label_windows,
                self.test_label_windows
            ):
            matrix_end_times = self.calculate_matrix_end_times(
                train_duration,
                train_label_window,
                test_label_window,
            )
            for matrix_end_time in matrix_end_times:
                matrix_set_definitions.append(
                    self.generate_matrix_definition(
                        matrix_end_time,
                        train_duration,
                        train_label_window,
                        test_label_window
                    )
                )
        return(matrix_set_definitions)

    def calculate_matrix_end_times(
            self,
            train_duration,
            train_label_window,
            test_label_window
                    ):
        update_delta = utils.convert_str_to_relativedelta(self.update_window)
        train_delta = utils.convert_str_to_relativedelta(train_duration)
        test_label_delta = utils.convert_str_to_relativedelta(test_label_window)
        matrix_end_times = []
        matrix_end_time = self.modeling_end_time - test_label_delta
        
        print('Initial matrix end time {}'.format(matrix_end_time))
        if matrix_end_time <= self.modeling_start_time:
            raise ValueError('No valid training periods in modeling time.')

        while matrix_end_time > self.modeling_start_time:
            matrix_end_times.insert(0, matrix_end_time)
            matrix_end_time -= update_delta

        if (matrix_end_time != self.modeling_start_time):
            warnings.warn('''Modeling period not evenly divisbile by update
                windows. Matrix end times: {}
            '''.format(matrix_end_times))

        return(matrix_end_times)

    # matrix_end_time is now matrix_end_time - label_window
    def calculate_as_of_times(
        self,
        matrix_start_time,
        matrix_end_time,
        example_frequency
    ):
        example_delta = utils.convert_str_to_relativedelta(example_frequency)
        logging.info(
            'Calculate as_of_times from %s to %s using example frequency %s',
            matrix_start_time,
            matrix_end_time,
            example_delta
        )

        as_of_times = []
        as_of_time = matrix_start_time
        
        while as_of_time <= matrix_end_time:
            as_of_times.append(as_of_time)
            as_of_time += example_delta
        return(as_of_times)

    def generate_matrix_definition(
            self,
            train_matrix_end_time,
            train_duration,
            train_label_window,
            test_label_window
        ):
        train_delta = utils.convert_str_to_relativedelta(train_duration)
        train_label_delta = utils.convert_str_to_relativedelta(train_label_window)
        train_matrix_start_time = train_matrix_end_time - train_label_delta - train_delta
        if train_matrix_start_time < self.modeling_start_time:
            train_matrix_start_time = self.modeling_start_time
        print('train end: {}'.format(train_matrix_end_time))
        print('train start: {}'.format(train_matrix_start_time))
        train_as_of_times = self.calculate_as_of_times(
            train_matrix_start_time,
            train_matrix_end_time - train_label_delta,
            self.train_example_frequency
        )
        test_matrices = self.define_test_matrices(
            train_matrix_end_time,
            test_label_window
        )
            
        matrix_definition = {
            'beginning_of_time': self.beginning_of_time,
            'modeling_start_time': self.modeling_start_time,
            'modeling_end_time': self.modeling_end_time,
            'train_matrix': {
                'matrix_start_time': train_matrix_start_time,
                'matrix_end_time': train_matrix_end_time,
                'as_of_times': train_as_of_times,
                'label_window': train_label_window,
                'example_frequency': self.train_example_frequency,
                'train_duration': train_duration
            },
            'test_matrices': test_matrices
        }
        return(matrix_definition)

    def define_test_matrices(
        self,
        train_matrix_end_time,
        test_label_window
    ):
        test_definitions = []
        test_end_times = []
        for test_duration in self.test_durations:
            test_duration_delta = utils.convert_str_to_relativedelta(test_duration)
            test_label_delta = utils.convert_str_to_relativedelta(test_label_window)
            test_end_time = train_matrix_end_time + test_duration_delta
            if test_end_time > self.modeling_end_time - test_label_delta:
                test_end_time = max(train_matrix_end_time, self.modeling_end_time - test_label_delta)
            if test_end_time not in test_end_times:
                test_as_of_times = self.calculate_as_of_times(
                    train_matrix_end_time,
                    test_end_time,
                    self.test_example_frequency
                )
                test_definition = {
                    'matrix_start_time': train_matrix_end_time,
                    'matrix_end_time': test_end_time,
                    'as_of_times': test_as_of_times,
                    'label_window': test_label_window,
                    'example_frequency': self.test_example_frequency
                }
                test_definitions.append(test_definition)
                test_end_times.append(test_end_time)
        return(test_definitions)

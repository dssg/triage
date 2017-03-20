from . import utils
from dateutil.relativedelta import relativedelta
import warnings

class Inspections(object):
    def __init__(self, beginning_of_time, modeling_start_time,
                 modeling_end_time, update_window, look_back_durations):
        self.beginning_of_time = beginning_of_time # earliest date included in features
        self.modeling_start_time = modeling_start_time # earliest date in any model
        self.modeling_end_time = modeling_end_time # all dates in any model are < this date
        self.update_window = update_window # how frequently to retrain models
        self.look_back_durations = look_back_durations # length of time included in a model
        if beginning_of_time > modeling_start_time:
            raise ValueError('Beginning of time is later than modeling start time.')

    def chop_time(self):
        matrix_set_definitions = []
        for look_back_duration in self.look_back_durations:
            matrix_end_times = self.calculate_matrix_end_times(look_back_duration)
            for matrix_end_time in matrix_end_times:
                matrix_set_definitions.append(
                    self.generate_matrix_definition(
                        matrix_end_time,
                        look_back_duration
                    )
                )
        return(matrix_set_definitions)

    def calculate_matrix_end_times(self, look_back_duration):
        matrix_end_times = []
        update_delta = utils.convert_str_to_relativedelta(self.update_window)
        matrix_end_time = self.modeling_end_time - update_delta
        look_back_delta = utils.convert_str_to_relativedelta(look_back_duration)
        matrix_start_time = matrix_end_time - look_back_delta
        if matrix_start_time < self.modeling_start_time:
            raise ValueError('''
                Start of first matrix, {}, is before modeling start time, {}.
            '''.format(matrix_start_time, self.modeling_start_time))

        while matrix_start_time >= self.modeling_start_time:
            matrix_end_times.insert(0, matrix_end_time)
            matrix_end_time -= update_delta
            matrix_start_time = matrix_end_time - look_back_delta

        if (matrix_start_time != self.modeling_start_time):
            warnings.warn('''Modeling period not evenly divisbile by update
                windows and/or look back durations. Matrix end times: {}
            '''.format(matrix_end_times))

        return(matrix_end_times)

    def calculate_as_of_times(self, matrix_start_time, matrix_end_time):
        as_of_times = []
        as_of_time = matrix_start_time
        while as_of_time < matrix_end_time:
            as_of_times.append(as_of_time)
            as_of_time += relativedelta(days = 1)
        return(as_of_times)

    def generate_matrix_definition(self, train_matrix_end_time, look_back_duration):
        look_back_delta = utils.convert_str_to_relativedelta(look_back_duration)
        train_matrix_start_time = train_matrix_end_time - look_back_delta
        print('train end: {}'.format(train_matrix_end_time))
        print('train start: {}'.format(train_matrix_start_time))
        if train_matrix_start_time < self.modeling_start_time:
            raise ValueError('''Update period not evenly divisbile by lookback
                time. Matrix start time {} earlier than modeling start time {}.
            '''.format(train_matrix_start_time, self.modeling_start_time))
        train_as_of_times = self.calculate_as_of_times(
            train_matrix_start_time,
            train_matrix_end_time
        )
        test_as_of_times = self.calculate_as_of_times(
            train_matrix_end_time,
            self.modeling_end_time
        )
        matrix_definition = {
            'beginning_of_time': self.beginning_of_time,
            'modeling_start_time': self.modeling_start_time,
            'modeling_end_time': self.modeling_end_time,
            'train_matrix': {
                'matrix_start_time': train_matrix_start_time,
                'matrix_end_time': train_matrix_end_time,
                'as_of_times': train_as_of_times
            },
            'test_matrices': [{
                'matrix_start_time': train_matrix_end_time,
                'matrix_end_time': self.modeling_end_time,
                'as_of_times': test_as_of_times
            }]
        }
        return(matrix_definition)

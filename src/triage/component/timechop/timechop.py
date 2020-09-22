import itertools

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)

from triage.util.conf import convert_str_to_relativedelta, dt_from_str
from triage.util.structs import AsOfTimeList

from . import utils

# Throughout the code here, we're going to follow an example
# based around the following config:
# {
#
#   feature_start_time: '1995-01-01',
#   feature_end_time: '2017-07-01',
#
#   label_start_time: '2012-01-01',
#   label_end_time: '2017-07-01',
#
#   model_update_frequency: '1year',
#
#   training_label_timespans: ['6month'],
#   test_label_timespans: ['6month'],
#
#   max_training_histories: ['2year'],
#   test_durations: ['3month'],
#
#   training_as_of_date_frequencies='1day',
#   test_as_of_date_frequencies='1month'
#
# }


class Timechop:
    def __init__(
        self,
        feature_start_time,
        feature_end_time,
        label_start_time,
        label_end_time,
        model_update_frequency,
        training_as_of_date_frequencies,
        max_training_histories,
        training_label_timespans,
        test_as_of_date_frequencies,
        test_durations,
        test_label_timespans,
    ):

        '''
        Date strings should follow the format `YYYY-MM-DD`. Date intervals
        should be strings of the Postgres interval input format.

        This class is often used within the Triage experiment pipeline, and
        initialized using parameters from a Triage [experiment config](../../../experiments/experiment-config/#time-splitting)

        Arguments:
            feature_start_time (str): Earliest date included in any feature
            feature_end_time (str): Day after last feature date (all data
                included in features are before this date)
            label_start_time (str): Earliest date for which labels are available
            label_end_time (str): Day AFTER last label date (all dates in any
                model are before this date)
            model_update_frequency (str): how frequently to retrain models
            training_as_of_date_frequencies (str): time between rows for same
                entity in train matrix
            max_training_histories (str): Interval specifying how much history
                for each entity to train on
            training_label_timespans (str): how much time is included in a label
                in the train matrix
            test_as_of_date_frequencies (str): time between rows for same entity
                in test matrix
            test_durations (str): How long into the future to make predictions
                for each entity. Controls the length of time included in a test
                matrix
            test_label_timespans (str): How much time is included in a label
                in the test matrix.
        '''
        self.feature_start_time = dt_from_str(
            feature_start_time
        )
        self.feature_end_time = dt_from_str(
            feature_end_time
        )
        if self.feature_start_time > self.feature_end_time:
            raise ValueError("Feature start time after feature end time.")

        self.label_start_time = dt_from_str(
            label_start_time
        )
        self.label_end_time = dt_from_str(
            label_end_time
        )
        if self.label_start_time > self.label_end_time:
            raise ValueError("Label start time after label end time.")

        self.model_update_frequency = convert_str_to_relativedelta(
            model_update_frequency
        )

        self.training_as_of_date_frequencies = utils.convert_to_list(
            training_as_of_date_frequencies
        )

        self.test_as_of_date_frequencies = utils.convert_to_list(
            test_as_of_date_frequencies
        )

        self.max_training_histories = utils.convert_to_list(max_training_histories)

        self.test_durations = utils.convert_to_list(test_durations)

        self.training_label_timespans = utils.convert_to_list(training_label_timespans)

        self.test_label_timespans = utils.convert_to_list(test_label_timespans)

    def chop_time(self):
        """ Given the attributes of the object, define all train/test splits
        for all combinations of the temporal parameters.

        return:
            list: a list of dictionaries defining train/test splits
        """
        matrix_set_definitions = []
        # in our example, we just have one value for each of these: 6month, 6month, and 3month
        for (
            training_label_timespan,
            test_label_timespan,
            test_duration,
        ) in itertools.product(
            self.training_label_timespans,
            self.test_label_timespans,
            self.test_durations,
        ):
            # calculating the train-test split times starts from the end and walks backwards
            # e.g., train_test_split_times for our example with a 1 year model_update_frequency
            # will be every Oct. 1 from 2012 to 2016 (see comments in the method for details
            # on the calculation):
            # train_test_split_times = [2012-10-01, 2013-10-01, 2014-10-01, 2015-10-01, 2016-10-01]
            logger.spam(
                f"Calculating train/test split times for training prediction span {training_label_timespan}, "
                f"test prediction span {test_label_timespan}, test span {test_duration}"
            )
            train_test_split_times = self.calculate_train_test_split_times(
                training_label_timespan=convert_str_to_relativedelta(
                    training_label_timespan
                ),
                test_label_timespan=convert_str_to_relativedelta(test_label_timespan),
                test_duration=test_duration,
            )
            logger.spam(f"Train/test split times: {train_test_split_times}")

            # handle each training_as_of_date_frequency and max_training_history separately
            # to create matrices for each train_test_split_time.
            # in our example, we only have one value for each: 1day and 2year
            for (
                training_as_of_date_frequency,
                max_training_history,
            ) in itertools.product(
                self.training_as_of_date_frequencies, self.max_training_histories
            ):
                logger.spam(
                    f"Generating matrix definitions for training_as_of_date_frequency {training_as_of_date_frequency}, "
                    f"max_training_history {max_training_history}"
                )
                for train_test_split_time in train_test_split_times:
                    logger.spam(f"Generating matrix definitions for split {train_test_split_time}")
                    matrix_set_definitions.append(
                        self.generate_matrix_definitions(
                            train_test_split_time=train_test_split_time,
                            training_as_of_date_frequency=training_as_of_date_frequency,
                            max_training_history=max_training_history,
                            test_duration=test_duration,
                            training_label_timespan=training_label_timespan,
                            test_label_timespan=test_label_timespan,
                        )
                    )
        return matrix_set_definitions

    def calculate_train_test_split_times(
        self, training_label_timespan, test_label_timespan, test_duration
    ):
        """ Calculate the split times between train and test matrices. All
        label spans in train matrices will end at this time, and this will be
        the first as of time in the respective test matrix.

        Arguments:
            training_label_timespan (dateutil.relativedelta.relativedelta): how much
                time is included in training labels
            test_label_timespan (dateutil.relativedelta.relativedelta): how much time is included in test labels
            test_duration (str): for how long after the end of a training matrix are
                test predictions made

        Returns:
            list: all split times for the temporal parameters
        Raises:
            ValueError: if there are no valid split times in the temporal
            config
        """

        # we always want to be sure we're using the most recent data, so for the splits,
        # we start from the very end of time for which we have labels and walk backwards,
        # ensuring we leave enough of a buffer for the test_label_timespan to get a full
        # set of labels for our last testing as_of_date
        #
        # in our example, last_test_label_time = 2017-07-01 - 6month = 2017-01-01
        last_test_label_time = self.label_end_time - test_label_timespan

        # final label must be able to have feature data associated with it
        if last_test_label_time > self.feature_end_time:
            last_test_label_time = self.feature_end_time
            raise ValueError(
                "Final test label date cannot be after end of feature time."
            )
        logger.spam(f"Final label as of date: {last_test_label_time}")

        # all split times have to allow at least one training label before them
        # e.g., earliest_possible_split_time = max(1995-01-01, 2012-01-01) + 6month = 2012-01-01
        earliest_possible_split_time = training_label_timespan + max(
            self.feature_start_time, self.label_start_time
        )
        logger.spam(f"Earliest possible train/test split time: {earliest_possible_split_time}")

        # last split is the first as of time in the final test matrix
        # that is, starting from the label_end_time, we've walked back by the test_label_timespan
        # (above) to allow a buffer for labels and now we walk back further by the test_duration to
        # ensure we have a full set of test data in the latest test matrix.
        #
        # e.g., last_split_time = 2017-01-01 - 3month = 2016-10-01
        test_delta = convert_str_to_relativedelta(test_duration)
        last_split_time = last_test_label_time - test_delta
        logger.spam(f"Final split time: {last_split_time}")
        if last_split_time < earliest_possible_split_time:
            raise ValueError("No valid train/test split times in temporal config.")

        train_test_split_times = []
        train_test_split_time = last_split_time

        # finally, starting from our last_split_time, simply step backwards by the
        # model_update_frequency until we hit the earliest allowable time to
        # yield the set of train_test_split_times
        #
        # e.g., train_test_split_times for our example with a 1 year model_update_frequency
        # will be every Oct. 1 from 2012 to 2016:
        # train_test_split_times = [2012-10-01, 2013-10-01, 2014-10-01, 2015-10-01, 2016-10-01]
        while train_test_split_time >= earliest_possible_split_time:
            train_test_split_times.insert(0, train_test_split_time)
            train_test_split_time -= self.model_update_frequency

        return train_test_split_times

    # matrix_end_time is now matrix_end_time - label_window
    def calculate_as_of_times(
        self, as_of_start_limit, as_of_end_limit, data_frequency, forward=False
    ):
        """ Given a start and stop time, a frequncy, and a direction, calculate the
        as of times for a matrix.

        Arguments:
            as_of_start_limit (datetime.datetime): the earliest possible as of time for a matrix
            as_of_end_limit (datetime.datetime): the last possible as of time for the matrix
            data_frequency (str): The time interval that should pass between rows
                of a single entity. Of the format `'date unit'`. For example,
                `'1 month'`.
            forward (boolean): whether to generate times forward from the start time
                            (True) or backward from the end time (False)

        return:
            list: list of as of times for the matrix
        """
        logger.spam(f"Calculating as_of_times from {as_of_start_limit} to {as_of_end_limit} using example frequency {data_frequency}")

        as_of_times = []

        # in our example, this will apply to the test matrix with parameters
        #   as_of_start_limit = 2016-10-01, as_of_end_limit = 2017-01-01,
        #   data_frequency = 1month, forward=True
        # so, we'll start at 2016-10-01 and append this to the list of
        # as_of_times, then step forward one month at a time until we hit (but
        # do not include) 2017-01-01, yielding three values:
        #   [2016-10-01, 2016-11-01, 2016-12-01]
        if forward:
            as_of_time = as_of_start_limit
            # essentially a do-while loop for test matrices since
            # identical start and end times should include the first
            # date (e.g., ['2017-01-01', '2017-01-01') should give
            # preference to the inclusive side)
            as_of_times.append(as_of_time)
            as_of_time += data_frequency
            while as_of_time < as_of_end_limit:
                as_of_times.append(as_of_time)
                as_of_time += data_frequency

        # in our example, this will apply to the training matrix with parameters
        #   as_of_start_limit = 2014-04-01, as_of_end_limit = 2016-04-01,
        #   data_frequency = 1day, forward=False
        # so, we'll start from 2016-04-01 and step back by one day at a time
        # appending the results to the list of as_of_times until we hit 2014-04-01
        # (which will also be included)
        else:
            as_of_time = as_of_end_limit
            while as_of_time >= as_of_start_limit:
                as_of_times.insert(0, as_of_time)
                as_of_time -= data_frequency

        return as_of_times

    def generate_matrix_definitions(
        self,
        train_test_split_time,
        training_as_of_date_frequency,
        max_training_history,
        test_duration,
        training_label_timespan,
        test_label_timespan,
    ):
        """ Given a split time and parameters for train and test matrices,
        generate as of times and metadata for the matrices in the split.

        Arguments:
            train_test_split_time (datetime.datetime): the limit of the last label in the matrix
            training_as_of_date_frequency (str): how much time between rows for an entity
                                            in a training matrix
            max_training_history (str): how far back from split do train
                                        as_of_times go
            test_duration (str): how far forward from split do test as_of_times go
            training_label_timespan (str): how much time covered by train labels
            test_label_timespan (str): how much time is covered by test labels

        returns:
            dict: dictionary defining the train and test matrices for a split
        """

        # continuing our example, let's consider the case when this is called for the last
        # train_test_split_time, so the parameters here are:
        #   train_test_split_time = 2016-10-01
        #   training_as_of_date_frequency = 1day
        #   max_training_history = 2year
        #   test_duration = 3month
        #   training_label_timespan = 6month
        #   test_label_timespan = 6month

        # for the example, the train matrix will contain as_of_dates for every day from
        # 2014-04-01 through 2016-04-01, including _both_ endpoints, providing a 6 month
        # buffer between the last as_of_time and the train-test split time for the last
        # set of labels (see comments in the method for details)
        train_matrix_definition = self.define_train_matrix(
            train_test_split_time=train_test_split_time,
            training_label_timespan=training_label_timespan,
            max_training_history=max_training_history,
            training_as_of_date_frequency=training_as_of_date_frequency,
        )

        # for the example, the test matrix will contain three as_of_dates:
        #   [2016-10-01, 2016-11-01, 2016-12-01]
        # since we start at the train_test_split_time (2016-10-01) and walk forward by
        # the test_as_of_date_frequency (1 month) until we've exhausted the test_duration
        # (3 months), exclusive (see comments in the method for details)
        test_matrix_definitions = self.define_test_matrices(
            train_test_split_time=train_test_split_time,
            test_duration=test_duration,
            test_label_timespan=test_label_timespan,
        )

        matrix_set_definition = {
            "feature_start_time": self.feature_start_time,
            "feature_end_time": self.feature_end_time,
            "label_start_time": self.label_start_time,
            "label_end_time": self.label_end_time,
            "train_matrix": train_matrix_definition,
            "test_matrices": test_matrix_definitions,
        }
        logger.spam(f"Matrix definitions for train/test split {train_test_split_time}: {matrix_set_definition}")

        return matrix_set_definition

    def define_train_matrix(
        self,
        train_test_split_time,
        training_label_timespan,
        max_training_history,
        training_as_of_date_frequency,
    ):
        """ Given a split time and the parameters of a training matrix, generate
        the as of times and metadata for a train matrix.

        Arguments:
            train_test_split_time (datetime.datetime): the limit of the last label in the matrix
            training_label_timespan (str): how much time is covered by the labels
            max_training_history (str): how far back from split do as_of_times go
            training_as_of_date_frequency (str): how much time between rows for an entity

        return:
            dict: dictionary containing the temporal parameters and as of times
                  for a train matrix
        """
        logger.debug(f"Generating train matrix definitions for trin/test split {train_test_split_time}")
        # for our example, this will be called with:
        #   train_test_split_time = 2016-10-01
        #   training_label_timespan = 6month
        #   max_training_history = 2year
        #   training_as_of_date_frequency = 1day

        # last as of time in the matrix is 1 label span before split to provide
        # enough of a buffer for the label data to avoid spilling into the test
        # matrix and causing a leakage problem.
        #
        # e.g., last_train_as_of_time = 2016-10-01 - 6month = 2016-04-01
        training_prediction_delta = convert_str_to_relativedelta(
            training_label_timespan
        )
        last_train_as_of_time = train_test_split_time - training_prediction_delta

        # earliest time in matrix can't be farther back than the latest of the beginning
        # of label time or the beginning of feature time -- whichever is latest is the
        # limit if the amount of history we want to take would go further back.
        #
        # e.g., 2016-04-01 - 2year = 2014-04-01, which is later than both our
        # label_start_time (2012-01-01) and our feature_start_time (1995-01-01), so we
        # can use earliest_possible_train_as_of_time = 2014-04-01
        max_training_delta = convert_str_to_relativedelta(max_training_history)
        earliest_possible_train_as_of_time = last_train_as_of_time - max_training_delta
        experiment_as_of_time_limit = max(
            self.label_start_time, self.feature_start_time
        )
        if earliest_possible_train_as_of_time < experiment_as_of_time_limit:
            earliest_possible_train_as_of_time = experiment_as_of_time_limit
        logger.spam(f"Earliest possible train as of time: {earliest_possible_train_as_of_time}")

        # with the last as of time and the earliest possible time known,
        # calculate all the as of times for the matrix, stepping backwards
        # from the last as of time (to ensure that we use the latest possible
        # training data even if there's a gap and things don't line up
        # exactly) by the training_as_of_date_frequency
        #
        # for our example, this will give us a list of every day from 2014-04-01
        # through 2016-04-01, including _both_ endpoints
        train_as_of_times = self.calculate_as_of_times(
            as_of_start_limit=earliest_possible_train_as_of_time,
            as_of_end_limit=last_train_as_of_time,
            data_frequency=convert_str_to_relativedelta(training_as_of_date_frequency),
        )
        logger.spam(f"Train as of times: {train_as_of_times}")

        # create a dict of the matrix metadata
        matrix_definition = {
            "first_as_of_time": min(train_as_of_times),
            "last_as_of_time": max(train_as_of_times),
            "matrix_info_end_time": train_test_split_time,
            "as_of_times": AsOfTimeList(train_as_of_times),
            "training_label_timespan": training_label_timespan,
            "training_as_of_date_frequency": training_as_of_date_frequency,
            "max_training_history": max_training_history,
        }

        return matrix_definition

    def define_test_matrices(
        self, train_test_split_time, test_duration, test_label_timespan
    ):
        """ Given a train/test split time and a set of testing parameters,
        generate the metadata and as of times for the test matrices in a split.

        Arguments:
            train_test_split_time (datetime.datetime): the limit of the last label in the matrix
            test_duration (str): how far forward from split do test as_of_times go
            test_label_timespan (str): how much time is covered by test labels

        return:
            list: list of dictionaries defining the test matrices for a split
        """

        # for our example, this will be called with:
        #   train_test_split_time = 2016-10-01
        #   test_duration = 3month
        #   test_label_timespan = 6month

        # the as_of_time_limit is simply the split time plus the test_duration and we
        # can avoid checking here for any issues with the label_end_time or
        # feature_end_time since we've guaranteed that those limits would be
        # satisfied when we calculated the train_test_split_times initially
        #
        # for the example, as_of_time_limit = 2016-10-01 + 3month = 2017-01-01
        # (note as well that this will be treated as an _exclusive_ limit)
        logger.debug(f"Generating test matrix definitions for train/test split {train_test_split_time}")
        test_definitions = []
        test_delta = convert_str_to_relativedelta(test_duration)
        as_of_time_limit = train_test_split_time + test_delta
        logger.spam("All test as of times before %s", as_of_time_limit)

        # calculate the as_of_times associated with each test data frequency
        # for our example, we just have one, 1month
        for test_as_of_date_frequency in self.test_as_of_date_frequencies:
            logger.spam(f"Generating test matrix definitions for test data frequency {test_as_of_date_frequency}")

            # for test as_of_times we step _forwards_ from the train_test_split_time
            # to ensure that we always have a prediction set made immediately after
            # training is done (so, the freshest possible predictions) even if the
            # frequency doesn't divide the test_duration evenly so there's a gap before
            # the as_of_time_limit
            #
            # for our example, this will give three as_of_dates:
            #   [2016-10-01, 2016-11-01, 2016-12-01]
            # since we start at the train_test_split_time (2016-10-01) and walk forward by
            # the test_as_of_date_frequency (1 month) until we've exhausted the test_duration
            # (3 months), exclusive (see comments in the method for details)
            test_as_of_times = self.calculate_as_of_times(
                as_of_start_limit=train_test_split_time,
                as_of_end_limit=as_of_time_limit,
                data_frequency=convert_str_to_relativedelta(test_as_of_date_frequency),
                forward=True,
            )
            logger.spam(f"test as of times: {test_as_of_times}")
            test_definition = {
                "first_as_of_time": train_test_split_time,
                "last_as_of_time": max(test_as_of_times),
                "matrix_info_end_time": max(test_as_of_times)
                + convert_str_to_relativedelta(test_label_timespan),
                "as_of_times": AsOfTimeList(test_as_of_times),
                "test_label_timespan": test_label_timespan,
                "test_as_of_date_frequency": test_as_of_date_frequency,
                "test_duration": test_duration,
            }
            test_definitions.append(test_definition)
        return test_definitions

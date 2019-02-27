import luigi
from luigi.contrib.simulate import RunAnywayTarget
from luigi.parameter import ParameterVisibility
import time
from datetime import datetime


class TriageExperimentRunner(luigi.WrapperTask):
    modeltype_hyperparams_featuregroups = [
        {
            "model_type": "sklearn.tree.DecisionTreeClassifier",
            "hyperparams": {"maxdepth": 2},
            "feature_groups": ["inspections", "results", "risks"]
        },
        {
            "model_type": "sklearn.ensemble.RandomForestClassifier",
            "hyperparams": {"n_estimator": 100},
            "feature_groups": ["inspections", "results"]

        },
        {
            "model_type": "tf.estimator.DNNClassifier",
            "hyperparams": {"hidden_units": [256, 32], "n_classes": 2, "dropout": 0.1},
            "feature_groups": ["inspections", "results", "risks"]
        }
    ]

    time_config = {
            'train_matrix': {
                'as_of_times': [
                    datetime(2014, 1, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    datetime(2014, 7, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    datetime(2015, 1, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    datetime(2015, 7, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    datetime(2016, 1, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S")
                ],
                'train_info': {
                    'last_as_of_time': datetime(2016, 1, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    'first_as_of_time': datetime(2014, 1, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    'matrix_info_end_time': datetime(2016, 7, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    'max_training_history': '2 years',
                    'training_as_of_date_frequency': '6 months',
                    'training_label_timespan': '6 months'

                }
            },
            'test_matrices': [{
                'as_of_times': [
                    datetime(2016, 7, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                ],
                'test_info': {
                    'last_as_of_time': datetime(2016, 7, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    'first_as_of_time': datetime(2016, 7, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    'matrix_info_end_time': datetime(2017, 1, 1, 0, 0).strftime("%Y-%m-%d %H:%M:%S"),
                    'test_as_of_date_frequency': '1 days',
                    'test_label_timespan': '6 months',
                    'test_duration': '0 days'
                }
            }],
        }

    def requires(self):
        for mg_config in self.modeltype_hyperparams_featuregroups:
            yield ModelGroup(
                    mg_config=mg_config,
                    time_config=self.time_config
                )


class ModelGroup(luigi.Task):
    mg_config = luigi.DictParameter()
    time_config = luigi.DictParameter(significant=False, visibility=ParameterVisibility.HIDDEN)

    def run(self):
        print("Write model group information into db")
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)

    def requires(self):
        for train_as_of_time in self.time_config["train_matrix"]["as_of_times"]:
            yield Evaluation(
                    mg_config=self.mg_config,
                    train_as_of_time=datetime.strptime(train_as_of_time, "%Y-%m-%d %H:%M:%S"),
                    train_info=self.time_config["train_matrix"]["train_info"],
                    test_matrices=self.time_config["test_matrices"]
                )


class Evaluation(luigi.Task):
    mg_config = luigi.DictParameter()
    train_as_of_time = luigi.DateParameter()
    train_info = luigi.DictParameter(visibility=ParameterVisibility.HIDDEN)
    test_matrices = luigi.DictParameter(visibility=ParameterVisibility.HIDDEN)

    @property
    def priority(self):
        if self.mg_config['model_type'] == 'sklearn.ensemble.RandomForestClassifier':
            return 80
        else:
            return 40

    def run(self):
        time.sleep(1)
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)

    def requires(self):
        return [
                Testing(
                    test_matrices=self.test_matrices,
                    train_as_of_time=self.train_as_of_time,
                    mg_config=self.mg_config,
                    train_info=self.train_info
                    )
                ]


class Testing(luigi.Task):
    test_matrices = luigi.DictParameter()
    mg_config = luigi.DictParameter()
    train_as_of_time = luigi.DateParameter()
    train_info = luigi.DictParameter(visibility=ParameterVisibility.HIDDEN)

    def run(self):
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)

    def requires(self):
        for test_matrix in self.test_matrices:
            for test_as_of_time in test_matrix["as_of_times"]:
                yield TestingMatrix(
                        test_as_of_time=datetime.strptime(test_as_of_time, "%Y-%m-%d %H:%M:%S"),
                        feature_groups=self.mg_config["feature_groups"]
                        )
                yield Model(
                        mg_config=self.mg_config,
                        train_as_of_time=self.train_as_of_time,
                        train_info=self.train_info,
                        )


class Model(luigi.Task):
    mg_config = luigi.DictParameter()
    train_as_of_time = luigi.DateParameter()
    train_info = luigi.DictParameter(visibility=ParameterVisibility.HIDDEN)

    def run(self):
        """
        Instantiate the model and start training
        """
        print(f"Training {self.mg_config['model_type']}")
        time.sleep(1)
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)

    def requires(self):
        return TrainingMatrix(
                train_as_of_time=self.train_as_of_time,
                feature_groups=self.mg_config["feature_groups"]
                )


class TrainingMatrix(luigi.Task):
    train_as_of_time = luigi.DateParameter()
    feature_groups = luigi.ListParameter()

    def run(self):
        print(f"Create training matrix for {self.train_as_of_time}")
        time.sleep(1)
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)

    def requires(self):
        return FeatureCreator(feature_groups=self.feature_groups)


class TestingMatrix(luigi.Task):
    test_as_of_time = luigi.DateParameter()
    feature_groups = luigi.ListParameter()

    def run(self):
        print(f"Creating testing matrix for {self.test_as_of_time}")
        time.sleep(1)
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)

    def requires(self):
        return FeatureCreator(feature_groups=self.feature_groups)


class FeatureCreator(luigi.Task):
    feature_groups = luigi.ListParameter()

    def run(self):
        time.sleep(1)
        for feature_group in self.feature_groups:
            yield FeatureGroupFactory.factory(feature_group)
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)


class FeatureGroupFactory(object):
    @staticmethod
    def factory(feature_group):
        for feature_object in FeatureGroupFactory.__subclasses__():
            print(feature_object.name)
            if feature_object.name == feature_group:
                return feature_object()


class FeatureGroupInspectionsCreator(FeatureGroupFactory, luigi.Task):
    name = "inspections"

    def run(self):
        time.sleep(1)
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)

    def requires(self):
        pass


class FeatureGroupResultsCreator(FeatureGroupFactory, luigi.Task):
    name = "results"

    def run(self):
        time.sleep(1)
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)

    def requires(self):
        pass


class FeatureGroupRisksCreator(FeatureGroupFactory, luigi.Task):
    name = "risks"

    def run(self):
        time.sleep(1)
        self.output().done()

    def output(self):
        return RunAnywayTarget(self)

    def requires(self):
        pass


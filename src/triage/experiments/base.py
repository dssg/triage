import logging
from abc import ABC, abstractmethod
import cProfile
import marshal
import time

from descriptors import cachedproperty
from timeout import timeout

from triage.component.architect.label_generators import (
    LabelGenerator,
    LabelGeneratorNoOp,
    DEFAULT_LABEL_NAME,
)

from triage.component.architect.features import (
    FeatureGenerator,
    FeatureDictionaryCreator,
    FeatureGroupCreator,
    FeatureGroupMixer,
)
from triage.component.architect.planner import Planner
from triage.component.architect.builders import MatrixBuilder
from triage.component.architect.cohort_table_generators import (
    CohortTableGenerator,
    CohortTableGeneratorNoOp,
)
from triage.component.timechop import Timechop
from triage.component.results_schema import upgrade_db
from triage.component.catwalk import (
    ModelTrainer,
    ModelEvaluator,
    Predictor,
    IndividualImportanceCalculator,
    ModelGrouper,
    ModelTrainTester
)
from triage.component.catwalk.utils import (
    save_experiment_and_get_hash,
    associate_models_with_experiment,
    associate_matrices_with_experiment,
    missing_matrix_uuids,
    missing_model_hashes,
    filename_friendly_hash
)
from triage.component.catwalk.storage import (
    CSVMatrixStore,
    ModelStorageEngine,
    ProjectStorage,
    MatrixStorageEngine,
)

from triage.experiments import CONFIG_VERSION
from triage.experiments.validate import ExperimentValidator

from triage.database_reflection import table_has_data
from triage.util.conf import dt_from_str


class ExperimentBase(ABC):
    """The base class for all Experiments.

    Subclasses must implement the following four methods:
    process_query_tasks
    process_matrix_build_tasks
    process_train_tasks
    process_model_test_tasks

    Look at singlethreaded.py for reference implementation of each.

    Args:
        config (dict)
        db_engine (triage.util.db.SerializableDbEngine or sqlalchemy.engine.Engine)
        project_path (string)
        replace (bool)
        cleanup_timeout (int)
        materialize_subquery_fromobjs (bool, default True) Whether or not to create and index
            tables for feature "from objects" that are subqueries. Can speed up performance
            when building features for many as-of-dates.
        profile (bool)
    """

    cleanup_timeout = 60  # seconds

    def __init__(
        self,
        config,
        db_engine,
        project_path=None,
        matrix_storage_class=CSVMatrixStore,
        replace=True,
        cleanup=False,
        cleanup_timeout=None,
        materialize_subquery_fromobjs=True,
        features_ignore_cohort=False,
        profile=False,
        save_predictions=True,
    ):
        self._check_config_version(config)
        self.config = config

        self.project_storage = ProjectStorage(project_path)
        self.model_storage_engine = ModelStorageEngine(self.project_storage)
        self.matrix_storage_engine = MatrixStorageEngine(
            self.project_storage, matrix_storage_class
        )
        self.project_path = project_path
        self.replace = replace
        self.save_predictions = save_predictions
        self.db_engine = db_engine
        upgrade_db(db_engine=self.db_engine)

        self.features_schema_name = "features"
        self.materialize_subquery_fromobjs = materialize_subquery_fromobjs
        self.features_ignore_cohort = features_ignore_cohort
        self.experiment_hash = save_experiment_and_get_hash(self.config, self.db_engine)
        self.initialize_components()

        self.cleanup = cleanup
        if self.cleanup:
            logging.info(
                "cleanup is set to True, so intermediate tables (labels and states) "
                "will be removed after matrix creation"
            )
        else:
            logging.info(
                "cleanup is set to False, so intermediate tables (labels and states) "
                "will not be removed after matrix creation"
            )
        self.cleanup_timeout = (
            self.cleanup_timeout if cleanup_timeout is None else cleanup_timeout
        )
        self.profile = profile
        logging.info("Generate profiling stats? (profile option): %s", self.profile)

    def _check_config_version(self, config):
        if "config_version" in config:
            config_version = config["config_version"]
        else:
            logging.warning(
                "config_version key not found in experiment config. "
                "Assuming v1, which may not be correct"
            )
            config_version = "v1"
        if config_version != CONFIG_VERSION:
            raise ValueError(
                "Experiment config '{}' "
                "does not match current version '{}'. "
                "Will not run experiment.".format(config_version, CONFIG_VERSION)
            )

    def initialize_components(self):
        split_config = self.config["temporal_config"]

        self.chopper = Timechop(**split_config)

        cohort_config = self.config.get("cohort_config", {})
        if "query" in cohort_config:
            self.cohort_table_name = "cohort_{}_{}".format(
                cohort_config.get('name', 'default'),
                filename_friendly_hash(cohort_config['query'])
            )
            self.cohort_table_generator = CohortTableGenerator(
                cohort_table_name=self.cohort_table_name,
                db_engine=self.db_engine,
                query=cohort_config["query"],
                replace=self.replace
            )
        else:
            logging.warning(
                "cohort_config missing or unrecognized. Without a cohort, "
                "you will not be able to make matrices, perform feature imputation, "
                "or save time by only computing features for that cohort."
            )
            self.features_ignore_cohort = True
            self.cohort_table_name = "cohort_{}".format(self.experiment_hash)
            self.cohort_table_generator = CohortTableGeneratorNoOp()

        if "label_config" in self.config:
            label_config = self.config["label_config"]
            self.labels_table_name = "labels_{}_{}".format(
                label_config.get('name', 'default'),
                filename_friendly_hash(label_config['query'])
            )
            self.label_generator = LabelGenerator(
                label_name=label_config.get("name", None),
                query=label_config["query"],
                replace=self.replace,
                db_engine=self.db_engine,
            )
        else:
            self.labels_table_name = "labels_{}".format(self.experiment_hash)
            self.label_generator = LabelGeneratorNoOp()
            logging.warning(
                "label_config missing or unrecognized. Without labels, "
                "you will not be able to make matrices."
            )

        self.feature_dictionary_creator = FeatureDictionaryCreator(
            features_schema_name=self.features_schema_name, db_engine=self.db_engine
        )

        self.feature_generator = FeatureGenerator(
            features_schema_name=self.features_schema_name,
            replace=self.replace,
            db_engine=self.db_engine,
            feature_start_time=split_config["feature_start_time"],
            materialize_subquery_fromobjs=self.materialize_subquery_fromobjs,
            features_ignore_cohort=self.features_ignore_cohort
        )

        self.feature_group_creator = FeatureGroupCreator(
            self.config.get("feature_group_definition", {"all": [True]})
        )

        self.feature_group_mixer = FeatureGroupMixer(
            self.config.get("feature_group_strategies", ["all"])
        )

        self.planner = Planner(
            feature_start_time=dt_from_str(split_config["feature_start_time"]),
            label_names=[
                self.config.get("label_config", {}).get("name", DEFAULT_LABEL_NAME)
            ],
            label_types=["binary"],
            cohort_names=[self.config.get("cohort_config", {}).get("name", None)],
            user_metadata=self.config.get("user_metadata", {}),
        )

        self.matrix_builder = MatrixBuilder(
            db_config={
                "features_schema_name": self.features_schema_name,
                "labels_schema_name": "public",
                "labels_table_name": self.labels_table_name,
                "cohort_table_name": self.cohort_table_name,
            },
            matrix_storage_engine=self.matrix_storage_engine,
            experiment_hash=self.experiment_hash,
            include_missing_labels_in_train_as=self.config.get("label_config", {}).get(
                "include_missing_labels_in_train_as", None
            ),
            engine=self.db_engine,
            replace=self.replace,
        )

        self.trainer = ModelTrainer(
            experiment_hash=self.experiment_hash,
            model_storage_engine=self.model_storage_engine,
            model_grouper=ModelGrouper(self.config.get("model_group_keys", [])),
            db_engine=self.db_engine,
            replace=self.replace,
        )

        self.predictor = Predictor(
            db_engine=self.db_engine,
            model_storage_engine=self.model_storage_engine,
            save_predictions=self.save_predictions,
            replace=self.replace,
        )

        self.individual_importance_calculator = IndividualImportanceCalculator(
            db_engine=self.db_engine,
            n_ranks=self.config.get("individual_importance", {}).get("n_ranks", 5),
            methods=self.config.get("individual_importance", {}).get("methods", ["uniform"]),
            replace=self.replace,
        )

        self.evaluator = ModelEvaluator(
            db_engine=self.db_engine,
            sort_seed=self.config.get("scoring", {}).get("sort_seed", None),
            testing_metric_groups=self.config.get("scoring", {}).get("testing_metric_groups", []),
            training_metric_groups=self.config.get("scoring", {}).get("training_metric_groups", []),
        )

        self.model_train_tester = ModelTrainTester(
            matrix_storage_engine=self.matrix_storage_engine,
            model_evaluator=self.evaluator,
            model_trainer=self.trainer,
            individual_importance_calculator=self.individual_importance_calculator,
            predictor=self.predictor
        )

    @cachedproperty
    def split_definitions(self):
        """Temporal splits based on the experiment's configuration

        Returns: (dict) temporal splits

        Example:
        ```
        {
            'feature_start_time': {datetime},
            'feature_end_time': {datetime},
            'label_start_time': {datetime},
            'label_end_time': {datetime},
            'train_matrix': {
                'first_as_of_time': {datetime},
                'last_as_of_time': {datetime},
                'matrix_info_end_time': {datetime},
                'training_label_timespan': {str},
                'training_as_of_date_frequency': {str},
                'max_training_history': {str},
                'as_of_times': [list of {datetime}s]
            },
            'test_matrices': [list of matrix defs similar to train_matrix]
        }
        ```

        (When updating/setting split definitions, matrices should have
        UUIDs.)

        """
        split_definitions = self.chopper.chop_time()
        logging.info("Computed and stored split definitions: %s", split_definitions)
        logging.info("\n----TIME SPLIT SUMMARY----\n")
        logging.info("Number of time splits: {}".format(len(split_definitions)))
        for split_index, split in enumerate(split_definitions):
            train_times = split["train_matrix"]["as_of_times"]
            test_times = [
                as_of_time
                for test_matrix in split["test_matrices"]
                for as_of_time in test_matrix["as_of_times"]
            ]
            logging.info(
                """Split index {}:
            Training as_of_time_range: {} to {} ({} total)
            Testing as_of_time range: {} to {} ({} total)\n\n""".format(
                    split_index,
                    min(train_times),
                    max(train_times),
                    len(train_times),
                    min(test_times),
                    max(test_times),
                    len(test_times),
                )
            )

        return split_definitions

    @cachedproperty
    def all_as_of_times(self):
        """All 'as of times' in experiment config

        Used for label and feature generation.

        Returns: (list) of datetimes

        """
        all_as_of_times = []
        for split in self.split_definitions:
            all_as_of_times.extend(split["train_matrix"]["as_of_times"])
            logging.debug(
                "Adding as_of_times from train matrix: %s",
                split["train_matrix"]["as_of_times"],
            )
            for test_matrix in split["test_matrices"]:
                logging.debug(
                    "Adding as_of_times from test matrix: %s",
                    test_matrix["as_of_times"],
                )
                all_as_of_times.extend(test_matrix["as_of_times"])

        logging.info(
            "Computed %s total as_of_times for label and feature generation",
            len(all_as_of_times),
        )
        distinct_as_of_times = list(set(all_as_of_times))
        logging.info(
            "Computed %s distinct as_of_times for label and feature generation",
            len(distinct_as_of_times),
        )
        logging.info(
            "You can view all as_of_times by inspecting `.all_as_of_times` on this Experiment"
        )
        return distinct_as_of_times

    @cachedproperty
    def collate_aggregations(self):
        """Collation of ``Aggregation`` objects used by this experiment.

        Returns: (list) of ``collate.Aggregation`` objects

        """
        logging.info("Creating collate aggregations")
        if "feature_aggregations" not in self.config:
            logging.warning("No feature_aggregation config is available")
            return []
        return self.feature_generator.aggregations(
            feature_aggregation_config=self.config["feature_aggregations"],
            feature_dates=self.all_as_of_times,
            state_table=self.cohort_table_name,
        )

    @cachedproperty
    def feature_aggregation_table_tasks(self):
        """All feature table query tasks specified by this
        ``Experiment``.

        Returns: (dict) keys are group table names, values are
            themselves dicts, each with keys for different stages of
            table creation (prepare, inserts, finalize) and with values
            being lists of SQL commands

        """
        logging.info(
            "Calculating feature tasks for %s as_of_times", len(self.all_as_of_times)
        )
        return self.feature_generator.generate_all_table_tasks(
            self.collate_aggregations, task_type="aggregation"
        )

    @cachedproperty
    def feature_imputation_table_tasks(self):
        """All feature imputation query tasks specified by this
        ``Experiment``.

        Returns: (dict) keys are group table names, values are
            themselves dicts, each with keys for different stages of
            table creation (prepare, inserts, finalize) and with values
            being lists of SQL commands

        """
        logging.info(
            "Calculating feature tasks for %s as_of_times", len(self.all_as_of_times)
        )
        return self.feature_generator.generate_all_table_tasks(
            self.collate_aggregations, task_type="imputation"
        )

    @cachedproperty
    def master_feature_dictionary(self):
        """All possible features found in the database. Not all features
        will necessarily end up in matrices

        Returns: (list) of dicts, keys being feature table names and
        values being lists of feature names

        """
        result = self.feature_dictionary_creator.feature_dictionary(
            feature_table_names=self.feature_imputation_table_tasks.keys(),
            index_column_lookup=self.feature_generator.index_column_lookup(
                self.collate_aggregations
            ),
        )
        logging.info("Computed master feature dictionary: %s", result)
        return result

    @property
    def feature_dicts(self):
        """Feature dictionaries, representing the feature tables and
        columns configured in this experiment after computing feature
        groups.

        Returns: (list) of dicts, keys being feature table names and
        values being lists of feature names

        """
        return self.feature_group_mixer.generate(
            self.feature_group_creator.subsets(self.master_feature_dictionary)
        )

    @cachedproperty
    def matrix_build_tasks(self):
        """Tasks for all matrices that need to be built as a part of
        this Experiment.

        Each task contains arguments understood by
        ``Architect.build_matrix``.

        Returns: (list) of dicts

        """
        if not table_has_data(self.cohort_table_name, self.db_engine):
            logging.warning("cohort table is not populated, cannot build any matrices")
            return {}
        if not table_has_data(self.labels_table_name, self.db_engine):
            logging.warning("labels table is not populated, cannot build any matrices")
            return {}
        (updated_split_definitions, matrix_build_tasks) = self.planner.generate_plans(
            self.split_definitions, self.feature_dicts
        )
        self.full_matrix_definitions = updated_split_definitions
        return matrix_build_tasks

    @cachedproperty
    def full_matrix_definitions(self):
        """Full matrix definitions

        Returns: (list) temporal and feature information for each matrix

        """
        (updated_split_definitions, matrix_build_tasks) = self.planner.generate_plans(
            self.split_definitions, self.feature_dicts
        )
        self.matrix_build_tasks = matrix_build_tasks
        return updated_split_definitions

    @property
    def all_label_timespans(self):
        """All train and test label timespans

        Returns: (list) label timespans, in string form as they appeared in the experiment config

        """
        return list(
            set(
                self.config["temporal_config"]["training_label_timespans"]
                + self.config["temporal_config"]["test_label_timespans"]
            )
        )

    def generate_labels(self):
        """Generate labels based on experiment configuration

        Results are stored in the database, not returned
        """
        self.label_generator.generate_all_labels(
            self.labels_table_name, self.all_as_of_times, self.all_label_timespans
        )

    def generate_cohort(self):
        self.cohort_table_generator.generate_cohort_table(
            as_of_dates=self.all_as_of_times
        )

    def log_split(self, split_num, split):
        logging.info(
            "Starting train/test for %s out of %s: train range: %s to %s",
            split_num + 1,
            len(self.full_matrix_definitions),
            split["train_matrix"]["first_as_of_time"],
            split["train_matrix"]["matrix_info_end_time"],
        )

    @abstractmethod
    def process_train_test_tasks(self, train_tasks):
        pass

    @abstractmethod
    def process_query_tasks(self, query_tasks):
        pass

    @abstractmethod
    def process_matrix_build_tasks(self, matrix_build_tasks):
        pass

    def generate_preimputation_features(self):
        self.process_query_tasks(self.feature_aggregation_table_tasks)
        logging.info(
            "Finished running preimputation feature queries. The final results are in tables: %s",
            ",".join(agg.get_table_name() for agg in self.collate_aggregations),
        )

    def impute_missing_features(self):
        self.process_query_tasks(self.feature_imputation_table_tasks)
        logging.info(
            "Finished running postimputation feature queries. The final results are in tables: %s",
            ",".join(
                agg.get_table_name(imputed=True) for agg in self.collate_aggregations
            ),
        )

    def build_matrices(self):
        associate_matrices_with_experiment(
            self.experiment_hash,
            self.matrix_build_tasks.keys(),
            self.db_engine
        )
        self.process_matrix_build_tasks(self.matrix_build_tasks)

    def generate_matrices(self):
        logging.info("Creating cohort")
        self.generate_cohort()
        logging.info("Creating labels")
        self.generate_labels()
        logging.info("Creating feature aggregation tables")
        self.generate_preimputation_features()
        logging.info("Creating feature imputation tables")
        self.impute_missing_features()
        logging.info("Building all matrices")
        self.build_matrices()

    def _all_train_test_tasks(self):
        if "grid_config" not in self.config:
            logging.warning(
                "No grid_config was passed in the experiment config. No models will be trained"
            )
            return

        train_test_tasks = []
        for split_num, split in enumerate(self.full_matrix_definitions):
            self.log_split(split_num, split)
            for task in self.model_train_tester.generate_tasks(
                split=split,
                grid_config=self.config.get('grid_config'),
                model_comment=self.config.get('model_comment', None)
            ):
                train_test_tasks.append(task)
        return train_test_tasks

    def train_and_test_models(self):
        tasks = self._all_train_test_tasks()
        if not tasks:
            logging.warning("No train/test tasks found, so no training to do")
            return

        logging.info("%s train/test tasks found. Beginning training.", len(tasks))
        associate_models_with_experiment(
            self.experiment_hash,
            set(task['train_kwargs']['model_hash'] for task in tasks),
            self.db_engine
        )
        self.process_train_test_tasks(tasks)

    def validate(self, strict=True):
        ExperimentValidator(self.db_engine, strict=strict).run(self.config)

    def _run(self):
        try:
            logging.info("Generating matrices")
            self.generate_matrices()
        finally:
            if self.cleanup:
                self.clean_up_tables()

        self.train_and_test_models()
        logging.info("Experiment complete")
        self._log_end_of_run_report()

    def _log_end_of_run_report(self):
        missing_models = missing_model_hashes(self.experiment_hash, self.db_engine)
        if len(missing_models) > 0:
            logging.info("Found %s missing model hashes."
                         "This means that they were supposed to either be trained or reused"
                         "by this experiment but are not present in the models table."
                         "Inspect the logs for any training errors. Full list: %s",
                         len(missing_models),
                         missing_models
                         )
        else:
            logging.info("All models that were supposed to be trained were trained. Awesome!")

        missing_matrices = missing_matrix_uuids(self.experiment_hash, self.db_engine)
        if len(missing_matrices) > 0:
            logging.info("Found %s missing matrix uuids."
                         "This means that they were supposed to either be build or reused"
                         "by this experiment but are not present in the matrices table."
                         "Inspect the logs for any matrix building errors. Full list: %s",
                         len(missing_matrices),
                         missing_matrices
                         )
        else:
            logging.info("All matrices that were supposed to be build were built. Awesome!")

    def clean_up_tables(self):
        logging.info("Cleaning up state and labels tables")
        with timeout(self.cleanup_timeout):
            self.cohort_table_generator.clean_up()
            self.label_generator.clean_up(self.labels_table_name)

    def _run_profile(self):
        cp = cProfile.Profile()
        cp.runcall(self._run)
        store = self.project_storage.get_store(
            ["profiling_stats"],
            f"{int(time.time())}.profile"
        )
        with store.open('wb') as fd:
            cp.create_stats()
            marshal.dump(cp.stats, fd)
            logging.info("Profiling stats of this Triage run calculated and written to %s"
                         "in cProfile format.",
                         store)

    def run(self):
        try:
            if self.profile:
                self._run_profile()
            else:
                self._run()
        except Exception:
            logging.exception("Run interrupted by uncaught exception")
            raise

    __call__ = run

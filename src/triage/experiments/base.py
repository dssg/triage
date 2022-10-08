import verboselogs, logging

logger = verboselogs.VerboseLogger(__name__)

from abc import ABC, abstractmethod
import cProfile
import marshal
import os
import random
import time
import itertools

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
from triage.component.architect.entity_date_table_generators import (
    EntityDateTableGenerator,
    CohortTableGeneratorNoOp,
)
from triage.component.timechop import Timechop
from triage.component import results_schema
from triage.component.catwalk import (
    ModelTrainer,
    ModelEvaluator,
    Predictor,
    IndividualImportanceCalculator,
    IndividualImportanceCalculatorNoOp,
    ModelGrouper,
    ModelTrainTester,
    Subsetter,
    SubsetterNoOp,
)
from triage.component.catwalk.protected_groups_generators import (
    ProtectedGroupsGenerator,
    ProtectedGroupsGeneratorNoOp,
)

from triage.component.catwalk.utils import (
    save_experiment_and_get_hash,
    associate_models_with_experiment,
    associate_matrices_with_experiment,
    missing_matrix_uuids,
    missing_model_hashes,
    filename_friendly_hash,
)
from triage.component.catwalk.storage import (
    CSVMatrixStore,
    ModelStorageEngine,
    ProjectStorage,
    MatrixStorageEngine,
)

from triage.experiments import CONFIG_VERSION
from triage.experiments.validate import ExperimentValidator
from triage.tracking import (
    initialize_tracking_and_get_run_id,
    experiment_entrypoint,
    record_cohort_table_name,
    record_labels_table_name,
    record_bias_hash,
    record_matrix_building_started,
    record_model_building_started,
)

from triage.experiments.defaults import (
    fill_timechop_config_missing,
    fill_feature_group_definition,
    fill_model_grid_presets,
)

from triage.database_reflection import table_has_data
from triage.util.conf import dt_from_str, parse_from_obj, load_query_if_needed
from triage.util.db import get_for_update
from triage.util.introspection import bind_kwargs, classpath


class ExperimentBase(ABC):
    """The base class for all Experiments.

    Subclasses must implement the following four methods:
    process_query_tasks
    process_matrix_build_tasks
    process_subset_tasks
    process_train_test_batches

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
        additional_bigtrain_classnames (list) Any additional class names to perform in the second batch
            of training, which focuses on large modeling algorithms that tend to run with less parallelization
            as there is generally parallelization and high memory requirements built into the algorithm.
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
        additional_bigtrain_classnames=None,
        profile=False,
        save_predictions=True,
        skip_validation=False,
        partial_run=False,
    ):
        # For a partial run, skip validation and avoid cleaning up
        # we'll also skip filling default config values below
        if partial_run:
            cleanup = False
            skip_validation = True

        experiment_kwargs = bind_kwargs(
            self.__class__,
            **{
                key: value
                for (key, value) in locals().items()
                if key not in {"db_engine", "config", "self"}
            },
        )

        self._check_config_version(config)
        self.config = config

        if self.config.get("cohort_config") is not None:
            self.config["cohort_config"] = load_query_if_needed(
                self.config["cohort_config"]
            )
        if self.config.get("label_config") is not None:
            self.config["label_config"] = load_query_if_needed(
                self.config["label_config"]
            )

        self.project_storage = ProjectStorage(project_path)
        self.model_storage_engine = ModelStorageEngine(self.project_storage)
        self.matrix_storage_engine = MatrixStorageEngine(
            self.project_storage, matrix_storage_class
        )
        self.project_path = project_path
        logger.verbose(
            f"Matrices and trained models will be saved in {self.project_path}"
        )
        self.replace = replace
        if self.replace:
            logger.notice(
                f"Replace flag is set to true. Matrices, models, "
                "evaluations and predictions (if they exist) will be replaced"
            )

        self.save_predictions = save_predictions
        if not self.save_predictions:
            logger.notice(
                f"Save predictions flag is set to false. "
                "Individual predictions won't be stored in the predictions "
                "table. This will decrease both the running time "
                "of an experiment and also decrease the space needed in the db"
            )

        self.skip_validation = skip_validation
        if self.skip_validation:
            logger.notice(
                f"Warning: Skip validation flag is set to true. "
                "The experiment config file specified won't be validated. "
                "This will reduce (a little) the running time of the experiment, "
                "but has some potential risks, e.g. the experiment could fail"
                "after some time due to some misconfiguration. Proceed with care."
            )

        self.db_engine = db_engine
        results_schema.upgrade_if_clean(dburl=self.db_engine.url)

        self.features_schema_name = "features"

        self.materialize_subquery_fromobjs = materialize_subquery_fromobjs
        if not self.materialize_subquery_fromobjs:
            logger.notice(
                "Materialize from_objs is set to false. "
                "The from_objs will be calculated on the fly every time."
            )

        self.features_ignore_cohort = features_ignore_cohort
        if self.features_ignore_cohort:
            logger.notice(
                "Features will be calculated for all the entities "
                "(i.e. ignoring cohort) this setting will have the effect "
                "that more db space will be used, but potentially could save "
                "time if you are running several similar experiments with "
                "different cohorts."
            )

        self.additional_bigtrain_classnames = additional_bigtrain_classnames
        # only fill default values for full runs
        if not partial_run:
            ## Defaults to sane values
            self.config["temporal_config"] = fill_timechop_config_missing(
                self.config, self.db_engine
            )
            ## Defaults to all the feature_aggregation's prefixes
            self.config["feature_group_definition"] = fill_feature_group_definition(
                self.config
            )

        grid_config = fill_model_grid_presets(self.config)
        self.config.pop("model_grid_preset", None)
        if grid_config is not None:
            self.config["grid_config"] = grid_config

        if not self.config.get("random_seed", None):
            logger.notice(
                "Random seed not specified. A random seed will be provided. "
                "This could have interesting side effects, "
                "e.g. new models per model group are trained, "
                "tested and evaluated everytime that you run this experiment configuration"
            )

        self.random_seed = self.config.pop("random_seed", random.randint(1, 1e7))

        logger.verbose(
            f"Using random seed [{self.random_seed}] for running the experiment"
        )
        random.seed(self.random_seed)

        ###################### RUBICON ######################

        self.experiment_hash = save_experiment_and_get_hash(self.config, self.db_engine)
        logger.debug(f"Experiment hash [{self.experiment_hash}] assigned")
        self.run_id = initialize_tracking_and_get_run_id(
            self.experiment_hash,
            experiment_class_path=classpath(self.__class__),
            random_seed=self.random_seed,
            experiment_kwargs=experiment_kwargs,
            db_engine=self.db_engine,
        )
        logger.debug(f"Experiment run id [{self.run_id}] assigned")

        self.initialize_components()

        self.cleanup = cleanup
        if self.cleanup:
            logger.notice(
                "Cleanup is set to true, so intermediate tables (labels and cohort) "
                "will be removed after matrix creation and subset tables will be "
                "removed after model training and testing"
            )

        self.cleanup_timeout = (
            self.cleanup_timeout if cleanup_timeout is None else cleanup_timeout
        )

        self.profile = profile
        if self.profile:
            logger.spam("Profiling will be stored using cProfile")

    def _check_config_version(self, config):
        if "config_version" in config:
            config_version = config["config_version"]
        else:
            raise ValueError("config_version key not found in experiment config. ")
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

        if "label_config" in self.config:
            label_config = self.config["label_config"]
            self.labels_table_name = "labels_{}_{}".format(
                label_config.get("name", "default"),
                filename_friendly_hash(label_config["query"]),
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
            logger.warning(
                "label_config missing or unrecognized. Without labels, "
                "you will not be able to make matrices."
            )
        record_labels_table_name(self.run_id, self.db_engine, self.labels_table_name)

        cohort_config = self.config.get("cohort_config", {})
        self.cohort_table_generator = None
        if "query" in cohort_config:
            self.cohort_hash = filename_friendly_hash(
                self.config["cohort_config"]["query"]
            )
        elif "query" in self.config.get("label_config", {}):
            logger.info(
                "cohort_config missing or unrecognized, but labels are configured. Labels will be used as the cohort."
            )
            self.cohort_hash = filename_friendly_hash(
                self.config["label_config"]["query"]
            )
        else:
            self.features_ignore_cohort = True
            self.cohort_hash = None
            self.cohort_table_name = "cohort_{}".format(self.experiment_hash)
            self.cohort_table_generator = CohortTableGeneratorNoOp()

        if not self.cohort_table_generator:
            self.cohort_table_name = "cohort_{}_{}".format(
                cohort_config.get("name", "default"), self.cohort_hash
            )
            self.cohort_table_generator = EntityDateTableGenerator(
                entity_date_table_name=self.cohort_table_name,
                db_engine=self.db_engine,
                query=cohort_config.get("query", None),
                labels_table_name=self.labels_table_name,
                replace=self.replace,
            )

        record_cohort_table_name(self.run_id, self.db_engine, self.cohort_table_name)

        if "bias_audit_config" in self.config:
            bias_config = self.config["bias_audit_config"]
            self.bias_hash = filename_friendly_hash(bias_config)
            self.protected_groups_table_name = f"protected_groups_{self.bias_hash}"
            self.protected_groups_generator = ProtectedGroupsGenerator(
                db_engine=self.db_engine,
                from_obj=parse_from_obj(bias_config, "bias_from_obj"),
                attribute_columns=bias_config.get("attribute_columns", None),
                entity_id_column=bias_config.get("entity_id_column", None),
                knowledge_date_column=bias_config.get("knowledge_date_column", None),
                protected_groups_table_name=self.protected_groups_table_name,
                replace=self.replace,
            )
            record_bias_hash(self.run_id, self.db_engine, self.bias_hash)
        else:
            self.protected_groups_generator = ProtectedGroupsGeneratorNoOp()
            logger.notice(
                "bias_audit_config missing in the configuration file or unrecognized. "
                "Without protected groups, you will not be able to audit your models for bias and fairness."
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
            features_ignore_cohort=self.features_ignore_cohort,
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
            run_id=self.run_id,
        )

        self.subsets = self.config.get("scoring", {}).get("subsets", [])
        if self.subsets:
            self.subsetter = Subsetter(
                db_engine=self.db_engine,
                replace=self.replace,
                as_of_times=self.all_as_of_times,
            )
        else:
            self.subsetter = SubsetterNoOp()
            logger.notice(
                "scoring.subsets missing in the configuration file or unrecognized. No subsets will be generated"
            )

        self.trainer = ModelTrainer(
            experiment_hash=self.experiment_hash,
            model_storage_engine=self.model_storage_engine,
            model_grouper=ModelGrouper(self.config.get("model_group_keys", [])),
            db_engine=self.db_engine,
            replace=self.replace,
            run_id=self.run_id,
        )

        self.predictor = Predictor(
            db_engine=self.db_engine,
            model_storage_engine=self.model_storage_engine,
            save_predictions=self.save_predictions,
            replace=self.replace,
            rank_order=self.config.get("prediction", {}).get(
                "rank_tiebreaker", "worst"
            ),
        )

        if "individual_importance" in self.config:
            self.individual_importance_calculator = IndividualImportanceCalculator(
                db_engine=self.db_engine,
                n_ranks=self.config.get("individual_importance", {}).get("n_ranks", 5),
                methods=self.config.get("individual_importance", {}).get(
                    "methods", ["uniform"]
                ),
                replace=self.replace,
            )
        else:
            self.individual_importance_calculator = IndividualImportanceCalculatorNoOp()
            logger.notice(
                "individual_importance missing in the configuration file or unrecognized, "
                "you will not be able to do analysis on individual feature importances."
            )

        self.evaluator = ModelEvaluator(
            db_engine=self.db_engine,
            testing_metric_groups=self.config.get("scoring", {}).get(
                "testing_metric_groups", []
            ),
            training_metric_groups=self.config.get("scoring", {}).get(
                "training_metric_groups", []
            ),
            bias_config=self.config.get("bias_audit_config", {}),
        )

        self.model_train_tester = ModelTrainTester(
            matrix_storage_engine=self.matrix_storage_engine,
            model_evaluator=self.evaluator,
            model_trainer=self.trainer,
            individual_importance_calculator=self.individual_importance_calculator,
            predictor=self.predictor,
            subsets=self.subsets,
            protected_groups_generator=self.protected_groups_generator,
            cohort_hash=self.cohort_hash,
            replace=self.replace,
            additional_bigtrain_classnames=self.additional_bigtrain_classnames,
        )

    def get_for_update(self):
        return get_for_update(
            self.db_engine, results_schema.Experiment, self.experiment_hash
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
        logger.verbose(f"Computed and stored temporal split definitions")
        logger.debug(f"Temporal split definitions: {split_definitions}")
        logger.spam("\n----TIME SPLIT SUMMARY----\n")
        logger.spam("Number of time splits: {len(split_definitions)}")
        for split_index, split in enumerate(split_definitions):
            train_times = split["train_matrix"]["as_of_times"]
            test_times = [
                as_of_time
                for test_matrix in split["test_matrices"]
                for as_of_time in test_matrix["as_of_times"]
            ]
            logger.spam(
                f"""Split index {split_index}:"""
                f"""Training as_of_time_range: {min(train_times)} to {max(train_times)} ({len(train_times)} total)"""
                f"""Testing as_of_time range: {min(test_times)} to {max(test_times)} ({len(test_times)} total)\n\n"""
            )

        with self.get_for_update() as experiment:
            experiment.time_splits = len(split_definitions)
        return split_definitions

    @cachedproperty
    def all_as_of_times(self):
        """All 'as of times' in experiment config

        Used for label and feature generation.

        Returns: (list) of datetimes

        """
        logger.spam("Calculating all the as_of_times")
        all_as_of_times = []
        for split in self.split_definitions:
            all_as_of_times.extend(split["train_matrix"]["as_of_times"])
            logger.spam(
                f'Adding as_of_times from train matrix: {split["train_matrix"]["as_of_times"]}'
            )
            for test_matrix in split["test_matrices"]:
                logger.spam(
                    f'Adding as_of_times from test matrix: {test_matrix["as_of_times"]}',
                )
                all_as_of_times.extend(test_matrix["as_of_times"])

        logger.spam(
            f"Computed {len(all_as_of_times)} total as_of_times for label and feature generation",
        )
        distinct_as_of_times = list(set(all_as_of_times))
        logger.debug(
            f"Computed {len(distinct_as_of_times)} distinct as_of_times for label and feature generation",
        )
        logger.spam(
            "You can view all as_of_times by inspecting `.all_as_of_times` on this Experiment"
        )
        with self.get_for_update() as experiment:
            experiment.as_of_times = len(distinct_as_of_times)
        return distinct_as_of_times

    @cachedproperty
    def collate_aggregations(self):
        """Collation of ``Aggregation`` objects used by this experiment.

        Returns: (list) of ``collate.Aggregation`` objects

        """
        logger.info("Creating collate aggregations")
        if "feature_aggregations" not in self.config:
            logger.warning("No feature_aggregation config is available")
            return []
        aggregations = self.feature_generator.aggregations(
            feature_aggregation_config=self.config["feature_aggregations"],
            feature_dates=self.all_as_of_times,
            state_table=self.cohort_table_name,
        )
        with self.get_for_update() as experiment:
            experiment.feature_blocks = len(aggregations)
        return aggregations

    @cachedproperty
    def feature_aggregation_table_tasks(self):
        """All feature table query tasks specified by this
        ``Experiment``.

        Returns: (dict) keys are group table names, values are
            themselves dicts, each with keys for different stages of
            table creation (prepare, inserts, finalize) and with values
            being lists of SQL commands

        """
        logger.spam(
            f"Calculating feature aggregation tasks for {len(self.all_as_of_times)} as_of_times"
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
        logger.spam(
            f"Calculating feature imputation tasks for {len(self.all_as_of_times)} as_of_times"
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
        logger.debug(f"Computed master feature dictionary: {result}")
        with self.get_for_update() as experiment:
            experiment.total_features = sum(
                1 for _feature in itertools.chain.from_iterable(result.values())
            )
        return result

    @cachedproperty
    def feature_dicts(self):
        """Feature dictionaries, representing the feature tables and
        columns configured in this experiment after computing feature
        groups.

        Returns: (list) of dicts, keys being feature table names and
        values being lists of feature names

        """
        if not self.master_feature_dictionary:
            logger.warning(
                "No features have been created. Either there is no feature configuration"
                "or there was some problem processing them."
            )
            return []
        combinations = self.feature_group_mixer.generate(
            self.feature_group_creator.subsets(self.master_feature_dictionary)
        )
        with self.get_for_update() as experiment:
            experiment.feature_group_combinations = len(combinations)
        return combinations

    @cachedproperty
    def matrix_build_tasks(self):
        """Tasks for all matrices that need to be built as a part of
        this Experiment.

        Each task contains arguments understood by
        ``Architect.build_matrix``.

        Returns: (list) of dicts

        """
        if not table_has_data(self.cohort_table_name, self.db_engine):
            logger.warning("cohort table is not populated, cannot build any matrices")
            return {}
        if not table_has_data(self.labels_table_name, self.db_engine):
            logger.warning("labels table is not populated, cannot build any matrices")
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

    @cachedproperty
    def subset_tasks(self):
        return self.subsetter.generate_tasks(self.subsets)

    @experiment_entrypoint
    def generate_labels(self):
        """Generate labels based on experiment configuration

        Results are stored in the database, not returned
        """
        logger.info("Setting up labels")
        self.label_generator.generate_all_labels(
            self.labels_table_name, self.all_as_of_times, self.all_label_timespans
        )
        logger.success(
            f"Labels set up in the table {self.labels_table_name} successfully "
        )

    @experiment_entrypoint
    def generate_cohort(self):
        logger.info("Setting up cohort")
        self.cohort_table_generator.generate_entity_date_table(
            as_of_dates=self.all_as_of_times
        )
        logger.success(
            f"Cohort set up in the table {self.cohort_table_name} successfully"
        )

    @experiment_entrypoint
    def generate_protected_groups(self):
        """Generate protected groups table based on experiment configuration

        Results are stored in the database, not returned
        """
        self.protected_groups_generator.generate_all_dates(
            self.all_as_of_times, self.cohort_table_name, self.cohort_hash
        )

    def log_split(self, split_num, split):
        logger.info(
            "Starting train/test for %s out of %s: train range: %s to %s",
            split_num + 1,
            len(self.full_matrix_definitions),
            split["train_matrix"]["first_as_of_time"],
            split["train_matrix"]["matrix_info_end_time"],
        )

    @abstractmethod
    def process_subset_tasks(self, subset_tasks):
        pass

    @abstractmethod
    def process_train_test_batches(self, train_test_batches):
        pass

    @abstractmethod
    def process_query_tasks(self, query_tasks):
        pass

    @abstractmethod
    def process_matrix_build_tasks(self, matrix_build_tasks):
        pass

    @experiment_entrypoint
    def generate_preimputation_features(self):
        logger.info("Creating features tables (before imputation) ")
        self.process_query_tasks(self.feature_aggregation_table_tasks)
        logger.success(
            f"Features (before imputation) were stored in the tables "
            f"{','.join(agg.get_table_name() for agg in self.collate_aggregations)} "
            f"successfully"
        )

    @experiment_entrypoint
    def impute_missing_features(self):
        logger.info("Imputing missing values in features")
        self.process_query_tasks(self.feature_imputation_table_tasks)
        logger.success(
            f"Imputed features were stored in the tables "
            f"{','.join(agg.get_table_name(imputed=True) for agg in self.collate_aggregations)} "
            f"successfully"
        )

    def build_matrices(self):
        associate_matrices_with_experiment(
            self.experiment_hash, self.matrix_build_tasks.keys(), self.db_engine
        )
        logger.info("Building matrices")
        logger.verbose(
            f"It is necessary to build {len(self.matrix_build_tasks.keys())} matrices"
        )
        with self.get_for_update() as experiment:
            experiment.matrices_needed = len(self.matrix_build_tasks.keys())
        record_matrix_building_started(self.run_id, self.db_engine)
        self.process_matrix_build_tasks(self.matrix_build_tasks)
        logger.success(
            f"Matrices were stored in {self.project_path}/matrices successfully"
        )

    @experiment_entrypoint
    def generate_matrices(self):
        self.all_as_of_times  # Forcing the calculation of all the as of times, so the logging makes more sense
        self.generate_labels()
        self.generate_cohort()
        self.generate_preimputation_features()
        self.impute_missing_features()
        self.build_matrices()

    @experiment_entrypoint
    def generate_subsets(self):
        self.process_subset_tasks(self.subset_tasks)

    def _all_train_test_batches(self):
        """A batch is a model_group to be train, test and evaluated"""
        if "grid_config" not in self.config:
            logger.warning(
                "No grid_config was passed in the experiment config. No models will be trained"
            )
            return

        return self.model_train_tester.generate_task_batches(
            splits=self.full_matrix_definitions,
            grid_config=self.config.get("grid_config"),
            model_comment=self.config.get("model_comment", None),
        )

    @experiment_entrypoint
    def train_and_test_models(self):
        batches = self._all_train_test_batches()
        if not batches:
            logger.notice("No train/test tasks found, so no training to do")
            return

        with self.get_for_update() as experiment:
            experiment.grid_size = sum(
                1
                for _param in self.trainer.flattened_grid_config(
                    self.config.get("grid_config")
                )
            )
            logger.info(
                f"{experiment.grid_size} models groups will be trained, tested and evaluated"
            )

        logger.info(f"Training, testing and evaluating models")
        logger.verbose(f"{len(batches)} train/test tasks found.")
        model_hashes = set(
            task["train_kwargs"]["model_hash"]
            for batch in batches
            for task in batch.tasks
        )
        associate_models_with_experiment(
            self.experiment_hash, model_hashes, self.db_engine
        )
        with self.get_for_update() as experiment:
            experiment.models_needed = len(model_hashes)
        record_model_building_started(self.run_id, self.db_engine)
        self.process_train_test_batches(batches)
        logger.success("Training, testing and evaluating models completed")

    def validate(self, strict=True):
        ExperimentValidator(self.db_engine, strict=strict).run(self.config)

    def _run(self):
        if not self.skip_validation:
            self.validate()

        try:
            self.generate_matrices()
            self.generate_subsets()
            self.generate_protected_groups()
            self.train_and_test_models()
            self._log_end_of_run_report()
        except Exception:
            logger.error("Uh oh... Houston we have a problem")
            raise
        finally:
            if self.cleanup:
                self.clean_up_matrix_building_tables()
                self.clean_up_subset_tables()
                logger.notice(
                    "Cleanup flag was set to True, so label, cohort and subset tables were deleted"
                )

    def _log_end_of_run_report(self):
        missing_matrices = missing_matrix_uuids(self.experiment_hash, self.db_engine)
        if len(missing_matrices) > 0:
            logger.notice(
                f"Found {len(missing_matrices)} missing matrix uuids."
                f"This means that they were supposed to either be build or reused"
                f"by this experiment but are not present in the matrices table."
                f"Inspect the logs for any matrix building errors. Full list: {missing_matrices}",
            )
        else:
            logger.success(
                "All matrices that were supposed to be build were built. Awesome!"
            )

        missing_models = missing_model_hashes(self.experiment_hash, self.db_engine)
        if len(missing_models) > 0:
            logger.notice(
                f"Found {len(missing_models)} missing model hashes. "
                f"This means that they were supposed to either be trained or reused "
                f"by this experiment but are not present in the models table. "
                f"Inspect the logs for any training errors. Full list: {missing_models}"
            )
        else:
            logger.success(
                "All models that were supposed to be trained were trained. Awesome!"
            )

    def clean_up_matrix_building_tables(self):
        logger.debug("Cleaning up cohort and labels tables")
        with timeout(self.cleanup_timeout):
            self.cohort_table_generator.clean_up()
            self.label_generator.clean_up(self.labels_table_name)
        logger.debug("Cleaning up cohort and labels tables: completed")

    def clean_up_subset_tables(self):
        logger.debug("Cleaning up cohort and labels tables")
        with timeout(self.cleanup_timeout):
            for subset_task in self.subset_tasks:
                subset_task["subset_table_generator"].clean_up()
        logger.debug("Cleaning up cohort and labels tables: completed")

    def _run_profile(self):
        cp = cProfile.Profile()
        cp.runcall(self._run)
        store = self.project_storage.get_store(
            ["profiling_stats"], f"{int(time.time())}.profile"
        )
        with store.open("wb") as fd:
            cp.create_stats()
            marshal.dump(cp.stats, fd)
            logger.spam(
                f"Profiling stats of this Triage run calculated and written to {store}"
                f"in cProfile format."
            )

    @experiment_entrypoint
    def run(self):
        try:
            if self.profile:
                self._run_profile()
            else:
                self._run()
        except Exception:
            logger.exception("Run interrupted by uncaught exception")
            raise

    __call__ = run

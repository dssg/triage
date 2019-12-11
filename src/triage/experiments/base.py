import logging
from abc import ABC, abstractmethod
import cProfile
import marshal
import random
import time
import os
import itertools
import yaml

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
    EntityDateTableGeneratorNoOp,
)
from triage.component.timechop import Timechop
from triage.component import results_schema
from triage.component.catwalk import (
    ModelTrainer,
    ModelEvaluator,
    Predictor,
    IndividualImportanceCalculator,
    ModelGrouper,
    ModelTrainTester,
    Subsetter
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
    record_matrix_building_started,
    record_model_building_started,
)

from triage.database_reflection import table_has_data
from triage.util.conf import dt_from_str, parse_from_obj
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
        skip_validation=False,
    ):
        experiment_kwargs = bind_kwargs(
            self.__class__,
            **{key: value for (key, value) in locals().items() if key not in {'db_engine', 'config', 'self'}}
        )

        self._check_config_version(config)
        self.config = config

        self.config['random_seed'] = self.config.get('random_seed', random.randint(1,1e7))

        random.seed(self.config['random_seed'])

        self.project_storage = ProjectStorage(project_path)
        self.model_storage_engine = ModelStorageEngine(self.project_storage)
        self.matrix_storage_engine = MatrixStorageEngine(
            self.project_storage, matrix_storage_class
        )
        self.project_path = project_path
        self.replace = replace
        self.save_predictions = save_predictions
        self.skip_validation = skip_validation
        self.db_engine = db_engine
        results_schema.upgrade_if_clean(dburl=self.db_engine.url)

        self.features_schema_name = "features"
        self.materialize_subquery_fromobjs = materialize_subquery_fromobjs
        self.features_ignore_cohort = features_ignore_cohort


        ## Defaults to sane values
        self.config['temporal_config'] = self._fill_timechop_config_missing()
        ## Defaults to all the entities found in the features_aggregation's from_obj
        self.config['cohort_config'] = self._fill_cohort_config_missing()
        ## Defaults to all the feature_aggregation's prefixes
        self.config['feature_group_definition'] = self._fill_feature_group_definition()

        # if using a model grid preset, fill in the actual grid
        if self.config.get('model_grid_preset'):
            if self.config.get('grid_config'):
                raise KeyError("There can only be one (cannot specify both model_grid_preset and grid_config)")
            self.config['grid_config'] = self._fill_model_grid_presets(self.config.get('model_grid_preset'))
            # remove the "model_grid_preset" key now that we've filled out the grid so you could re-run
            # the resulting exepriment config
            self.config.pop('model_grid_preset')

        ###################### RUBICON ######################

        self.experiment_hash = save_experiment_and_get_hash(self.config, self.db_engine)
        self.run_id = initialize_tracking_and_get_run_id(
            self.experiment_hash,
            experiment_class_path=classpath(self.__class__),
            experiment_kwargs=experiment_kwargs,
            db_engine=self.db_engine
        )
        self.initialize_components()

        self.cleanup = cleanup
        if self.cleanup:
            logging.info(
                "cleanup is set to True, so intermediate tables (labels and cohort) "
                "will be removed after matrix creation and subset tables will be "
                "removed after model training and testing"
            )
        else:
            logging.info(
                "cleanup is set to False, so intermediate tables (labels, cohort, and subsets) "
                "will not be removed"
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

    @cachedproperty
    def cohort_hash(self):
        if "query" in self.config.get("cohort_config", {}):
            return filename_friendly_hash(self.config["cohort_config"]["query"])
        else:
            return None

    def initialize_components(self):
        split_config = self.config["temporal_config"]

        self.chopper = Timechop(**split_config)

        cohort_config = self.config.get("cohort_config", {})
        if "query" in cohort_config:
            self.cohort_table_name = "cohort_{}_{}".format(
                cohort_config.get('name', 'default'),
                self.cohort_hash
            )
            self.cohort_table_generator = EntityDateTableGenerator(
                entity_date_table_name=self.cohort_table_name,
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
            self.cohort_table_generator = EntityDateTableGeneratorNoOp()

        self.subsets = [None] + self.config.get("scoring", {}).get("subsets", [])

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

        if "bias_audit_config" in self.config:
            bias_config = self.config["bias_audit_config"]
            self.bias_hash = filename_friendly_hash(bias_config)
            self.protected_groups_table_name = f"protected_groups_{self.bias_hash}"
            self.protected_groups_generator = ProtectedGroupsGenerator(
                db_engine=self.db_engine,
                from_obj=parse_from_obj(bias_config, 'bias_from_obj'),
                attribute_columns=bias_config.get("attribute_columns", None),
                entity_id_column=bias_config.get("entity_id_column", None),
                knowledge_date_column=bias_config.get("knowledge_date_column", None),
                protected_groups_table_name=self.protected_groups_table_name,
                replace=self.replace
            )
        else:
            self.protected_groups_generator = ProtectedGroupsGeneratorNoOp()
            logging.warning(
                "bias_audit_config missing or unrecognized. Without protected groups, "
                "you will not audit your models for bias and fairness."
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
            run_id=self.run_id,
        )

        self.subsetter = Subsetter(
            db_engine=self.db_engine,
            replace=self.replace,
            as_of_times=self.all_as_of_times
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
            rank_order=self.config.get("prediction", {}).get("rank_tiebreaker", "worst"),
        )

        self.individual_importance_calculator = IndividualImportanceCalculator(
            db_engine=self.db_engine,
            n_ranks=self.config.get("individual_importance", {}).get("n_ranks", 5),
            methods=self.config.get("individual_importance", {}).get("methods", ["uniform"]),
            replace=self.replace,
        )

        self.evaluator = ModelEvaluator(
            db_engine=self.db_engine,
            testing_metric_groups=self.config.get("scoring", {}).get("testing_metric_groups", []),
            training_metric_groups=self.config.get("scoring", {}).get("training_metric_groups", []),
            bias_config=self.config.get("bias_audit_config", {})
        )

        self.model_train_tester = ModelTrainTester(
            matrix_storage_engine=self.matrix_storage_engine,
            model_evaluator=self.evaluator,
            model_trainer=self.trainer,
            individual_importance_calculator=self.individual_importance_calculator,
            predictor=self.predictor,
            subsets=self.subsets,
            protected_groups_generator=self.protected_groups_generator,
            cohort_hash=self.cohort_hash
        )

    def get_for_update(self):
        return get_for_update(self.db_engine, results_schema.Experiment, self.experiment_hash)

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

        with self.get_for_update() as experiment:
            experiment.time_splits = len(split_definitions)
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
        with self.get_for_update() as experiment:
            experiment.as_of_times = len(distinct_as_of_times)
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
        with self.get_for_update() as experiment:
            experiment.total_features = sum(1 for _feature in itertools.chain.from_iterable(result.values()))
        return result

    @cachedproperty
    def feature_dicts(self):
        """Feature dictionaries, representing the feature tables and
        columns configured in this experiment after computing feature
        groups.

        Returns: (list) of dicts, keys being feature table names and
        values being lists of feature names

        """
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

    @cachedproperty
    def subset_tasks(self):
        return self.subsetter.generate_tasks(self.subsets)

    @experiment_entrypoint
    def generate_labels(self):
        """Generate labels based on experiment configuration

        Results are stored in the database, not returned
        """
        self.label_generator.generate_all_labels(
            self.labels_table_name, self.all_as_of_times, self.all_label_timespans
        )

    @experiment_entrypoint
    def generate_cohort(self):
        self.cohort_table_generator.generate_entity_date_table(
            as_of_dates=self.all_as_of_times
        )

    @experiment_entrypoint
    def generate_protected_groups(self):
        """Generate protected groups table based on experiment configuration

        Results are stored in the database, not returned
        """
        self.protected_groups_generator.generate_all_dates(
            self.all_as_of_times, self.cohort_table_name, self.cohort_hash
        )

    def generate_subset(self, subset_hash):
        self.subsets["subset_hash"].subset_table_generator.generate_entity_date_table(
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
        self.process_query_tasks(self.feature_aggregation_table_tasks)
        logging.info(
            "Finished running preimputation feature queries. The final results are in tables: %s",
            ",".join(agg.get_table_name() for agg in self.collate_aggregations),
        )

    @experiment_entrypoint
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
        with self.get_for_update() as experiment:
            experiment.matrices_needed = len(self.matrix_build_tasks.keys())
        record_matrix_building_started(self.run_id, self.db_engine)
        self.process_matrix_build_tasks(self.matrix_build_tasks)

    @experiment_entrypoint
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

    @experiment_entrypoint
    def generate_subsets(self):
        if self.subsets:
            logging.info("Beginning subset generation")
            self.process_subset_tasks(self.subset_tasks)
        else:
            logging.info("No subsets found. Proceeding to training and testing models")

    def _all_train_test_batches(self):
        if "grid_config" not in self.config:
            logging.warning(
                "No grid_config was passed in the experiment config. No models will be trained"
            )
            return

        return self.model_train_tester.generate_task_batches(
            splits=self.full_matrix_definitions,
            grid_config=self.config.get('grid_config'),
            model_comment=self.config.get('model_comment', None)
        )

    @experiment_entrypoint
    def train_and_test_models(self):
        self.generate_subsets()
        logging.info("Creating protected groups table")
        self.generate_protected_groups()
        batches = self._all_train_test_batches()
        if not batches:
            logging.warning("No train/test tasks found, so no training to do")
            return

        with self.get_for_update() as experiment:
            experiment.grid_size = sum(
                1 for _param in self.trainer.flattened_grid_config(self.config.get('grid_config')))

        logging.info("%s train/test batches found. Beginning training.", len(batches))
        model_hashes = set(task['train_kwargs']['model_hash'] for batch in batches for task in batch.tasks)
        associate_models_with_experiment(
            self.experiment_hash,
            model_hashes,
            self.db_engine
        )
        with self.get_for_update() as experiment:
            experiment.models_needed = len(model_hashes)
        record_model_building_started(self.run_id, self.db_engine)
        self.process_train_test_batches(batches)

    def validate(self, strict=True):
        ExperimentValidator(self.db_engine, strict=strict).run(self.config)

    def _run(self):
        if not self.skip_validation:
            self.validate()

        logging.info("Generating matrices")
        try:
            self.generate_matrices()
            self.train_and_test_models()
        finally:
            if self.cleanup:
                self.clean_up_matrix_building_tables()
                self.clean_up_subset_tables()
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

    def clean_up_matrix_building_tables(self):
        logging.info("Cleaning up cohort and labels tables")
        with timeout(self.cleanup_timeout):
            self.cohort_table_generator.clean_up()
            self.label_generator.clean_up(self.labels_table_name)

    def clean_up_subset_tables(self):
        logging.info("Cleaning up cohort and labels tables")
        with timeout(self.cleanup_timeout):
            for subset_task in self.subset_tasks:
                subset_task["subset_table_generator"].clean_up()

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

    def _fill_timechop_config_missing(self):
        """
        Fill with default values the temporal_config params if they are missing
        """
        timechop_config = self.config['temporal_config']

        default_config = {'model_update_frequency': '100y',
                          'training_as_of_date_frequencies': '100y',
                          'test_as_of_date_frequencies': '100y',
                          'max_training_histories': '0d',
                          'test_durations': '0d',
                          }

        # Checks if label_timespan is present
        if 'label_timespans' in timechop_config.keys():
            if any([k in timechop_config.keys() for k in ['training_label_timespans', 'test_labels_timespans']]):
                raise KeyError("You can't always get what you want")
            default_config['training_label_timespans'] = default_config['test_label_timespans'] = timechop_config['label_timespans']
            timechop_config.pop('label_timespans') ## We don't need this value anymore

        # Checks if some of the date range  limits  is missing, if so repalces with
        # min, max accordingy from de from_objs
        if any([k not in timechop_config.keys() for k in ['feature_start_time', 'feature_end_time', 'label_start_time', 'label_end_time']]):
            from_query = "(select min({knowledge_date}) as min_date, max({knowledge_date}) as max_date from (select * from {from_obj}) as t)"

            feature_aggregations = self.config['feature_aggregations']

            from_queries = [from_query.format(knowledge_date = agg['knowledge_date_column'], from_obj=agg['from_obj']) for agg in feature_aggregations]

            unions = "\n union \n".join(from_queries)

            query = "select to_char(min(min_date), 'YYYY-MM-DD'), to_char(max(max_date), 'YYYY-MM-DD') from ({unions}) as u".format(unions=unions)

            with self.db_engine.connect() as conn:
                rs = conn.execute(query)
                min_date, max_date = rs.fetchall()[0]

            default_config['feature_start_time'] = default_config['label_start_time'] = min_date
            default_config['feature_end_time'] = default_config['label_end_time'] = max_date

        # Replaces missing values
        default_config.update(timechop_config)

        return default_config


    def _fill_cohort_config_missing(self):
        """
        If none cohort_config section is provided, include all the entities by default
        """
        from_query = "(select entity_id, {knowledge_date} as knowledge_date from (select * from {from_obj}) as t)"

        feature_aggregations = self.config['feature_aggregations']

        from_queries = [from_query.format(knowledge_date = agg['knowledge_date_column'], from_obj=agg['from_obj']) for agg in feature_aggregations]

        unions = "\n union \n".join(from_queries)

        query = f"select distinct entity_id from ({unions}) as e" +" where knowledge_date < '{as_of_date}'"

        cohort_config = self.config.get('cohort_config', {})
        default_config = {'query': query, 'name': 'all_entities'}

        default_config.update(cohort_config)

        return default_config

    def _fill_feature_group_definition(self):
        """
        If feature_group_definition is not presents, this function sets it to all
        the distinct feature_aggregations' prefixes
        """
        feature_group_definition = self.config.get('feature_group_definition', {})
        if not feature_group_definition:
            feature_aggregations = self.config['feature_aggregations']

            feature_group_definition['prefix'] = list({agg['prefix'] for agg in feature_aggregations})

        return feature_group_definition

    def _fill_model_grid_presets(self, grid_type):
        """Load a preset model grid.

           Args:
                grid_type (string) The type of preset grid to load. May
                    by `quickstart`, `small`, `medium`, `large`, or `texas`

            Returns: (dict) a triage model grid config
        """

        # Load the model grid presets from a yaml file, which should be structured
        # with grid-types as a top level key and each grid-type building on
        presets_file = os.path.join(os.path.dirname(__file__), 'model_grid_presets.yaml')
        with open(presets_file, 'r') as f:
            model_grid_presets = yaml.safe_load(f)

        # output is a collector for the resulting grid, so initialize it with the parameters
        # at the level of preset grid we want and find the next-lowest level to incorporate
        output = model_grid_presets[grid_type]['grid'].copy()
        prev_type = model_grid_presets[grid_type]['prev']

        # collapse the grid parameters down the levels until we reach one with no lower level
        while prev_type is not None:
            prev = model_grid_presets[prev_type]['grid'].copy()

            # look for new model types and hyperparameters to incorporate into the output
            for model_type in set(output.keys()).union(set(prev.keys())):
                curr_model = output.get(model_type, {}).copy()
                # if the model type exists in the lower-level preset, update any associated hyperparameter
                # values in the output (those only in the higher level grid will pass through unchanged)
                for hyperparam in prev.get(model_type, {}).keys():
                    curr_model[hyperparam] = sorted(list(set(curr_model.get(hyperparam, []) + prev[model_type][hyperparam])), key=lambda x: x if x is not None else 0)
                output[model_type] = curr_model

            # traverse the linked list to one level deeper and repeat
            prev_type = model_grid_presets[prev_type]['prev']

        return output


    @experiment_entrypoint
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

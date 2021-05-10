import os
from datetime import datetime
from tempfile import TemporaryDirectory
import yaml


import testing.postgresql
from triage import create_engine

from triage.component.results_schema import Base
from triage.component.timechop import Timechop
from triage.component.architect.features import (
    FeatureGenerator,
    FeatureDictionaryCreator,
    FeatureGroupCreator,
    FeatureGroupMixer,
)
from triage.component.architect.label_generators import LabelGenerator
from triage.component.architect.entity_date_table_generators import EntityDateTableGenerator
from triage.component.architect.planner import Planner
from triage.component.architect.builders import MatrixBuilder
from triage.component.catwalk.storage import ProjectStorage

from tests.utils import sample_config


def populate_source_data(db_engine):
    cat_complaints = [
        (1, "2010-10-01", 5),
        (1, "2011-10-01", 4),
        (1, "2011-11-01", 4),
        (1, "2011-12-01", 4),
        (1, "2012-02-01", 5),
        (1, "2012-10-01", 4),
        (1, "2013-10-01", 5),
        (2, "2010-10-01", 5),
        (2, "2011-10-01", 5),
        (2, "2011-11-01", 4),
        (2, "2011-12-01", 4),
        (2, "2012-02-01", 6),
        (2, "2012-10-01", 5),
        (2, "2013-10-01", 6),
        (3, "2010-10-01", 5),
        (3, "2011-10-01", 3),
        (3, "2011-11-01", 4),
        (3, "2011-12-01", 4),
        (3, "2012-02-01", 4),
        (3, "2012-10-01", 3),
        (3, "2013-10-01", 4),
    ]

    dog_complaints = [
        (1, "2010-10-01", 5),
        (1, "2011-10-01", 4),
        (1, "2011-11-01", 4),
        (1, "2011-12-01", 4),
        (1, "2012-02-01", 5),
        (1, "2012-10-01", 4),
        (1, "2013-10-01", 5),
        (2, "2010-10-01", 5),
        (2, "2011-10-01", 5),
        (2, "2011-11-01", 4),
        (2, "2011-12-01", 4),
        (2, "2012-02-01", 6),
        (2, "2012-10-01", 5),
        (2, "2013-10-01", 6),
        (3, "2010-10-01", 5),
        (3, "2011-10-01", 3),
        (3, "2011-11-01", 4),
        (3, "2011-12-01", 4),
        (3, "2012-02-01", 4),
        (3, "2012-10-01", 3),
        (3, "2013-10-01", 4),
    ]

    events = [
        (1, 1, "2011-01-01"),
        (1, 1, "2011-06-01"),
        (1, 1, "2011-09-01"),
        (1, 1, "2012-01-01"),
        (1, 1, "2012-01-10"),
        (1, 1, "2012-06-01"),
        (1, 1, "2013-01-01"),
        (1, 0, "2014-01-01"),
        (1, 1, "2015-01-01"),
        (2, 1, "2011-01-01"),
        (2, 1, "2011-06-01"),
        (2, 1, "2011-09-01"),
        (2, 1, "2012-01-01"),
        (2, 1, "2013-01-01"),
        (2, 1, "2014-01-01"),
        (2, 1, "2015-01-01"),
        (3, 0, "2011-01-01"),
        (3, 0, "2011-06-01"),
        (3, 0, "2011-09-01"),
        (3, 0, "2012-01-01"),
        (3, 0, "2013-01-01"),
        (3, 1, "2014-01-01"),
        (3, 0, "2015-01-01"),
    ]

    db_engine.execute(
        """create table cat_complaints (
        entity_id int,
        as_of_date date,
        cat_sightings int
        )"""
    )

    for complaint in cat_complaints:
        db_engine.execute(
            "insert into cat_complaints values (%s, %s, %s)", complaint)

    db_engine.execute(
        """create table dog_complaints (
        entity_id int,
        as_of_date date,
        dog_sightings int
        )"""
    )

    for complaint in dog_complaints:
        db_engine.execute(
            "insert into dog_complaints values (%s, %s, %s)", complaint)

    db_engine.execute(
        """create table events (
        entity_id int,
        outcome int,
        outcome_date date
    )"""
    )

    for event in events:
        db_engine.execute("insert into events values (%s, %s, %s)", event)


def basic_integration_test(
    cohort_names,
    feature_group_create_rules,
    feature_group_mix_rules,
    expected_matrix_multiplier,
    expected_group_lists,
):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        Base.metadata.create_all(db_engine)
        populate_source_data(db_engine)

        with TemporaryDirectory() as temp_dir:
            chopper = Timechop(
                feature_start_time=datetime(2010, 1, 1),
                feature_end_time=datetime(2014, 1, 1),
                label_start_time=datetime(2011, 1, 1),
                label_end_time=datetime(2014, 1, 1),
                model_update_frequency="1year",
                training_label_timespans=["6months"],
                test_label_timespans=["6months"],
                training_as_of_date_frequencies="1day",
                test_as_of_date_frequencies="3months",
                max_training_histories=["1months"],
                test_durations=["1months"],
            )

            entity_date_table_generator = EntityDateTableGenerator(
                db_engine=db_engine,
                entity_date_table_name="cohort_abcd",
                query="select distinct(entity_id) from events"
            )

            label_generator = LabelGenerator(
                db_engine=db_engine, query=sample_config()[
                    "label_config"]["query"]
            )

            feature_generator = FeatureGenerator(
                db_engine=db_engine, features_schema_name="features", replace=True
            )

            feature_dictionary_creator = FeatureDictionaryCreator(
                db_engine=db_engine, features_schema_name="features"
            )

            feature_group_creator = FeatureGroupCreator(
                feature_group_create_rules)

            feature_group_mixer = FeatureGroupMixer(feature_group_mix_rules)
            project_storage = ProjectStorage(temp_dir)
            planner = Planner(
                feature_start_time=datetime(2010, 1, 1),
                label_names=["outcome"],
                label_types=["binary"],
                cohort_names=cohort_names,
                user_metadata={},
            )

            builder = MatrixBuilder(
                engine=db_engine,
                db_config={
                    "features_schema_name": "features",
                    "labels_schema_name": "public",
                    "labels_table_name": "labels",
                    "cohort_table_name": "cohort_abcd",
                },
                experiment_hash=None,
                matrix_storage_engine=project_storage.matrix_storage_engine(),
                replace=True,
            )

            # chop time
            split_definitions = chopper.chop_time()
            num_split_matrices = sum(
                1 + len(split["test_matrices"]) for split in split_definitions
            )

            # generate as_of_times for feature/label/state generation
            all_as_of_times = []
            for split in split_definitions:
                all_as_of_times.extend(split["train_matrix"]["as_of_times"])
                for test_matrix in split["test_matrices"]:
                    all_as_of_times.extend(test_matrix["as_of_times"])
            all_as_of_times = list(set(all_as_of_times))

            # generate entity_date state table
            entity_date_table_generator.generate_entity_date_table(
                as_of_dates=all_as_of_times)

            # create labels table
            label_generator.generate_all_labels(
                labels_table="labels",
                as_of_dates=all_as_of_times,
                label_timespans=["6months"],
            )

            # create feature table tasks
            # we would use FeatureGenerator#create_all_tables but want to use
            # the tasks dict directly to create a feature dict
            aggregations = feature_generator.aggregations(
                feature_aggregation_config=[
                    {
                        "prefix": "cat",
                        "from_obj": "cat_complaints",
                        "knowledge_date_column": "as_of_date",
                        "aggregates": [
                            {
                                "quantity": "cat_sightings",
                                "metrics": ["count", "avg"],
                                "imputation": {"all": {"type": "mean"}},
                            }
                        ],
                        "intervals": ["1y"],
                        "groups": ["entity_id"],
                    },
                    {
                        "prefix": "dog",
                        "from_obj": "dog_complaints",
                        "knowledge_date_column": "as_of_date",
                        "aggregates_imputation": {
                            "count": {"type": "constant", "value": 7},
                            "sum": {"type": "mean"},
                            "avg": {"type": "zero"},
                        },
                        "aggregates": [
                            {"quantity": "dog_sightings",
                                "metrics": ["count", "avg"]}
                        ],
                        "intervals": ["1y"],
                        "groups": ["entity_id"],
                    },
                ],
                feature_dates=all_as_of_times,
                state_table=entity_date_table_generator.entity_date_table_name,
            )
            feature_table_agg_tasks = feature_generator.generate_all_table_tasks(
                aggregations, task_type="aggregation"
            )

            # create feature aggregation tables
            feature_generator.process_table_tasks(feature_table_agg_tasks)

            feature_table_imp_tasks = feature_generator.generate_all_table_tasks(
                aggregations, task_type="imputation"
            )

            # create feature imputation tables
            feature_generator.process_table_tasks(feature_table_imp_tasks)

            # build feature dictionaries from feature tables and
            # subsetting config
            master_feature_dict = feature_dictionary_creator.feature_dictionary(
                feature_table_names=feature_table_imp_tasks.keys(),
                index_column_lookup=feature_generator.index_column_lookup(
                    aggregations),
            )

            feature_dicts = feature_group_mixer.generate(
                feature_group_creator.subsets(master_feature_dict)
            )

            # figure out what matrices need to be built
            _, matrix_build_tasks = planner.generate_plans(
                split_definitions, feature_dicts
            )

            # go and build the matrices
            builder.build_all_matrices(matrix_build_tasks)

            # super basic assertion: did matrices we expect get created?
            matrices_records = list(
                db_engine.execute(
                    """select matrix_uuid, num_observations, matrix_type
                    from triage_metadata.matrices
                    """
                )
            )
            matrix_directory = os.path.join(temp_dir, "matrices")
            matrices = [path for path in os.listdir(
                matrix_directory) if ".csv" in path]
            metadatas = [
                path for path in os.listdir(matrix_directory) if ".yaml" in path
            ]
            assert len(matrices) == num_split_matrices * \
                expected_matrix_multiplier
            assert len(metadatas) == num_split_matrices * \
                expected_matrix_multiplier
            assert len(matrices) == len(matrices_records)
            feature_group_name_lists = []
            for metadata_path in metadatas:
                with open(os.path.join(matrix_directory, metadata_path)) as f:
                    metadata = yaml.load(f, Loader=yaml.Loader)
                    feature_group_name_lists.append(metadata["feature_groups"])

            for matrix_uuid, num_observations, matrix_type in matrices_records:
                assert matrix_uuid in matrix_build_tasks  # the hashes of the matrices
                assert type(num_observations) is int
                assert matrix_type == matrix_build_tasks[matrix_uuid]["matrix_type"]

            def deep_unique_tuple(l):
                return set([tuple(i) for i in l])

            assert deep_unique_tuple(feature_group_name_lists) == deep_unique_tuple(
                expected_group_lists
            )


def test_integration_simple():
    basic_integration_test(
        cohort_names=["mycohort"],
        feature_group_create_rules={"prefix": ["cat", "dog"]}, # Will be set by defaults.py
        feature_group_mix_rules=["all"],
        # only looking at one state, and one feature group.
        # so we don't multiply timechop's output by anything
        expected_matrix_multiplier=1,
        expected_group_lists=[["prefix: cat", "prefix: dog"]],
    )


def test_integration_feature_grouping():
    basic_integration_test(
        cohort_names=["mycohort"],
        feature_group_create_rules={"prefix": ["cat", "dog"]},
        feature_group_mix_rules=["leave-one-out", "all"],
        # 3 feature groups (cat/dog/cat+dog),
        # so the # of matrices should be each train/test split *3
        expected_matrix_multiplier=3,
        expected_group_lists=[
            ["prefix: cat"],
            ["prefix: cat", "prefix: dog"],
            ["prefix: dog"],
        ],
    )

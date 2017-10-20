import testing.postgresql
from sqlalchemy import create_engine
import os
from datetime import datetime
from tempfile import TemporaryDirectory
from results_schema import Base
from timechop.timechop import Timechop
from architect.features import \
    FeatureGenerator,\
    FeatureDictionaryCreator,\
    FeatureGroupCreator,\
    FeatureGroupMixer
from architect.state_table_generators import StateTableGenerator
from architect.label_generators import BinaryLabelGenerator
from architect.planner import Planner


def populate_source_data(db_engine):
    cat_complaints = [
        (1, '2010-10-01', 5),
        (1, '2011-10-01', 4),
        (1, '2011-11-01', 4),
        (1, '2011-12-01', 4),
        (1, '2012-02-01', 5),
        (1, '2012-10-01', 4),
        (1, '2013-10-01', 5),
        (2, '2010-10-01', 5),
        (2, '2011-10-01', 5),
        (2, '2011-11-01', 4),
        (2, '2011-12-01', 4),
        (2, '2012-02-01', 6),
        (2, '2012-10-01', 5),
        (2, '2013-10-01', 6),
        (3, '2010-10-01', 5),
        (3, '2011-10-01', 3),
        (3, '2011-11-01', 4),
        (3, '2011-12-01', 4),
        (3, '2012-02-01', 4),
        (3, '2012-10-01', 3),
        (3, '2013-10-01', 4),
    ]

    dog_complaints = [
        (1, '2010-10-01', 5),
        (1, '2011-10-01', 4),
        (1, '2011-11-01', 4),
        (1, '2011-12-01', 4),
        (1, '2012-02-01', 5),
        (1, '2012-10-01', 4),
        (1, '2013-10-01', 5),
        (2, '2010-10-01', 5),
        (2, '2011-10-01', 5),
        (2, '2011-11-01', 4),
        (2, '2011-12-01', 4),
        (2, '2012-02-01', 6),
        (2, '2012-10-01', 5),
        (2, '2013-10-01', 6),
        (3, '2010-10-01', 5),
        (3, '2011-10-01', 3),
        (3, '2011-11-01', 4),
        (3, '2011-12-01', 4),
        (3, '2012-02-01', 4),
        (3, '2012-10-01', 3),
        (3, '2013-10-01', 4),
    ]

    events = [
        (1, 1, '2011-01-01'),
        (1, 1, '2011-06-01'),
        (1, 1, '2011-09-01'),
        (1, 1, '2012-01-01'),
        (1, 1, '2012-01-10'),
        (1, 1, '2012-06-01'),
        (1, 1, '2013-01-01'),
        (1, 0, '2014-01-01'),
        (1, 1, '2015-01-01'),
        (2, 1, '2011-01-01'),
        (2, 1, '2011-06-01'),
        (2, 1, '2011-09-01'),
        (2, 1, '2012-01-01'),
        (2, 1, '2013-01-01'),
        (2, 1, '2014-01-01'),
        (2, 1, '2015-01-01'),
        (3, 0, '2011-01-01'),
        (3, 0, '2011-06-01'),
        (3, 0, '2011-09-01'),
        (3, 0, '2012-01-01'),
        (3, 0, '2013-01-01'),
        (3, 1, '2014-01-01'),
        (3, 0, '2015-01-01'),
    ]

    states = [
        (1, 'state_one', '2012-01-01', '2016-01-01'),
        (1, 'state_two', '2013-01-01', '2016-01-01'),
        (2, 'state_one', '2012-01-01', '2016-01-01'),
        (2, 'state_two', '2013-01-01', '2016-01-01'),
        (3, 'state_one', '2012-01-01', '2016-01-01'),
        (3, 'state_two', '2013-01-01', '2016-01-01'),
    ]

    db_engine.execute('''create table cat_complaints (
        entity_id int,
        as_of_date date,
        cat_sightings int
        )''')

    for complaint in cat_complaints:
        db_engine.execute(
            "insert into cat_complaints values (%s, %s, %s)",
            complaint
        )

    db_engine.execute('''create table dog_complaints (
        entity_id int,
        as_of_date date,
        dog_sightings int
        )''')

    for complaint in dog_complaints:
        db_engine.execute(
            "insert into dog_complaints values (%s, %s, %s)",
            complaint
        )

    db_engine.execute('''create table events (
        entity_id int,
        outcome int,
        outcome_date date
    )''')

    for event in events:
        db_engine.execute(
            "insert into events values (%s, %s, %s)",
            event
        )

    db_engine.execute('''create table states (
        entity_id int,
        state text,
        start_time timestamp,
        end_time timestamp
    )''')

    for state in states:
        db_engine.execute(
            'insert into states values (%s, %s, %s, %s)',
            state
        )


def basic_integration_test(
    state_filters,
    feature_group_create_rules,
    feature_group_mix_rules,
    expected_num_matrices
):
    with testing.postgresql.Postgresql() as postgresql:
        db_engine = create_engine(postgresql.url())
        Base.metadata.create_all(db_engine)
        populate_source_data(db_engine)

        with TemporaryDirectory() as temp_dir:
            chopper = Timechop(
                beginning_of_time=datetime(2010, 1, 1),
                modeling_start_time=datetime(2011, 1, 1),
                modeling_end_time=datetime(2014, 1, 1),
                update_window='1y',
                train_label_windows=['6months'],
                test_label_windows=['6months'],
                train_example_frequency='1day',
                test_example_frequency='3months',
                train_durations=['1months'],
                test_durations=['1months'],
            )

            state_table_generator = StateTableGenerator(
                db_engine=db_engine,
                experiment_hash='abcd',
                dense_state_table='states',
            )


            label_generator = BinaryLabelGenerator(
                db_engine=db_engine,
                events_table='events'
            )

            feature_generator = FeatureGenerator(
                db_engine=db_engine,
                features_schema_name='features',
                replace=True,
            )

            feature_dictionary_creator = FeatureDictionaryCreator(
                db_engine=db_engine,
                features_schema_name='features'
            )

            feature_group_creator = FeatureGroupCreator(feature_group_create_rules)

            feature_group_mixer = FeatureGroupMixer(feature_group_mix_rules)

            planner = Planner(
                engine=db_engine,
                beginning_of_time=datetime(2010, 1, 1),
                label_names=['outcome'],
                label_types=['binary'],
                db_config={
                    'features_schema_name': 'features',
                    'labels_schema_name': 'public',
                    'labels_table_name': 'labels',
                    'sparse_state_table_name': 'tmp_sparse_states_abcd',
                },
                matrix_directory=os.path.join(temp_dir, 'matrices'),
                states=state_filters,
                user_metadata={},
                replace=True
            )

            # chop time
            split_definitions = chopper.chop_time()

            # generate as_of_times for feature/label/state generation
            all_as_of_times = []
            for split in split_definitions:
                all_as_of_times.extend(split['train_matrix']['as_of_times'])
                for test_matrix in split['test_matrices']:
                    all_as_of_times.extend(test_matrix['as_of_times'])
            all_as_of_times = list(set(all_as_of_times))

            feature_aggregation_config = [{
                'prefix': 'cat',
                'from_obj': 'cat_complaints',
                'knowledge_date_column': 'as_of_date',
                'aggregates': [{
                    'quantity': 'cat_sightings',
                    'metrics': ['count', 'avg'],
                    'imputation': {
                        'all': {'type': 'mean'}
                    }
                }],
                'intervals': ['1y'],
                'groups': ['entity_id']
            }, {
                'prefix': 'dog',
                'from_obj': 'dog_complaints',
                'knowledge_date_column': 'as_of_date',
                'aggregates_imputation': {
                    'count': {'type': 'constant', 'value': 7},
                    'sum': {'type': 'mean'},
                    'avg': {'type': 'zero'}
                },
                'aggregates': [{
                    'quantity': 'dog_sightings',
                    'metrics': ['count', 'avg'],
                }],
                'intervals': ['1y'],
                'groups': ['entity_id']
            }]

            state_table_generator.validate()
            label_generator.validate()
            feature_generator.validate(feature_aggregation_config)
            feature_group_creator.validate()
            planner.validate()

            # generate sparse state table
            state_table_generator.generate_sparse_table(
                as_of_dates=all_as_of_times
            )

            # create labels table
            label_generator.generate_all_labels(
                labels_table='labels',
                as_of_dates=all_as_of_times,
                label_windows=['6months']
            )

            # create feature table tasks
            # we would use FeatureGenerator#create_all_tables but want to use
            # the tasks dict directly to create a feature dict
            aggregations = feature_generator.aggregations(
                feature_aggregation_config=[{
                    'prefix': 'cat',
                    'from_obj': 'cat_complaints',
                    'knowledge_date_column': 'as_of_date',
                    'aggregates': [{
                        'quantity': 'cat_sightings',
                        'metrics': ['count', 'avg'],
                        'imputation': {
                            'all': {'type': 'mean'}
                        }
                    }],
                    'intervals': ['1y'],
                    'groups': ['entity_id']
                }, {
                    'prefix': 'dog',
                    'from_obj': 'dog_complaints',
                    'knowledge_date_column': 'as_of_date',
                    'aggregates_imputation': {
                        'count': {'type': 'constant', 'value': 7},
                        'sum': {'type': 'mean'},
                        'avg': {'type': 'zero'}
                    },
                    'aggregates': [{
                        'quantity': 'dog_sightings',
                        'metrics': ['count', 'avg'],
                    }],
                    'intervals': ['1y'],
                    'groups': ['entity_id']
                }],
                feature_dates=all_as_of_times,
                state_table=state_table_generator.sparse_table_name
            )
            feature_table_agg_tasks = feature_generator.generate_all_table_tasks(aggregations, task_type='aggregation')

            # create feature aggregation tables
            feature_generator.process_table_tasks(feature_table_agg_tasks)

            feature_table_imp_tasks = feature_generator.generate_all_table_tasks(aggregations, task_type='imputation')

            # create feature imputation tables
            feature_generator.process_table_tasks(feature_table_imp_tasks)

            # build feature dictionaries from feature tables and
            # subsetting config
            master_feature_dict = feature_dictionary_creator.feature_dictionary(
                feature_table_names=feature_table_imp_tasks.keys(),
                index_column_lookup=feature_generator.index_column_lookup(aggregations)
            )

            feature_dicts = feature_group_mixer.generate(
                feature_group_creator.subsets(master_feature_dict)
            )

            # figure out what matrices need to be built
            _, matrix_build_tasks =\
                planner.generate_plans(
                    split_definitions,
                    feature_dicts
                )

            # go and build the matrices
            planner.build_all_matrices(matrix_build_tasks)

            # super basic assertion: did matrices we expect get created?
            matrix_directory = os.path.join(temp_dir, 'matrices')
            matrices = [path for path in os.listdir(matrix_directory) if '.csv' in path]
            metadatas = [path for path in os.listdir(matrix_directory) if '.yaml' in path]
            assert len(matrices) == expected_num_matrices
            assert len(metadatas) == expected_num_matrices


def test_integration_simple():
    basic_integration_test(
        state_filters=['state_one OR state_two'],
        feature_group_create_rules={'all': [True]},
        feature_group_mix_rules=['all'],
        expected_num_matrices=4,
    )


def test_integration_more_state_filtering():
    basic_integration_test(
        state_filters=['state_one OR state_two', 'state_one', 'state_two'],
        feature_group_create_rules={'all': [True]},
        feature_group_mix_rules=['all'],
        expected_num_matrices=4*3,  # 4 base, 3 state filters
    )


def test_integration_feature_grouping():
    basic_integration_test(
        state_filters=['state_one OR state_two'],
        feature_group_create_rules={'prefix': ['cat', 'dog']},
        feature_group_mix_rules=['leave-one-out', 'all'],
        expected_num_matrices=4*3,  # 4 base, cat/dog/cat+dog
    )

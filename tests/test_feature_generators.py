from architect.feature_generators import FeatureGenerator
import testing.postgresql
from sqlalchemy import create_engine
from datetime import date
import pandas
from unittest import TestCase
import copy
from collate.spacetime import SpacetimeAggregation
from collate.collate import Aggregate, Categorical


INPUT_DATA = [
    # entity_id, knowledge_date, zip_code, cat_one, quantity_one
    (1, date(2014, 1, 1), '60120', 'good', 10000),
    (1, date(2014, 10, 11), '60120', 'good', None),
    (3, date(2012, 6, 8), '60653', 'bad', 342),
    (3, date(2014, 12, 21), '60653', 'inbetween', 600),
    (4, date(2014, 4, 4), '60653', 'bad', 1236)
]


def setup_db(engine):
    engine.execute("""
        create table data (
            entity_id int,
            knowledge_date date,
            zip_code text,
            cat_one varchar,
            quantity_one float
        )
    """)
    for row in INPUT_DATA:
        engine.execute(
            'insert into data values (%s, %s, %s, %s, %s)',
            row
        )


def test_feature_generation():
    aggregate_config = [{
        'prefix': 'aprefix',
        'aggregates': [
            {'quantity': 'quantity_one', 'metrics': ['sum', 'count']},
        ],
        'categoricals': [
            {
                'column': 'cat_one',
                'choices': ['good', 'bad'],
                'metrics': ['sum']
            },
        ],
        'groups': ['entity_id', 'zip_code'],
        'intervals': ['all'],
        'knowledge_date_column': 'knowledge_date',
        'from_obj': 'data'
    }]

    expected_output = {
        'aprefix_entity_id': [
            {
                'entity_id': 1,
                'as_of_date': date(2014, 9, 30),
                'aprefix_entity_id_all_quantity_one_sum': 10000,
                'aprefix_entity_id_all_quantity_one_count': 1,
                'aprefix_entity_id_all_cat_one_good_sum': 1,
                'aprefix_entity_id_all_cat_one_bad_sum': 0
            },
            {
                'entity_id': 3,
                'as_of_date': date(2013, 9, 30),
                'aprefix_entity_id_all_quantity_one_sum': 342,
                'aprefix_entity_id_all_quantity_one_count': 1,
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1
            },
            {
                'entity_id': 3,
                'as_of_date': date(2014, 9, 30),
                'aprefix_entity_id_all_quantity_one_sum': 342,
                'aprefix_entity_id_all_quantity_one_count': 1,
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1
            },
            {
                'entity_id': 4,
                'as_of_date': date(2014, 9, 30),
                'aprefix_entity_id_all_quantity_one_sum': 1236,
                'aprefix_entity_id_all_quantity_one_count': 1,
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1
            },
        ],
        'aprefix_zip_code': [
            {
                'zip_code': '60120',
                'as_of_date': date(2014, 9, 30),
                'aprefix_zip_code_all_quantity_one_sum': 10000,
                'aprefix_zip_code_all_quantity_one_count': 1,
                'aprefix_zip_code_all_cat_one_good_sum': 1,
                'aprefix_zip_code_all_cat_one_bad_sum': 0
            },
            {
                'zip_code': '60653',
                'as_of_date': date(2013, 9, 30),
                'aprefix_zip_code_all_quantity_one_sum': 342,
                'aprefix_zip_code_all_quantity_one_count': 1,
                'aprefix_zip_code_all_cat_one_good_sum': 0,
                'aprefix_zip_code_all_cat_one_bad_sum': 1
            },
            {
                'zip_code': '60653',
                'as_of_date': date(2014, 9, 30),
                'aprefix_zip_code_all_quantity_one_sum': 1578,
                'aprefix_zip_code_all_quantity_one_count': 2,
                'aprefix_zip_code_all_cat_one_good_sum': 0,
                'aprefix_zip_code_all_cat_one_bad_sum': 2
            }
        ],
        'aprefix_aggregation': [
            {
                'entity_id': 1,
                'as_of_date': date(2014, 9, 30),
                'zip_code': '60120',
                'aprefix_entity_id_all_quantity_one_sum': 10000,
                'aprefix_entity_id_all_quantity_one_count': 1,
                'aprefix_entity_id_all_cat_one_good_sum': 1,
                'aprefix_entity_id_all_cat_one_bad_sum': 0,
                'aprefix_zip_code_all_quantity_one_sum': 10000,
                'aprefix_zip_code_all_quantity_one_count': 1,
                'aprefix_zip_code_all_cat_one_good_sum': 1,
                'aprefix_zip_code_all_cat_one_bad_sum': 0
            },
            {
                'entity_id': 3,
                'as_of_date': date(2013, 9, 30),
                'zip_code': '60653',
                'aprefix_entity_id_all_quantity_one_sum': 342,
                'aprefix_entity_id_all_quantity_one_count': 1,
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1,
                'aprefix_zip_code_all_quantity_one_sum': 342,
                'aprefix_zip_code_all_quantity_one_count': 1,
                'aprefix_zip_code_all_cat_one_good_sum': 0,
                'aprefix_zip_code_all_cat_one_bad_sum': 1
            },
            {
                'entity_id': 3,
                'as_of_date': date(2014, 9, 30),
                'zip_code': '60653',
                'aprefix_entity_id_all_quantity_one_sum': 342,
                'aprefix_entity_id_all_quantity_one_count': 1,
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1,
                'aprefix_zip_code_all_quantity_one_sum': 1578,
                'aprefix_zip_code_all_quantity_one_count': 2,
                'aprefix_zip_code_all_cat_one_good_sum': 0,
                'aprefix_zip_code_all_cat_one_bad_sum': 2
            },
            {
                'entity_id': 4,
                'as_of_date': date(2014, 9, 30),
                'zip_code': '60653',
                'aprefix_entity_id_all_quantity_one_sum': 1236,
                'aprefix_entity_id_all_quantity_one_count': 1,
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1,
                'aprefix_zip_code_all_quantity_one_sum': 1578,
                'aprefix_zip_code_all_quantity_one_count': 2,
                'aprefix_zip_code_all_cat_one_good_sum': 0,
                'aprefix_zip_code_all_cat_one_bad_sum': 2
            },
        ]
    }

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        setup_db(engine)

        features_schema_name = 'features'
        output_tables = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name
        ).create_all_tables(
            feature_dates=['2013-09-30', '2014-09-30'],
            feature_aggregation_config=aggregate_config,
        )

        for output_table in output_tables:
            records = pandas.read_sql(
                'select * from {}.{} order by 1, 2'
                .format(features_schema_name, output_table),
                engine
            ).to_dict('records')
            for record, expected_record in zip(records, expected_output[output_table]):
                assert record == expected_record


def test_index_column_lookup():
    aggregations = [
        SpacetimeAggregation(
            prefix='prefix1',
            aggregates=[Categorical(col='cat_one', function='sum', choices=['good', 'bad', 'inbetween'])],
            groups=['entity_id'],
            intervals=['all'],
            date_column='knowledge_date',
            output_date_column='as_of_date',
            dates=['2013-09-30', '2014-09-30'],
            schema='features',
            from_obj='data'
        ),
        SpacetimeAggregation(
            prefix='prefix2',
            aggregates=[Aggregate(quantity='quantity_one', function='count')],
            groups=['entity_id', 'zip_code'],
            intervals=['all'],
            date_column='knowledge_date',
            output_date_column='as_of_date',
            dates=['2013-09-30', '2014-09-30'],
            schema='features',
            from_obj='data'
        )
    ]
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        setup_db(engine)

        features_schema_name = 'features'
        feature_generator = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name
        )
        lookup = feature_generator.index_column_lookup(aggregations)
        assert lookup == {
            'prefix1_aggregation': ['as_of_date', 'entity_id',],
            'prefix2_aggregation': ['as_of_date', 'entity_id', 'zip_code']
        }


def test_feature_generation_beginning_of_time():
    aggregate_config = [{
        'prefix': 'aprefix',
        'aggregates': [
            {'quantity': 'quantity_one', 'metrics': ['sum']},
        ],
        'groups': ['entity_id'],
        'intervals': ['all'],
        'knowledge_date_column': 'knowledge_date',
        'from_obj': 'data'
    }]

    expected_output = {
        'aprefix_entity_id': [
            {
                'entity_id': 1,
                'as_of_date': date(2015, 1, 1),
                'aprefix_entity_id_all_quantity_one_sum': 10000,
            },
            {
                'entity_id': 3,
                'as_of_date': date(2015, 1, 1),
                'aprefix_entity_id_all_quantity_one_sum': 600,
            },
            {
                'entity_id': 4,
                'as_of_date': date(2015, 1, 1),
                'aprefix_entity_id_all_quantity_one_sum': 1236,
            },
        ]
    }
    expected_output['aprefix_aggregation'] = expected_output['aprefix_entity_id']

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        setup_db(engine)

        features_schema_name = 'features'
        output_tables = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name,
            beginning_of_time='2013-01-01',
        ).create_all_tables(
            feature_dates=['2015-01-01'],
            feature_aggregation_config=aggregate_config,
        )

        for output_table in output_tables:
            records = pandas.read_sql(
                'select * from {}.{} order by as_of_date, entity_id'
                .format(features_schema_name, output_table),
                engine
            ).to_dict('records')
            assert records == expected_output[output_table]


def test_dynamic_categoricals():
    aggregate_config = [{
        'prefix': 'aprefix',
        'categoricals': [
            {
                'column': 'cat_one',
                'choice_query': 'select distinct(cat_one) from data',
                'metrics': ['sum']
            },
        ],
        'groups': ['entity_id'],
        'intervals': ['all'],
        'knowledge_date_column': 'knowledge_date',
        'from_obj': 'data'
    }]
    expected_output = {
        'aprefix_entity_id': [
            {
                'entity_id': 3,
                'as_of_date': date(2013, 9, 30),
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_inbetween_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1
            },
            {
                'entity_id': 1,
                'as_of_date': date(2014, 9, 30),
                'aprefix_entity_id_all_cat_one_good_sum': 1,
                'aprefix_entity_id_all_cat_one_inbetween_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 0
            },
            {
                'entity_id': 3,
                'as_of_date': date(2014, 9, 30),
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_inbetween_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1
            },
            {
                'entity_id': 4,
                'as_of_date': date(2014, 9, 30),
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_inbetween_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1
            },
        ]
    }
    expected_output['aprefix_aggregation'] = expected_output['aprefix_entity_id']

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        setup_db(engine)

        features_schema_name = 'features'

        output_tables = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name
        ).create_all_tables(
            feature_dates=['2013-09-30', '2014-09-30'],
            feature_aggregation_config=aggregate_config,
        )

        for output_table in output_tables:
            records = pandas.read_sql(
                'select * from {}.{} order by as_of_date, entity_id'.format(
                    features_schema_name, output_table
                ),
                engine
            ).to_dict('records')
            assert records == expected_output[output_table]


def test_array_categoricals():
    aggregate_config = [{
        'prefix': 'aprefix',
        'array_categoricals': [
            {
                'column': 'cat_one',
                'choices': ['good', 'bad', 'inbetween'],
                'metrics': ['sum']
            },
        ],
        'groups': ['entity_id'],
        'intervals': ['all'],
        'knowledge_date_column': 'knowledge_date',
        'from_obj': 'data'
    }]
    expected_output = {
        'aprefix_entity_id': [
            {
                'entity_id': 3,
                'as_of_date': date(2013, 9, 30),
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_inbetween_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1
            },
            {
                'entity_id': 1,
                'as_of_date': date(2014, 9, 30),
                'aprefix_entity_id_all_cat_one_good_sum': 1,
                'aprefix_entity_id_all_cat_one_inbetween_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 0
            },
            {
                'entity_id': 3,
                'as_of_date': date(2014, 9, 30),
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_inbetween_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1
            },
            {
                'entity_id': 4,
                'as_of_date': date(2014, 9, 30),
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_inbetween_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1
            },
        ]
    }
    expected_output['aprefix_aggregation'] = expected_output['aprefix_entity_id']

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        input_data = [
            # entity_id, knowledge_date, cat_one, quantity_one
            (1, date(2014, 1, 1), ['good', 'good'], 10000),
            (1, date(2014, 10, 11), ['good'], None),
            (3, date(2012, 6, 8), ['bad'], 342),
            (3, date(2014, 12, 21), ['inbetween'], 600),
            (4, date(2014, 4, 4), ['bad'], 1236)
        ]

        engine.execute("""
            create table data (
                entity_id int,
                knowledge_date date,
                cat_one varchar[],
                quantity_one float
            )
        """)
        for row in input_data:
            engine.execute(
                'insert into data values (%s, %s, %s, %s)',
                row
            )

        features_schema_name = 'features'

        output_tables = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name
        ).create_all_tables(
            feature_dates=['2013-09-30', '2014-09-30'],
            feature_aggregation_config=aggregate_config,
        )

        for output_table in output_tables:
            records = pandas.read_sql(
                'select * from {}.{} order by as_of_date, entity_id'.format(
                    features_schema_name, output_table
                ),
                engine
            ).to_dict('records')
            assert records == expected_output[output_table]


def test_generate_table_tasks():
    aggregations = [
        SpacetimeAggregation(
            prefix='prefix1',
            aggregates=[Categorical(col='cat_one', function='sum', choices=['good', 'bad', 'inbetween'])],
            groups=['entity_id'],
            intervals=['all'],
            date_column='knowledge_date',
            output_date_column='as_of_date',
            dates=['2013-09-30', '2014-09-30'],
            schema='features',
            from_obj='data'
        ),
        SpacetimeAggregation(
            prefix='prefix2',
            aggregates=[Aggregate(quantity='quantity_one', function='count')],
            groups=['entity_id'],
            intervals=['all'],
            date_column='knowledge_date',
            output_date_column='as_of_date',
            dates=['2013-09-30', '2014-09-30'],
            schema='features',
            from_obj='data'
        )
    ]
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        setup_db(engine)

        features_schema_name = 'features'

        table_tasks = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name
        ).generate_all_table_tasks(aggregations)
        for table_name, task in table_tasks.items():
            assert 'DROP TABLE' in task['prepare'][0]
            assert 'CREATE TABLE' in str(task['prepare'][1])
            assert 'CREATE INDEX' in task['finalize'][0]
            assert isinstance(task['inserts'], list)


def test_aggregations():
    aggregate_config = [{
        'prefix': 'prefix1',
        'categoricals': [
            {
                'column': 'cat_one',
                'choice_query': 'select distinct(cat_one) from data',
                'metrics': ['sum']
            },
        ],
        'groups': ['entity_id'],
        'intervals': ['all'],
        'knowledge_date_column': 'knowledge_date',
        'from_obj': 'data'
    }, {
        'prefix': 'prefix2',
        'aggregates': [
            {'quantity': 'quantity_one', 'metrics': ['count']},
        ],
        'groups': ['entity_id'],
        'intervals': ['all'],
        'knowledge_date_column': 'knowledge_date',
        'from_obj': 'data'
    }]
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        setup_db(engine)

        features_schema_name = 'features'

        aggregations = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name
        ).aggregations(
            feature_dates=['2013-09-30', '2014-09-30'],
            feature_aggregation_config=aggregate_config,
        )
        for aggregation in aggregations:
            assert isinstance(aggregation, SpacetimeAggregation)

def test_replace():
    aggregate_config = [{
        'prefix': 'aprefix',
        'aggregates': [
            {'quantity': 'quantity_one', 'metrics': ['sum', 'count']},
        ],
        'categoricals': [
            {
                'column': 'cat_one',
                'choices': ['good', 'bad'],
                'metrics': ['sum']
            },
        ],
        'groups': ['entity_id'],
        'intervals': ['all'],
        'knowledge_date_column': 'knowledge_date',
        'from_obj': 'data'
    }]

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        setup_db(engine)

        features_schema_name = 'features'
        feature_tables = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name,
            replace=False
        ).create_all_tables(
            feature_dates=['2013-09-30', '2014-09-30'],
            feature_aggregation_config=aggregate_config,
        )

        assert len(feature_tables) == 2

        feature_generator = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name,
            replace=False
        )
        aggregations = feature_generator.aggregations(
            feature_dates=['2013-09-30', '2014-09-30'],
            feature_aggregation_config=aggregate_config,
        )
        table_tasks = feature_generator.generate_all_table_tasks(aggregations)

        assert len(table_tasks['aprefix_entity_id'].keys()) == 0


class TestValidations(TestCase):
    def setUp(self):
        self.postgresql = testing.postgresql.Postgresql()
        engine = create_engine(self.postgresql.url())
        setup_db(engine)
        self.feature_generator = FeatureGenerator(engine, 'features')

        self.base_config = {
            'prefix': 'aprefix',
            'categoricals': [
                {
                    'column': 'cat_one',
                    'choices': ['good', 'bad'],
                    'metrics': ['sum']
                },
            ],
            'groups': ['entity_id'],
            'intervals': ['all'],
            'knowledge_date_column': 'knowledge_date',
            'from_obj': 'data'
        }

    def tearDown(self):
        self.postgresql.stop()

    def test_correct_keys(self):
        self.feature_generator.validate([self.base_config])

        with self.assertRaises(ValueError):
            no_group = copy.deepcopy(self.base_config)
            del no_group['groups']
            self.feature_generator.validate([no_group])

        with self.assertRaises(ValueError):
            no_intervals = copy.deepcopy(self.base_config)
            del no_intervals['intervals']
            self.feature_generator.validate([no_intervals])

        with self.assertRaises(ValueError):
            no_kdate = copy.deepcopy(self.base_config)
            del no_kdate['knowledge_date_column']
            self.feature_generator.validate([no_kdate])

        with self.assertRaises(ValueError):
            no_from_obj = copy.deepcopy(self.base_config)
            del no_from_obj['from_obj']
            self.feature_generator.validate([no_from_obj])

        with self.assertRaises(ValueError):
            no_aggs = copy.deepcopy(self.base_config)
            del no_aggs['categoricals']
            self.feature_generator.validate([no_aggs])

    def test_bad_from_obj(self):
        bad_from_obj = copy.deepcopy(self.base_config)
        bad_from_obj['from_obj'] = 'where thing is other_thing'
        with self.assertRaises(ValueError):
            self.feature_generator.validate([bad_from_obj])

    def test_bad_interval(self):
        bad_interval = copy.deepcopy(self.base_config)
        bad_interval['intervals'] = ['1y', '1fortnight']
        with self.assertRaises(ValueError):
            self.feature_generator.validate([bad_interval])

    def test_bad_group(self):
        bad_group = copy.deepcopy(self.base_config)
        bad_group['groups'] = ['entity_id', 'otherthing']
        with self.assertRaises(ValueError):
            self.feature_generator.validate([bad_group])

    def test_bad_choice_query(self):
        bad_choice_query = copy.deepcopy(self.base_config)
        del bad_choice_query['categoricals'][0]['choices']
        bad_choice_query['categoricals'][0]['choice_query'] = 'select distinct cat_two from data'
        with self.assertRaises(ValueError):
            self.feature_generator.validate([bad_choice_query])

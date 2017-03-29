from triage.feature_generators import FeatureGenerator
import testing.postgresql
from sqlalchemy import create_engine
from datetime import date
import pandas


INPUT_DATA = [
    # entity_id, knowledge_date, cat_one, quantity_one
    (1, date(2014, 1, 1), 'good', 10000),
    (1, date(2014, 10, 11), 'good', None),
    (3, date(2012, 6, 8), 'bad', 342),
    (3, date(2014, 12, 21), 'inbetween', 600),
    (4, date(2014, 4, 4), 'bad', 1236)
]


def setup_db(engine):
    engine.execute("""
        create table data (
            entity_id int,
            knowledge_date date,
            cat_one varchar,
            quantity_one float
        )
    """)
    for row in INPUT_DATA:
        engine.execute(
            'insert into data values (%s, %s, %s, %s)',
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
                'aprefix_entity_id_all_quantity_one_sum': 342,
                'aprefix_entity_id_all_quantity_one_count': 1,
                'aprefix_entity_id_all_cat_one_good_sum': 0,
                'aprefix_entity_id_all_cat_one_bad_sum': 1
            },
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
        ]

    }

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        setup_db(engine)

        features_schema_name = 'features'
        output_tables = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name
        ).generate(
            feature_dates=['2013-09-30', '2014-09-30'],
            feature_aggregations=aggregate_config,
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

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        setup_db(engine)

        features_schema_name = 'features'

        output_tables = FeatureGenerator(
            db_engine=engine,
            features_schema_name=features_schema_name
        ).generate(
            feature_dates=['2013-09-30', '2014-09-30'],
            feature_aggregations=aggregate_config,
        )

        for output_table in output_tables:
            records = pandas.read_sql(
                'select * from {}.{} order by as_of_date, entity_id'.format(
                    features_schema_name, output_table
                ),
                engine
            ).to_dict('records')
            assert records == expected_output[output_table]

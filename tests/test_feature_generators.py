from triage.feature_generators import FeatureGenerator
import testing.postgresql
from sqlalchemy import create_engine
from datetime import date
import pandas


input_data = [
    # entity_id, knowledge_date, cat_one, quantity_one
    (1, date(2014, 1, 1), 'good', 10000),
    (1, date(2014, 10, 11), 'good', None),
    (3, date(2012, 6, 8), 'bad', 342),
    (3, date(2014, 12, 21), 'inbetween', 600),
    (4, date(2014, 4, 4), 'bad', 1236)
]

aggregate_config = [{
    'prefix': 'aprefix',
    'aggregates': [
        {'quantity': 'quantity_one', 'metrics': ['sum', 'count']},
    ],
    'categoricals': [
        {'column': 'cat_one', 'choices': ['good', 'bad'], 'metrics': ['sum']},
    ],
    'groups': ['entity_id'],
    'intervals': ['all'],
    'knowledge_date_column': 'knowledge_date',
    'from_obj': 'data'
}]

expected_output = {
    '"aprefix_aggregation"': [
        {
            'entity_id': 3,
            'date': date(2013, 9, 30),
            'aprefix_entity_id_all_quantity_one_sum': 342,
            'aprefix_entity_id_all_quantity_one_count': 1,
            'aprefix_entity_id_all_cat_one_good_sum': 0,
            'aprefix_entity_id_all_cat_one_bad_sum': 1
        },
        {
            'entity_id': 1,
            'date': date(2014, 9, 30),
            'aprefix_entity_id_all_quantity_one_sum': 10000,
            'aprefix_entity_id_all_quantity_one_count': 1,
            'aprefix_entity_id_all_cat_one_good_sum': 1,
            'aprefix_entity_id_all_cat_one_bad_sum': 0
        },
        {
            'entity_id': 3,
            'date': date(2014, 9, 30),
            'aprefix_entity_id_all_quantity_one_sum': 342,
            'aprefix_entity_id_all_quantity_one_count': 1,
            'aprefix_entity_id_all_cat_one_good_sum': 0,
            'aprefix_entity_id_all_cat_one_bad_sum': 1
        },
        {
            'entity_id': 4,
            'date': date(2014, 9, 30),
            'aprefix_entity_id_all_quantity_one_sum': 1236,
            'aprefix_entity_id_all_quantity_one_count': 1,
            'aprefix_entity_id_all_cat_one_good_sum': 0,
            'aprefix_entity_id_all_cat_one_bad_sum': 1
        },
    ]

}


def test_training_label_generation():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())

        engine.execute("""
            create table data (
                entity_id int,
                knowledge_date date,
                cat_one varchar,
                quantity_one float
            )
        """)
        for row in input_data:
            engine.execute(
                'insert into data values (%s, %s, %s, %s)',
                row
            )

        output_tables = FeatureGenerator(
            db_engine=engine,
        ).generate(
            feature_dates=['2013-09-30', '2014-09-30'],
            feature_aggregations=aggregate_config,
        )

        for output_table in output_tables:
            records = pandas.read_sql(
                'select * from {} order by date, entity_id'.format(output_table),
                engine
            ).to_dict('records')
            assert records == expected_output[output_table]

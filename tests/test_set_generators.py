from triage.set_generators import SetGenerator
import testing.postgresql
from sqlalchemy import create_engine
from datetime import date
import pandas
import boto
from moto import mock_s3
from config import config
import tempfile


def path_to_list(bucket, output_path):
    with tempfile.NamedTemporaryFile() as f:
        boto.s3.key.Key(bucket=bucket, name=output_path).get_contents_to_file(f)
        f.seek(0)
        return list(pandas.read_csv(f.name).fillna('n/a').itertuples(index=False))


def test_set_generation():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())

        engine.execute(
            'create table features (entity_id int, date date, quantity_one_sum float, quantity_one_count int, quantity_two_count int, quantity_two_min float)'
        )

        features_data = [
            # entity_id, date, quantity_one_sum, quantity_one_count, quantity_two_count, quantity_two_min
            (3, date(2013, 9, 30), 342, 1, 1, 9234),
            (1, date(2014, 9, 30), 10000, 1, 0, None),
            (3, date(2014, 9, 30), 342, 1, 1, 9234),
            (4, date(2014, 9, 30), 1236, 1, 1, 6270),
        ]

        for row in features_data:
            engine.execute(
                'insert into features values (%s, %s, %s, %s, %s, %s)',
                row
            )

        engine.execute(
            'create table labels (entity_id int, outcome_date date, outcome bool)'
        )

        labels_data = [
            # entity_id, outcome_date, outcome
            (1, date(2014, 11, 10), False),
            (1, date(2015, 1, 1), False),
            (2, date(2015, 6, 8), False),
            (3, date(2015, 3, 3), True),
            (3, date(2015, 7, 24), False),
            (4, date(2014, 12, 13), False),
        ]

        for row in labels_data:
            engine.execute(
                'insert into labels values (%s, %s, %s)',
                row
            )

        with mock_s3():
            s3_conn = boto.connect_s3()
            s3_conn.create_bucket(config['shared_bucket'])
            bucket = s3_conn.get_bucket(config['shared_bucket'])

            generator = SetGenerator(
                features_table='features',
                labels_table='labels',
                db_engine=engine,
                s3_conn=s3_conn
            )
            first_year_path = generator.generate(
                start_date='2013-01-01',
                end_date='2014-01-01'
            )

            assert path_to_list(bucket, first_year_path) == [
                (3, '2013-09-30', 342, 1, 1, 9234, 'n/a'),
                (1, '2014-09-30', 10000, 1, 0, 'n/a', 'n/a'),
                (3, '2014-09-30', 342, 1, 1, 9234, 'n/a'),
                (4, '2014-09-30', 1236, 1, 1, 6270, 'n/a')
            ]

            second_year_path = generator.generate(
                start_date='2014-01-01',
                end_date='2015-01-01'
            )
            assert path_to_list(bucket, second_year_path) == [
                (3, '2013-09-30', 342, 1, 1, 9234, 'n/a'),
                (1, '2014-09-30', 10000, 1, 0, 'n/a', 'f'),
                (1, '2014-09-30', 10000, 1, 0, 'n/a', 'f'),
                (3, '2014-09-30', 342, 1, 1, 9234, 'n/a'),
                (4, '2014-09-30', 1236, 1, 1, 6270, 'f')
            ]

from architect.features import FeatureDictionaryCreator
import testing.postgresql
from sqlalchemy import create_engine


def test_feature_dictionary_creator():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        engine.execute('create schema features')
        engine.execute('''
            create table features.feature_table_one (
                entity_id int,
                as_of_date date,
                feature_one float,
                feature_two float
            )
        ''')
        engine.execute('''
            create table features.feature_table_two (
                entity_id int,
                as_of_date date,
                feature_three float,
                feature_four float
            )
        ''')
        engine.execute('''
            create table features.random_other_table (
                another_column float
            )
        ''')

        creator = FeatureDictionaryCreator(
            features_schema_name='features',
            db_engine=engine
        )
        feature_dictionary = creator.feature_dictionary(
            ['feature_table_one', 'feature_table_two']
        )
        assert feature_dictionary == {
            'feature_table_one': ['feature_one', 'feature_two'],
            'feature_table_two': ['feature_three', 'feature_four'],
        }

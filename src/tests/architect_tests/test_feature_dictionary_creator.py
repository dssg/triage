import testing.postgresql

from sqlalchemy.sql.expression import text 
from triage import create_engine
from triage.component.architect.features import FeatureDictionaryCreator


def test_feature_dictionary_creator():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        with engine.begin() as conn:
            conn.execute(text("create schema features"))
            conn.execute(
                text(
                    """
                    create table features.prefix1_entity_id (
                        entity_id int,
                        as_of_date date,
                        feature_one float,
                        feature_two float
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    create table features.prefix1_zipcode (
                        zipcode text,
                        as_of_date date,
                        feature_three float,
                        feature_four float
                    )
                    """
                )
            )
            conn.execute(
                text(
                """
                create table features.prefix1_aggregation (
                    entity_id int,
                    as_of_date date,
                    zipcode text,
                    feature_one float,
                    feature_two float,
                    feature_three float,
                    feature_four float
                )
                """
                )
            )
            conn.execute(
                text(
                    """
                    create table features.prefix1_aggregation_imputed (
                        entity_id int,
                        as_of_date date,
                        zipcode text,
                        feature_one float,
                        feature_two float,
                        feature_three float,
                        feature_three_imp int,
                        feature_four float
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    create table features.random_other_table (
                        another_column float
                    )
                    """
                )
            )

        creator = FeatureDictionaryCreator(
            features_schema_name="features", db_engine=engine
        )
        feature_dictionary = creator.feature_dictionary(
            feature_table_names=[
                "prefix1_entity_id",
                "prefix1_zip_code",
                "prefix1_aggregation",
                "prefix1_aggregation_imputed",
            ],
            index_column_lookup={
                "prefix1_aggregation_imputed": ["entity_id", "zipcode", "as_of_date"]
            },
        )
        assert feature_dictionary == {
            "prefix1_aggregation_imputed": [
                "feature_one",
                "feature_two",
                "feature_three",
                "feature_three_imp",
                "feature_four",
            ]
        }

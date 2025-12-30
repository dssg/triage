from datetime import datetime, date

from sqlalchemy import text
from unittest.mock import MagicMock

from triage.component.catwalk.protected_groups_generators import ProtectedGroupsGenerator


def create_demographics_table(db_engine, data):
    with db_engine.connect() as conn:
        conn.execute(text(
            """drop table if exists demographics;
            create table demographics (person_id int, event_date date, race text, sex text, age_bucket int)
            """
        ))
        for event in data:
            conn.execute(
                text("insert into demographics values (:p1, :p2, :p3, :p4, :p5)"),
                {"p1": event[0], "p2": event[1], "p3": event[2], "p4": event[3], "p5": event[4]}
            )
        conn.commit()


def create_cohort_table(db_engine, data):
    with db_engine.connect() as conn:
        conn.execute(text(
            "create table cohort_abcdef (entity_id int, as_of_date timestamp)"
        ))
        for event in data:
            conn.execute(
                text("insert into cohort_abcdef values (:entity_id, :as_of_date)"),
                {"entity_id": event[0], "as_of_date": event[1]}
            )
        conn.commit()


def default_demographics():
    return [
        (1, datetime(2015, 12, 30), 'aa', 'male', 1),
        (1, datetime(2016, 2, 1), 'aa', 'male', 1),
        (1, datetime(2016, 3, 1), 'aa', 'female', 1),
        (2, datetime(2015, 12, 30), 'wh', 'male', 3),
        (2, datetime(2016, 3, 1), 'wh', 'male', 3),
        (3, datetime(2015, 12, 30), 'aa', 'male', 1),
        (3, datetime(2016, 3, 1), 'aa', 'male', 1),
        (5, datetime(2016, 2, 1), 'wh', 'female', 2),
        (5, datetime(2016, 3, 1), 'wh', 'female', 2),
    ]


def default_cohort():
    return [
        (1, datetime(2016, 1, 1)),
        (1, datetime(2016, 3, 1)),
        (1, datetime(2016, 4, 1)),
        (2, datetime(2016, 1, 1)),
        (2, datetime(2016, 3, 1)),
        (2, datetime(2016, 4, 1)),
        (3, datetime(2016, 1, 1)),
        (3, datetime(2016, 3, 1)),
        (3, datetime(2016, 4, 1)),
        (4, datetime(2016, 1, 1)),
        (4, datetime(2016, 3, 1)),
        (4, datetime(2016, 4, 1)),
        (5, datetime(2016, 1, 1)),
        (5, datetime(2016, 3, 1)),
        (5, datetime(2016, 4, 1)),
    ]


def assert_data(table_generator):
    expected_output = [
        (1, date(2016, 1, 1), 'aa', 'male', '1', 'abcdef'),
        (1, date(2016, 3, 1), 'aa', 'male', '1', 'abcdef'),
        (1, date(2016, 4, 1), 'aa', 'female', '1', 'abcdef'),
        (2, date(2016, 1, 1), 'wh', 'male', '3', 'abcdef'),
        (2, date(2016, 3, 1), 'wh', 'male', '3', 'abcdef'),
        (2, date(2016, 4, 1), 'wh', 'male', '3', 'abcdef'),
        (3, date(2016, 1, 1), 'aa', 'male', '1', 'abcdef'),
        (3, date(2016, 3, 1), 'aa', 'male', '1', 'abcdef'),
        (3, date(2016, 4, 1), 'aa', 'male', '1', 'abcdef'),
        (4, date(2016, 1, 1), None, None, None, 'abcdef'),
        (4, date(2016, 3, 1), None, None, None, 'abcdef'),
        (4, date(2016, 4, 1), None, None, None, 'abcdef'),
        (5, date(2016, 1, 1), None, None, None, 'abcdef'),
        (5, date(2016, 3, 1), 'wh', 'female', '2', 'abcdef'),
        (5, date(2016, 4, 1), 'wh', 'female', '2', 'abcdef'),
    ]
    with table_generator.db_engine.connect() as conn:
        results = list(
            conn.execute(
                text(f"""
                select entity_id, as_of_date, race, sex, age_bucket, cohort_hash
                from {table_generator.protected_groups_table_name}
                order by entity_id, as_of_date
            """)
            )
        )
    assert results == expected_output


def test_protected_groups_generator_replace(db_engine):
    demographics_data = default_demographics()
    cohort_data = default_cohort()
    create_demographics_table(db_engine, demographics_data)
    create_cohort_table(db_engine, cohort_data)
    table_generator = ProtectedGroupsGenerator(
        from_obj="demographics",
        attribute_columns=['race', 'sex', 'age_bucket'],
        entity_id_column="person_id",
        knowledge_date_column="event_date",
        db_engine=db_engine,
        protected_groups_table_name="protected_groups_abcdef",
        replace=True
    )
    as_of_dates = [
        datetime(2016, 1, 1),
        datetime(2016, 3, 1),
        datetime(2016, 4, 1),
    ]
    table_generator.generate_all_dates(
        as_of_dates,
        cohort_table_name='cohort_abcdef',
        cohort_hash='abcdef'
    )
    assert_data(table_generator)

    table_generator.generate_all_dates(
        as_of_dates,
        cohort_table_name='cohort_abcdef',
        cohort_hash='abcdef'
    )
    assert_data(table_generator)


def test_protected_groups_generator_noreplace(db_engine):
    demographics_data = default_demographics()
    cohort_data = default_cohort()
    create_demographics_table(db_engine, demographics_data)
    create_cohort_table(db_engine, cohort_data)
    table_generator = ProtectedGroupsGenerator(
        from_obj="demographics",
        attribute_columns=['race', 'sex', 'age_bucket'],
        entity_id_column="person_id",
        knowledge_date_column="event_date",
        db_engine=db_engine,
        protected_groups_table_name="protected_groups_abcdef",
        replace=False
    )
    as_of_dates = [
        datetime(2016, 1, 1),
        datetime(2016, 3, 1),
        datetime(2016, 4, 1),
    ]
    table_generator.generate_all_dates(
        as_of_dates,
        cohort_table_name='cohort_abcdef',
        cohort_hash='abcdef'
    )
    assert_data(table_generator)
    table_generator.generate = MagicMock()
    table_generator.generate_all_dates(
        as_of_dates,
        cohort_table_name='cohort_abcdef',
        cohort_hash='abcdef'
    )
    table_generator.generate.assert_not_called()
    assert_data(table_generator)


def test_as_dataframe(db_engine):
    attribute_columns = ['race', 'sex', 'age_bucket']
    demographics_data = default_demographics()
    cohort_data = default_cohort()
    create_demographics_table(db_engine, demographics_data)
    create_cohort_table(db_engine, cohort_data)
    table_generator = ProtectedGroupsGenerator(
        from_obj="demographics",
        attribute_columns=attribute_columns,
        entity_id_column="person_id",
        knowledge_date_column="event_date",
        db_engine=db_engine,
        protected_groups_table_name="protected_groups_abcdef",
        replace=True
    )
    as_of_dates = [
        datetime(2016, 1, 1),
        datetime(2016, 3, 1),
        datetime(2016, 4, 1),
    ]
    table_generator.generate_all_dates(
        as_of_dates,
        cohort_table_name='cohort_abcdef',
        cohort_hash='abcdef'
    )
    protected_df = table_generator.as_dataframe(
        as_of_dates,
        cohort_hash='abcdef'
    )
    assert(protected_df.shape == (15, 3))
    assert(set(attribute_columns).issubset(protected_df.columns))
    for attr_col in attribute_columns:
        assert(protected_df[attr_col].dtype == 'object')
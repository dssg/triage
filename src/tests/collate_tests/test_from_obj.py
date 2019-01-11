from datetime import date
from itertools import product
import sqlalchemy
import testing.postgresql
from triage.component.collate import FromObj
from triage.database_reflection import table_exists
import pytest


events_data = [
    # entity id, event_date, outcome
    [1, date(2014, 1, 1), True],
    [1, date(2014, 11, 10), False],
    [1, date(2015, 1, 1), False],
    [1, date(2015, 11, 10), True],
    [2, date(2013, 6, 8), True],
    [2, date(2014, 6, 8), False],
    [3, date(2014, 3, 3), False],
    [3, date(2014, 7, 24), False],
    [3, date(2015, 3, 3), True],
    [3, date(2015, 7, 24), False],
    [4, date(2015, 12, 13), False],
    [4, date(2016, 12, 13), True],
]

# distinct entity_id, event_date pairs
state_data = sorted(
    product(
        set([l[0] for l in events_data]),
        set([l[1] for l in events_data] + [date(2016, 1, 1)]),
    )
)


def test_materialized_from_obj_create():
    materialized_query = FromObj(
        from_obj='events where event_date < "2016-01-01"',
        name="myquery",
        knowledge_date_column='knowledge_date'
    )
    assert materialized_query.create_materialized_table_sql == 'create table myquery_from_obj as ' +\
        '(select * from events where event_date < "2016-01-01")'

def test_materialized_from_obj_index():
    materialized_query = FromObj(
        from_obj='events where event_date < "2016-01-01"',
        name="myquery",
        knowledge_date_column='knowledge_date'
    )
    assert materialized_query.index_materialized_table_sql == 'create index on myquery_from_obj (knowledge_date)'

def test_materialized_from_obj_drop():
    materialized_query = FromObj(
        from_obj='events where event_date < "2016-01-01"',
        name="myquery",
        knowledge_date_column='knowledge_date'
    )
    assert materialized_query.drop_materialized_table_sql == 'drop table if exists myquery_from_obj'


@pytest.fixture(name="db_engine_with_events_table", scope='function')
def db_engine_with_events_table(db_engine):
    db_engine.execute(
        "create table events (entity_id int, event_date date, outcome bool)"
    )
    for event in events_data:
        db_engine.execute("insert into events values (%s, %s, %s::bool)", event)
    return db_engine


def test_materialized_from_obj_validate_needs_entity_id(db_engine_with_events_table):
    from_obj = FromObj(
        from_obj="(select event_date from events where event_date < '2016-01-01') from_obj",
        name="myquery",
        knowledge_date_column='event_date'
    )
    db_engine_with_events_table.execute(from_obj.create_materialized_table_sql)
    with pytest.raises(ValueError):
        from_obj.validate(db_engine_with_events_table)


def test_materialized_from_obj_validate_needs_knowledge_date(db_engine_with_events_table):
    from_obj = FromObj(
        from_obj="(select entity_id from events where event_date < '2016-01-01') from_obj",
        name="myquery",
        knowledge_date_column='event_date'
    )
    db_engine_with_events_table.execute(from_obj.create_materialized_table_sql)
    with pytest.raises(ValueError):
        from_obj.validate(db_engine_with_events_table)


def test_materialized_from_obj_validate_success(db_engine_with_events_table):
    from_obj = FromObj(
        from_obj="events where event_date < '2016-01-01'",
        name="myquery",
        knowledge_date_column='event_date'
    )
    db_engine_with_events_table.execute(from_obj.create_materialized_table_sql)
    from_obj.validate(db_engine_with_events_table)


def test_materialized_from_obj_should_not_materialize_tbl():
    from_obj = FromObj(from_obj="mytable1", name="events", knowledge_date_column="date")
    assert not from_obj.should_materialize()
    assert from_obj.table == "mytable1"

def test_materialized_from_obj_should_not_materialize_tbl_with_alias():
    from_obj = FromObj(from_obj="mytable1 as mt1", name="events", knowledge_date_column="date")
    assert not from_obj.should_materialize()
    assert from_obj.table == "mytable1 as mt1"

def test_materialized_from_obj_should_not_materialize_join():
    from_obj = FromObj(from_obj="mytable1 join entities using(entity_id)", name="events", knowledge_date_column="date")
    assert not from_obj.should_materialize()
    assert from_obj.table == "mytable1 join entities using(entity_id)"

def test_materialized_from_obj_should_materialize_subquery():
    from_obj = FromObj(from_obj="(select entity_id, date from mytable1 join entities using(entity_id)) joined_events", name="events", knowledge_date_column="date")
    assert from_obj.should_materialize()
    assert from_obj.table == "events_from_obj"

def test_materialized_from_obj_should_handle_leading_whitespace():
    q = """    (
      SELECT entity_id, date
      from mytable1
      join entities using (entity_id)
    ) AS joined_events"""
    from_obj = FromObj(from_obj=q, name="events", knowledge_date_column="date")
    assert from_obj.should_materialize()
    assert from_obj.table == "events_from_obj"

def test_materialized_from_obj_should_handle_keywords():
    from_obj = FromObj(from_obj="events", name="events", knowledge_date_column="date")
    assert not from_obj.should_materialize()
    assert from_obj.table == "events"


def test_materialized_from_obj_maybe_materialize(db_engine_with_events_table):
    from_obj = FromObj(
        from_obj="events", 
        name="myquery",
        knowledge_date_column='event_date'
    )
    from_obj.should_materialize = lambda: True
    from_obj.maybe_materialize(db_engine_with_events_table)
    assert table_exists(from_obj.table, db_engine_with_events_table)

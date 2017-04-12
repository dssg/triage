#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_spacetime
----------------------------------

Unit tests for `collate.spacetime` module.
"""

import pytest
from collate.collate import Aggregate
from collate.spacetime import SpacetimeAggregation

import sqlalchemy
import testing.postgresql
from datetime import date

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


def test_basic_spacetime():
    with testing.postgresql.Postgresql() as psql:
        engine = sqlalchemy.create_engine(psql.url())
        engine.execute(
            'create table events (entity_id int, date date, outcome bool)'
        )
        for event in events_data:
            engine.execute(
                'insert into events values (%s, %s, %s::bool)',
                event
            )
        
        st = SpacetimeAggregation([Aggregate('outcome::int',['sum','avg'])],
            from_obj = 'events',
            groups = ['entity_id'],
            intervals = ['1y', '2y', 'all'],
            dates = ['2016-01-01', '2015-01-01'],
            date_column = '"date"')

        st.execute(engine.connect())
        
        r = engine.execute('select * from events_entity_id order by entity_id, date')
        rows = [x for x in r]
        assert rows[0]['entity_id'] == 1
        assert rows[0]['date'] == date(2015, 1, 1)
        assert rows[0]['events_entity_id_1y_outcome::int_sum'] == 1
        assert rows[0]['events_entity_id_1y_outcome::int_avg'] == 0.5
        assert rows[0]['events_entity_id_2y_outcome::int_sum'] == 1
        assert rows[0]['events_entity_id_2y_outcome::int_avg'] == 0.5
        assert rows[0]['events_entity_id_all_outcome::int_sum'] == 1
        assert rows[0]['events_entity_id_all_outcome::int_avg'] == 0.5
        assert rows[1]['entity_id'] == 1
        assert rows[1]['date'] == date(2016, 1, 1)
        assert rows[1]['events_entity_id_1y_outcome::int_sum'] == 1
        assert rows[1]['events_entity_id_1y_outcome::int_avg'] == 0.5
        assert rows[1]['events_entity_id_2y_outcome::int_sum'] == 2
        assert rows[1]['events_entity_id_2y_outcome::int_avg'] == 0.5
        assert rows[1]['events_entity_id_all_outcome::int_sum'] == 2
        assert rows[1]['events_entity_id_all_outcome::int_avg'] == 0.5
        
        assert rows[2]['entity_id'] == 2
        assert rows[2]['date'] == date(2015, 1, 1)
        assert rows[2]['events_entity_id_1y_outcome::int_sum'] == 0
        assert rows[2]['events_entity_id_1y_outcome::int_avg'] == 0
        assert rows[2]['events_entity_id_2y_outcome::int_sum'] == 1
        assert rows[2]['events_entity_id_2y_outcome::int_avg'] == 0.5
        assert rows[2]['events_entity_id_all_outcome::int_sum'] == 1
        assert rows[2]['events_entity_id_all_outcome::int_avg'] == 0.5
        assert rows[3]['entity_id'] == 2
        assert rows[3]['date'] == date(2016, 1, 1)
        assert rows[3]['events_entity_id_1y_outcome::int_sum'] == None
        assert rows[3]['events_entity_id_1y_outcome::int_avg'] == None
        assert rows[3]['events_entity_id_2y_outcome::int_sum'] == 0
        assert rows[3]['events_entity_id_2y_outcome::int_avg'] == 0
        assert rows[3]['events_entity_id_all_outcome::int_sum'] == 1
        assert rows[3]['events_entity_id_all_outcome::int_avg'] == 0.5
        
        assert rows[4]['entity_id'] == 3
        assert rows[4]['date'] == date(2015, 1, 1)
        assert rows[4]['events_entity_id_1y_outcome::int_sum'] == 0
        assert rows[4]['events_entity_id_1y_outcome::int_avg'] == 0
        assert rows[4]['events_entity_id_2y_outcome::int_sum'] == 0
        assert rows[4]['events_entity_id_2y_outcome::int_avg'] == 0
        assert rows[4]['events_entity_id_all_outcome::int_sum'] == 0
        assert rows[4]['events_entity_id_all_outcome::int_avg'] == 0
        assert rows[5]['entity_id'] == 3
        assert rows[5]['date'] == date(2016, 1, 1)
        assert rows[5]['events_entity_id_1y_outcome::int_sum'] == 1
        assert rows[5]['events_entity_id_1y_outcome::int_avg'] == 0.5
        assert rows[5]['events_entity_id_2y_outcome::int_sum'] == 1
        assert rows[5]['events_entity_id_2y_outcome::int_avg'] == 0.25
        assert rows[5]['events_entity_id_all_outcome::int_sum'] == 1
        assert rows[5]['events_entity_id_all_outcome::int_avg'] == 0.25
        
        assert rows[6]['entity_id'] == 4
        # rows[6]['date'] == date(2015, 1, 1) is skipped due to no data!
        assert rows[6]['date'] == date(2016, 1, 1)
        assert rows[6]['events_entity_id_1y_outcome::int_sum'] == 0
        assert rows[6]['events_entity_id_1y_outcome::int_avg'] == 0
        assert rows[6]['events_entity_id_2y_outcome::int_sum'] == 0
        assert rows[6]['events_entity_id_2y_outcome::int_avg'] == 0
        assert rows[6]['events_entity_id_all_outcome::int_sum'] == 0
        assert rows[6]['events_entity_id_all_outcome::int_avg'] == 0
        assert len(rows) == 7
        
def test_input_min_date():
    with testing.postgresql.Postgresql() as psql:
        engine = sqlalchemy.create_engine(psql.url())
        engine.execute(
            'create table events (entity_id int, date date, outcome bool)'
        )
        for event in events_data:
            engine.execute(
                'insert into events values (%s, %s, %s::bool)',
                event
            )
        
        st = SpacetimeAggregation([Aggregate('outcome::int',['sum','avg'])],
            from_obj = 'events',
            groups = ['entity_id'],
            intervals = ['all'],
            dates = ['2016-01-01'],
            date_column = '"date"',
            input_min_date = '2015-11-10')

        st.execute(engine.connect())
        
        r = engine.execute('select * from events_entity_id order by entity_id')
        rows = [x for x in r]
        
        assert rows[0]['entity_id'] == 1
        assert rows[0]['date'] == date(2016, 1, 1)
        assert rows[0]['events_entity_id_all_outcome::int_sum'] == 1
        assert rows[0]['events_entity_id_all_outcome::int_avg'] == 1
        assert rows[1]['entity_id'] == 4
        assert rows[1]['date'] == date(2016, 1, 1)
        assert rows[1]['events_entity_id_all_outcome::int_sum'] == 0
        assert rows[1]['events_entity_id_all_outcome::int_avg'] == 0
        
        assert len(rows) == 2

        st = SpacetimeAggregation([Aggregate('outcome::int',['sum','avg'])],
            from_obj = 'events',
            groups = ['entity_id'],
            intervals = ['1y', 'all'],
            dates = ['2016-01-01', '2015-01-01'],
            date_column = '"date"',
            input_min_date = '2014-11-10')
        with pytest.raises(ValueError):
            st.validate(engine.connect())
        with pytest.raises(ValueError):
            st.execute(engine.connect())

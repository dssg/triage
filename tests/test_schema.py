from triage.schema import InspectionsSchema
import yaml


with open('tests/example_schema.yaml') as stream:
    CONFIG = yaml.load(stream)
schema = InspectionsSchema(CONFIG)


def test_create_schema():
    assert len(schema.models) == 7


def test_many_to_many_temporal():
    assert 'residencies' in schema.models
    table = schema.models['residencies']
    assert len(table.columns) == 5
    assert len(table.foreign_key_constraints) == 2


def test_one_to_many_temporal():
    assert 'house_insurance_policies' in schema.models
    table = schema.models['house_insurance_policies']
    assert len(table.columns) == 5
    assert len(table.foreign_key_constraints) == 2


def test_one_to_many_fixed():
    assert 'house_addresses' not in schema.models
    assert 'addresses' in schema.models
    table = schema.models['houses']
    assert 'address_id' in table.columns.keys()


def test_attributes():
    assert 'houses' in schema.models
    table = schema.models['houses']
    assert 'siding' in table.columns.keys()


def test_basic_pk():
    assert 'id' in schema.models['houses'].columns.keys()
    assert schema.models['houses'].columns['id'].primary_key


def test_relationship_pk():
    assert 'id' in schema.models['residencies'].columns.keys()
    assert schema.models['residencies'].columns['id'].primary_key


def test_events_pk():
    assert 'id' in schema.models['inspections'].columns.keys()
    assert schema.models['inspections'].columns['id'].primary_key


def test_events_datetime():
    assert 'event_datetime' in schema.models['inspections'].columns.keys()
    assert schema.models['inspections'].columns['event_datetime'].index

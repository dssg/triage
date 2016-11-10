from sqlalchemy import \
    Boolean, Integer, String, DateTime,\
    Column, ForeignKey, MetaData, Table
from geoalchemy2 import Geometry
from inflection import pluralize

coltype_lookup = {
    'bool': Boolean,
    'str': String,
    'int': Integer,
}


def join_column(entity):
    return '{}_id'.format(entity)


def fk_column(entity):
    return Column(
        join_column(entity),
        Integer,
        ForeignKey('{}.id'.format(pluralize(entity)))
    )


def relationship_name(relationship):
    name = relationship.get('name')
    if not name:
        name = '{}_{}'.format(
            relationship['entity_one'],
            relationship['entity_two']
        )
    return pluralize(name)


def attach_attributes(table, model):
    attributes = model.get('attributes') or {}
    for attribute_name, datatype in attributes.items():
        table.append_column(
            Column(attribute_name, coltype_lookup[datatype])
        )


class InspectionsSchema(object):
    """
    A configurable schema with the entities and relationships
    suitable for modeling inspections
    Builds a metadata object that can be used to create the
    database and to build SQLAlchemy model classes
    """

    def __init__(self, config):
        """
        Read in config containing entities and relationships,
        and return SQLAlchemy models describing the schema
        """
        self.models = {}
        self.metadata = MetaData()
        self._basic_entities(config['entities'])
        self._relationships(config['relationships'])

    def _basic_entities(self, entity_config):
        """Create top-level entities

        Arguments:
        entity_config -- configuration dict representing the entity
        """
        for entity in entity_config:
            entity_name = pluralize(entity['name'])
            self.models[entity_name] = Table(
                entity_name,
                self.metadata,
                Column('id', Integer, primary_key=True)
            )
            if entity.get('event'):
                self.models[entity_name].append_column(
                    Column('event_datetime', DateTime, index=True)
                )
            if entity.get('spatial'):
                self.models[entity_name].append_column(
                    Column('geom', Geometry())
                )
            attach_attributes(self.models[entity_name], entity)

    def _relationships(self, relationship_config):
        """Create relationships between top-level entities

        Arguments:
        relationship_config -- configuration dict representing the relationship
        """
        for relationship in relationship_config:
            name = relationship_name(relationship)
            if relationship['type'] == 'm2m':
                if relationship.get('temporal'):
                    self.models[name] = Table(
                        name,
                        self.metadata,
                        Column('id', Integer, primary_key=True),
                        fk_column(relationship['entity_one']),
                        fk_column(relationship['entity_two']),
                        Column('start_datetime', DateTime, index=True),
                        Column('end_datetime', DateTime, index=True)
                    )

                else:
                    self.models[name] = Table(
                        name,
                        self.metadata,
                        Column('id', Integer, primary_key=True),
                        fk_column(relationship['entity_one']),
                        fk_column(relationship['entity_two'])
                    )
            else:
                if relationship.get('temporal'):
                    self.models[name] = Table(
                        name,
                        self.metadata,
                        Column('id', Integer, primary_key=True),
                        fk_column(relationship['entity_one']),
                        fk_column(relationship['entity_two']),
                        Column('start_datetime', DateTime, index=True),
                        Column('end_datetime', DateTime, index=True)
                    )
                else:
                    self.models[
                        pluralize(relationship['entity_one'])
                    ].append_column(
                        fk_column(relationship['entity_two'])
                    )
            if name in self.models:
                attach_attributes(self.models[name], relationship)

class EntityFeatureDateGenerator(object):
    """
    Generate unique combinations of entity_id and feature date

    Args:
        split (dict) with:
        train_start, train_end, test_start, test_end, prediction_window
        events_table (string)
    """
    def __init__(self, split, events_table, db_engine):
        self.split = split
        self.events_table = events_table
        self.db_engine = db_engine

    def _select_entity_feature_dates(self):
        assert False

    def generate(self):
        table_name = 'entity_feature_dates'
        query = """
            create table {table_name} (entity_id, feature_date) as
            {select_query}
        """.format(table_name=table_name,
                   select_query=self._select_entity_feature_dates())
        self.db_engine.execute(query)
        return table_name


class TimeOfEventFeatureDateGenerator(EntityFeatureDateGenerator):
    def _select_entity_feature_dates(self):
        return """
        select entity_id, event_date
        from {events_table}
        where event_date between '{start}' and '{end}'
        group by 1, 2
        """.format(
            start=self.split['train_start'],
            end=self.split['train_end'],
            events_table=self.events_table
        )


class WholeWindowFeatureDateGenerator(EntityFeatureDateGenerator):
    def _select_entity_feature_dates(self):
        return """
        select entity_id, '{start}'::date
        from {events_table}
        where event_date between '{start}' and '{end}'
        group by 1, 2
        """.format(
            start=self.split['train_start'],
            end=self.split['train_end'],
            events_table=self.events_table
        )

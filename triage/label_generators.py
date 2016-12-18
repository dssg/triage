class LabelGenerator(object):
    def __init__(self, events_table, start_date, end_date, db_engine):
        self.events_table = events_table
        self.start_date = start_date
        self.end_date = end_date
        self.db_engine = db_engine

    def generate(self):
        labels_table = 'labels'
        query = """create table {labels_table} as (
            select
            entity_id, event_date as outcome_date, bool_or(outcome)
            from {events_table}
            where '{train_start}' <= event_date
            and event_date < '{train_end}'
            group by 1, 2
        )""".format(
            events_table=self.events_table,
            labels_table=labels_table,
            train_start=self.start_date,
            train_end=self.end_date
        )
        self.db_engine.execute(query)
        return labels_table

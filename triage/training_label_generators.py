class TrainingLabelGenerator(object):
    def __init__(self, entity_feature_dates_table, events_table, db_engine):
        self.entity_feature_dates_table = entity_feature_dates_table
        self.events_table = events_table
        self.db_engine = db_engine

    def generate(self):
        training_labels_table = 'training_labels'
        query = """create table {labels_table} as (
            select
            e.entity_id, ef.feature_date as event_date, bool_or(e.outcome)
            from {events_table} e
            join {entity_feature_dates_table} ef
            on (e.entity_id = ef.entity_id and ef.feature_date < e.event_date)
            group by 1, 2
        )""".format(
            events_table=self.events_table,
            entity_feature_dates_table=self.entity_feature_dates_table,
            labels_table=training_labels_table
        )
        self.db_engine.execute(query)
        return training_labels_table

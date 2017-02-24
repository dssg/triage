import logging


class LabelGenerator(object):
    def __init__(self, events_table, db_engine):
        self.events_table = events_table
        self.db_engine = db_engine

    def generate(self, start_date, end_date, labels_table):
        query = """insert into {labels_table} (
            select
                entity_id,
                '{start_date}'::date as outcome_date,
                bool_or(outcome) as outcome
            from {events_table}
            where '{start_date}' <= date
            and date < '{end_date}'
            group by 1, 2
        )""".format(
            events_table=self.events_table,
            labels_table=labels_table,
            start_date=start_date,
            end_date=end_date
        )
        logging.debug(query)
        self.db_engine.execute(query)
        return labels_table

import logging
import tempfile
import boto
from config import config
import uuid


class SetGenerator(object):
    def __init__(self, features_table, labels_table, db_engine, s3_conn=None):
        self.features_table = features_table
        self.labels_table = labels_table
        self.db_engine = db_engine
        self.s3_conn = s3_conn or boto.connect_s3()

    def generate(self, start_date, end_date):
        query = """
            select features.*, labels.outcome
            from {features_table} features
            left join {labels_table} labels on (
                labels.entity_id = features.entity_id and
                outcome_date > '{start_date}' and outcome_date <= '{end_date}'
            )
        """.format(
            features_table=self.features_table,
            labels_table=self.labels_table,
            start_date=start_date,
            end_date=end_date
        )
        conn = self.db_engine.raw_connection()
        cursor = conn.cursor()
        bucket_name = config['shared_bucket']
        key_path = 'sets/{}/{}'.format(
            config['project'],
            str(uuid.uuid4())
        )
        with tempfile.NamedTemporaryFile() as f:
            cursor.copy_expert('COPY ({}) to STDOUT CSV HEADER'.format(query), f)
            f.seek(0)
            bucket = self.s3_conn.get_bucket(bucket_name)
            key = boto.s3.key.Key(bucket=bucket, name=key_path)
            key.set_contents_from_file(f)
        return key_path


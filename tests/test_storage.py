from triage.storage import S3Store, FSStore
from moto import mock_s3
import boto3
import os


class SomeClass(object):
    def __init__(self, val):
        self.val = val


def test_S3Store():
    with mock_s3():
        s3_conn = boto3.resource('s3')
        s3_conn.create_bucket(Bucket='a-bucket')
        store = S3Store(s3_conn.Object('a-bucket', 'a-path'))
        assert not store.exists()
        store.write(SomeClass('val'))
        assert store.exists()
        newVal = store.load()
        assert newVal.val == 'val'


def test_FSStore():
    if os.path.isfile('tmpfile'):
        os.remove('tmpfile')
    store = FSStore('tmpfile')
    assert not store.exists()
    store.write(SomeClass('val'))
    assert store.exists()
    newVal = store.load()
    assert newVal.val == 'val'
    os.remove('tmpfile')

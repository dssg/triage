from pkg_resources import resource_filename

example_uuid_fname = resource_filename(__name__, '/data/.matrix_uuids')
example_directory = resource_filename(__name__, '/data/')
example_data = resource_filename(__name__, '/data/titanic.csv')
example_aws_creds = resource_filename(__name__, '/data/rootkey.yaml')

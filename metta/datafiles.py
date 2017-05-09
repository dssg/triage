from pkg_resources import resource_filename

example_directory = resource_filename(__name__, '/data/')
example_data_csv = resource_filename(__name__, '/data/titanic.csv')
example_data_h5 = resource_filename(__name__, '/data/titanic.h5')
example_aws_creds = resource_filename(__name__, '/data/rootkey.yaml')

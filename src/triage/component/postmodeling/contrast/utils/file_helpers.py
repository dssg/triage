'''
File helpers to download matrices from s3:

To run this functions properly is necessary to define the AWS S3 
credentials. s3fs library will first look for environment variables 
or a ~/.aws/credentials file. Also, since s3fs is based in boto3, is 
possible to define S3 credentials in ~/.boto. 

'''

import s3fs
import pandas as pd
import pickle


def download_s3(path, read_csv=True, model_file=False):

    fs = s3fs.S3FileSystem()

    if read_csv:
        try:
            path = str(path) + '.csv'
            print('Downloading {} from {} s3 bucket'.format(path))
            with fs.open(path) as s3_file:
                df = pd.read_csv(s3_file)
                return df
        except FileNotFoundError:
            print('No matrix in bucket')

    elif model_file:
        try:
            path = str(path)
            print(path)
            print('Downloading {} model s3 bucket'.format(path))
            with fs.open(path, 'rb') as s3_file:
                model = pickle.load(s3_file)
                return model
        except FileNotFoundError:
            print('No matrix in bucket')
    else:
        try:
            path = str(path) + '.csv'
            print('Download first line from {} s3 bucket'.format(path))
            with fs.open(path, 'rb') as s3_file:
                line = s3_file.readline().decode().replace('\n', '')
                line_list = str(line).split(',')
                return line_list
        except FileNotFoundError:
            print('No matrix in bucket')


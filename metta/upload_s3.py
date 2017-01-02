"""
Upload files to s3
"""
import boto3
import botocore
import os
import glob
import logging


def upload_to_s3(access_key_id,
                 secret_access_key,
                 bucket,
                 folder,
                 ls_check_for_files=['matrix_pairs.txt', '.matrix_uuids'],
                 directory='.'):
    """
    Uploads files in a directory to a S3 bucket while checking for files
    metadatafiles and concatenating them.

    Parameters
    ----------
    access_key_id: str
        AWS access key
    secret_access_key: str
        AWS secret access key
    bucket: str
        Bucket name
    folder: str
        Folder to store data in S3 bucket
    ls_check_for_files: ls[str]
        Filenames to check for(default ['matching_pairs.txt','.pairs'])
    directory: str
        Directory to look for files in (default .)

    Returns
    -------
    logfile: file
        Logfile is written to /tmp/upload_s3.log
    """

    abs_path_dir = os.path.abspath(directory)

    logging.basicConfig(filename=abs_path_dir + '/upload_s3.log',
                        level=logging.DEBUG)

    s3 = boto3.client('s3',
                      aws_access_key_id=access_key_id,
                      aws_secret_access_key=secret_access_key)

    for filename in ls_check_for_files:
        try:
            s3.download_file(Bucket=bucket,
                             Key=folder + '/' + filename,
                             Filename=directory + '/' + "old_" + filename)

            if os.path.isfile("old_" + filename):
                abs_fname = directory + '/' + filename
                abs_oldfname = directory + '/' + "old_" + filename
                cmd = 'cat {} {} > tmp;mv tmp {};rm {}'.format(abs_fname,
                                                               abs_oldfname,
                                                               abs_fname,
                                                               abs_oldfname)
                logging.info(cmd)
                os.system(cmd)

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                logging.info('File {} not in bucket'.format(filename))
        else:
            logging.info('Downloaded: {}'.format(filename))

    abs_path_check_files = [abs_path_dir + '/' +
                            fname for fname in ls_check_for_files]
    ls_files_to_upload = glob.glob(abs_path_dir + '/*') + abs_path_check_files
    for fname in ls_files_to_upload:
        logging.info('Uploading: {}'.format(fname))
        s3.upload_file(Bucket=bucket,
                       Key=folder + '/' + fname.split('/')[-1],
                       Filename=fname)

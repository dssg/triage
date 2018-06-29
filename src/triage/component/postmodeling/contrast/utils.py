from sqlalchemy import create_engine
import logging


def get_conn_engine(dbfile):
    """
    Returns connection engine for postgresql

    Parameters
    ----------
    dbfile: file
        yaml file of db creds

    Returns
    -------
    conn_engine: str
        connection engine for postgres
    """

    try:
        with open(db_credentials_file) as f:
            dbdict = yaml.load(f)
        conn_string = '{dbtype}://{user}:{password}@{host}:{port}/{db}'.format(
            dbtype='postgresql',
            user=dbdict['user'],
            password=dbdict['password'],
            host=dbdict['host'],
            port=dbdict['port'],
            db=dbdict['database'])
    except FileNotFoundError:
        logging.error('Cannot find file {}'.format(db_credentials_file))

    return create_engine(conn_string)

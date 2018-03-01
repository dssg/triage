from sqlalchemy import create_engine
import json
import os

def get_engine():

    with open(os.path.join(os.path.dirname(os.path.dirname(
              os.path.abspath(__file__))), 'database_credentials.json'), 'r') as f:
       
        creds = json.load(f)
    
    conn_str = "postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
            **creds)

    return create_engine(conn_str)


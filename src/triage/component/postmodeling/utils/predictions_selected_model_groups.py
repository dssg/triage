import sys
import pandas as pd 
import logging
import sqlalchemy
import yaml
import argparse

from triage.component.catwalk.predictors import Predictor
from triage.component.catwalk.storage import ProjectStorage
from triage import create_engine


def _fetch_relevant_model_matrix_info(db_engine, model_groups, experiment_hashes=None):
    """ For the given experiment and model groups, fetch the model_ids, and match them with their train/test matrix pairs 
        Args:
            model_groups (List[int]): A list of model groups in the experiment
            experiment_hashes (List[str]): Optional. A list of experiment hashes we are interested in.
                If this is provided, only the model_ids that are relevant to the given experiments will be returned

        Return: 
            A DataFrame that contains the relevant model_ids, train_end_times, and matrix information
    """

    q = """
        SELECT 
            model_id,
            train_end_time,
            model_hash,
            model_group_id,
            train_matrix_uuid,
            c.matrix_uuid as test_matrix_uuid
        FROM triage_metadata.experiment_models a 
            JOIN triage_metadata.models b using(model_hash)
                JOIN test_results.prediction_metadata c using(model_id)
        WHERE model_group_id in ({model_groups})"""

    args_dict = {
        'model_groups': ', '.join([str(x) for x in model_groups])
    }

    if experiment_hashes is not None:
        q = q + " AND experiment_hash in ({experiment_hashes})"
        args_dict['experiment_hashes'] = ', '.join(["'" + str(x) + "'" for x in experiment_hashes])
    
    q = q.format(**args_dict)

    return pd.read_sql(q, db_engine)

    
def generate_predictions(db_engine, model_groups, project_path, experiment_hashes=None, range_train_end_times=None, rank_odrer='worst', replace=False):
    """ Generate predictions and write to DB for a set of model_groups in an experiment
        Args:
            db_conn: Sqlalchemy engine
            model_groups (list): The list of model group ids we are interested in (ideally, chosen through audition)
            project_path (str): Path where the created matrices and trained model objects are stored for the experiment
            experiment_hashes (List[str]): hash of the experiment (currently handling only one experiment)
            range_train_end_times (): Optional. If given, only the models with train_end_times that fall in the range are scored.
                                        Should be a list of two dates (str). [range_start_date, range_end_date]
            rank_order (str) : How to deal with ties in the scores. 
            replace (bool) : Whether to overwrite the preditctions for a model_id, if already found in the DB

        Returns: None
            This directly writes to the test_results.predictions table
    """
    model_matrix_info = _fetch_relevant_model_matrix_info(
        db_engine=db_engine,
        model_groups=model_groups,
        experiment_hashes=experiment_hashes
    )

    if len(model_matrix_info)==0:
        raise ValueError('No models were found for the given experiment and model group(s)')
    
    # All the model groups we want to save predictions for, should be in the DB 
    not_fetched_model_grps = [x for x in model_groups if not x in model_matrix_info['model_group_id'].unique()]
    if len(not_fetched_model_grps) > 0:
        raise ValueError('No models were found for the model groups {}. All specified model groups should be a part of the given experiment'.format(not_fetched_model_grps))
    
    logging.info('Found {} model ids'.format(len(model_matrix_info)))

    # If we are only generating predictions for a specific time range
    if range_train_end_times is not None: 
        range_st = range_train_end_times[0]
        range_en = range_train_end_times[1]

        logging.info('Filtering out models with a train_end_time outside of the range ({}, {})'.format(range_st, range_en))

        msk = (model_matrix_info['train_end_time'] >= range_st) & (model_matrix_info['train_end_time'] <= range_en)
        model_matrix_info = model_matrix_info[msk]

        if len(model_matrix_info) == 0:
            raise ValueError('No models were found for the given time range')

        logging.info('Scoring {} models'.format(len(model_matrix_info)))

    logging.info('Instantiating storage engines and the predictor')
    
    # Storage objects to handle already stored models and matrices
    project_storage = ProjectStorage(project_path)
    model_storage_engine = project_storage.model_storage_engine()
    matrix_storage_engine = project_storage.matrix_storage_engine()

    # Prediction generation is handled by the Predictor class in catwalk
    predictor = Predictor(
        model_storage_engine=model_storage_engine,
        db_engine=db_engine,
        rank_order=rank_odrer,
        replace=replace,
        save_predictions=True
    )

    # Organizing prediction run over unique (train_mat, test_mat) pairs
    # This is to reduce no. the times the matrices get loaded to memory
    groupby_obj = model_matrix_info.groupby(['train_matrix_uuid', 'test_matrix_uuid'])

    for group, _ in groupby_obj:
        train_uuid = group[0]
        test_uuid = group[1]

        df_grp = groupby_obj.get_group(group)

        logging.info('Processing {} model_ids for train matrix {} and test matrix {}'.format(
            len(df_grp), train_uuid, test_uuid
        ))

        train_matrix_store = matrix_storage_engine.get_store(matrix_uuid=train_uuid)

        # To ensure that the column order we use for predictions match the order we used in model training
        train_matrix_columns = list(train_matrix_store.design_matrix.columns)
        
        test_matrix_store = matrix_storage_engine.get_store(matrix_uuid=test_uuid)

        for model_id in df_grp['model_id'].tolist():
            logging.info('Writing predictions for model_id {}'.format(model_id))
            predictor.predict(
                model_id=model_id,
                matrix_store=test_matrix_store,
                train_matrix_columns=train_matrix_columns,
                misc_db_parameters={}
            )

    logging.info('Successfully generated predictions for {} models!'.format(len(model_matrix_info)))


def _load_yaml(file_path):
    try: 
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    except:
        raise FileNotFoundError('File {} was not found'.format(file_path))

    return config


def run(config_file, db_credentials_file='database.yaml'):
    """Run the prediction generation pipeline
        Args: 
            config_file (str) : path to the config file
            db_credentials_file (str): Path to the database credentials file. 
                                If not specified, the working directory should contain a database.yaml.
    """

    db_conf = _load_yaml(db_credentials_file)
    dburl = sqlalchemy.engine.url.URL(
        "postgres",
        host=db_conf["host"],
        username=db_conf["user"],
        database=db_conf["db"],
        password=db_conf["pass"],
        port=db_conf["port"],
    )
    db_engine = sqlalchemy.create_engine(dburl, poolclass=sqlalchemy.pool.QueuePool)
   
    config = _load_yaml(config_file)
    generate_predictions(
        db_engine=db_engine,
        model_groups=config['model_group_ids'],
        project_path=config['project_path'],
        experiment_hashes=config.get('experiments'),
        range_train_end_times=config.get('train_end_times')
    )


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c",
        "--configfile",
        type=str,
        help="Path to the configuration file (required)",
        required=True
    )
    arg_parser.add_argument(
        "-d",
        "--dbfile",
        type=str,
        help="Pass the db connection information",
        default='database.yaml'
    )

    args = arg_parser.parse_args()

    run(
        config_file=args.configfile, 
        db_credentials_file=args.dbfile
    )



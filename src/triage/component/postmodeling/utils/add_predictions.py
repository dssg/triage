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


def _summary_of_models(models_df):
    """generates logging summaries and warnings about the models being scored"""

    # how many models per model group?
    cts = models_df.groupby('model_group_id').count()['model_id']
    cts.name = 'num_models'
    model_counts = cts.unique()
    if len(model_counts) > 1:
        logging.warning('No. of models is different between model groups. See below: \n {}'.format(cts.reset_index()))
    else:
        logging.info('Each model group contains {} models'.format(model_counts[0]))

    # how many models per model group at each train_end_time?
    # Sometimes there are more than one model for each model group with different random seeds
    time_cts = models_df.groupby(['train_end_time', 'model_group_id']).count()['model_id']
    time_cts.name = 'num_models'
    msk = time_cts > 1

    if len(time_cts[msk]) > 1:
        logging.warning('There are model groups with more than one model per train_end_time. See below: \n {}'.format(time_cts[msk].reset_index()))
        
    
def add_predictions(db_engine, model_groups, project_path, experiment_hashes=None, train_end_times_range=None, rank_order='worst', replace=True):
    """ For a set of modl_groups generate test predictions and write to DB
        Args:
            db_engine: Sqlalchemy engine
            model_groups (list): The list of model group ids we are interested in (ideally, chosen through audition)
            project_path (str): Path where the created matrices and trained model objects are stored for the experiment
            experiment_hashes (List[str]): Optional. hash(es) of the experiments we are interested in. Can be used to narrow down the model_ids in the model groups specified
            range_train_end_times (Dict): Optional. If provided, only the models with train_end_times that fall in the range are scored. 
                                        This too, helps narrow down model_ids in the model groups specified.
                                        A dictionary with two possible keys 'range_start_date' and 'range_end_date'. Either or both could be set
            rank_order (str) : How to deal with ties in the scores. 
            replace (bool) : Whether to overwrite the preditctions for a model_id, if already found in the DB.

        Returns: None
            This directly writes to the test_results.predictions table
    """

    model_matrix_info = _fetch_relevant_model_matrix_info(
        db_engine=db_engine,
        model_groups=model_groups,
        experiment_hashes=experiment_hashes
    )

    # If we are only generating predictions for a specific time range
    if train_end_times_range is not None: 
        if 'range_start_date' in train_end_times_range:
            range_start = train_end_times_range['range_start_date']
            msk = (model_matrix_info['train_end_time'] >= range_start)
            logging.info('Filtering out models with a train_end_time before {}'.format(range_start))

            model_matrix_info = model_matrix_info[msk]

        if 'range_end_date' in train_end_times_range:       
            range_end = train_end_times_range['range_end_date']
            msk = (model_matrix_info['train_end_time'] <= range_end)
            logging.info('Filtering out models with a train_end_time after {}'.format(range_end))

            model_matrix_info = model_matrix_info[msk]

    if len(model_matrix_info)==0:
        raise ValueError('Configis not valid. No models were found!')
    
    # Al the model groups specified in the config file should valid (even if the experiment_hashes and train_end_times are specified)
    not_fetched_model_grps = [x for x in model_groups if not x in model_matrix_info['model_group_id'].unique()]
    
    if len(not_fetched_model_grps) > 0:
        raise ValueError('The config is not valid. No models were found for the model group(s) {}. All specified model groups should be present'.format(not_fetched_model_grps))
    
    logging.info('Scoring {} model ids'.format(len(model_matrix_info)))

    # summary of the models that we are scoring. To check any special things worth noting
    _summary_of_models(model_matrix_info)
    
    logging.info('Instantiating storage engines and the predictor')
    
    # Storage objects to handle already stored models and matrices
    project_storage = ProjectStorage(project_path)
    model_storage_engine = project_storage.model_storage_engine()
    matrix_storage_engine = project_storage.matrix_storage_engine()

    # Prediction generation is handled by the Predictor class in catwalk
    predictor = Predictor(
        model_storage_engine=model_storage_engine,
        db_engine=db_engine,
        rank_order=rank_order,
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

from triage.component.results_schema import upgrade_db
from triage.component.architect.entity_date_table_generators import EntityDateTableGenerator, DEFAULT_ACTIVE_STATE
from triage.component.architect.features import (
        FeatureGenerator, 
        FeatureDictionaryCreator, 
        FeatureGroupCreator,
        FeatureGroupMixer,
)
from triage.component.architect.feature_group_creator import FeatureGroup
from triage.component.architect.builders import MatrixBuilder
from triage.component.architect.planner import Planner
from triage.component.architect.label_generators import LabelGenerator
from triage.component.timechop import Timechop
from triage.component.catwalk.storage import ModelStorageEngine, ProjectStorage
from triage.component.catwalk import ModelTrainer
from triage.component.catwalk.model_trainers import flatten_grid_config
from triage.component.catwalk.predictors import Predictor
from triage.component.catwalk.utils import retrieve_model_hash_from_id, filename_friendly_hash
from triage.util.conf import convert_str_to_relativedelta, dt_from_str
from triage.util.db import scoped_session
from .utils import (
    experiment_config_from_model_id,
    experiment_config_from_model_group_id,
    get_model_group_info,
    train_matrix_info_from_model_id,
    get_feature_names,
    get_feature_needs_imputation_in_train,
    get_feature_needs_imputation_in_production,
)


from collections import OrderedDict
import json
import random
from datetime import datetime

import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)



def predict_forward_with_existed_model(db_engine, project_path, model_id, as_of_date):
    """Predict forward given model_id and as_of_date and store the prediction in database

    Args:
            db_engine (sqlalchemy.db.engine)
            project_storage (catwalk.storage.ProjectStorage)
            model_id (int) The id of a given model in the database
            as_of_date (string) a date string like "YYYY-MM-DD"
    """
    logger.spam("In PREDICT LIST................")
    upgrade_db(db_engine=db_engine)
    project_storage = ProjectStorage(project_path)
    matrix_storage_engine = project_storage.matrix_storage_engine()
    # 1. Get feature and cohort config from database
    (train_matrix_uuid, matrix_metadata) = train_matrix_info_from_model_id(db_engine, model_id)
    experiment_config = experiment_config_from_model_id(db_engine, model_id)
 
    # 2. Generate cohort
    cohort_table_name = f"triage_production.cohort_{experiment_config['cohort_config']['name']}"
    cohort_table_generator = EntityDateTableGenerator(
        db_engine=db_engine,
        query=experiment_config['cohort_config']['query'],
        entity_date_table_name=cohort_table_name
    )
    cohort_table_generator.generate_entity_date_table(as_of_dates=[dt_from_str(as_of_date)])
    
    # 3. Generate feature aggregations
    feature_generator = FeatureGenerator(
        db_engine=db_engine,
        features_schema_name="triage_production",
        feature_start_time=experiment_config['temporal_config']['feature_start_time'],
    )
    collate_aggregations = feature_generator.aggregations(
        feature_aggregation_config=experiment_config['feature_aggregations'],
        feature_dates=[as_of_date],
        state_table=cohort_table_name
    )
    feature_generator.process_table_tasks(
        feature_generator.generate_all_table_tasks(
            collate_aggregations,
            task_type='aggregation'
        )
    )

    # 4. Reconstruct feature disctionary from feature_names and generate imputation
    
    reconstructed_feature_dict = FeatureGroup()
    imputation_table_tasks = OrderedDict()

    for aggregation in collate_aggregations:
        feature_group, feature_names = get_feature_names(aggregation, matrix_metadata)
        reconstructed_feature_dict[feature_group] = feature_names

        # Make sure that the features imputed in training should also be imputed in production
        
        features_imputed_in_train = get_feature_needs_imputation_in_train(aggregation, feature_names)
        
        features_imputed_in_production = get_feature_needs_imputation_in_production(aggregation, db_engine)

        total_impute_cols = set(features_imputed_in_production) | set(features_imputed_in_train)
        total_nonimpute_cols = set(f for f in set(feature_names) if '_imp' not in f) - total_impute_cols
        
        task_generator = feature_generator._generate_imp_table_tasks_for
        
        imputation_table_tasks.update(task_generator(
            aggregation,
            impute_cols=list(total_impute_cols),
            nonimpute_cols=list(total_nonimpute_cols)
            )
        )
    feature_generator.process_table_tasks(imputation_table_tasks)

    # 5. Build matrix
    db_config = {
        "features_schema_name": "triage_production",
        "labels_schema_name": "public",
        "cohort_table_name": cohort_table_name,
    }

    matrix_builder = MatrixBuilder(
        db_config=db_config,
        matrix_storage_engine=matrix_storage_engine,
        engine=db_engine,
        experiment_hash=None,
        replace=True,
    )
       
    feature_start_time = experiment_config['temporal_config']['feature_start_time']
    label_name = experiment_config['label_config']['name']
    label_type = 'binary'
    cohort_name = experiment_config['cohort_config']['name']
    user_metadata = experiment_config['user_metadata']
    
    # Use timechop to get the time definition for production
    temporal_config = experiment_config["temporal_config"]
    timechopper = Timechop(**temporal_config)
    prod_definitions = timechopper.define_test_matrices(
            train_test_split_time=dt_from_str(as_of_date), 
            test_duration=temporal_config['test_durations'][0],
            test_label_timespan=temporal_config['test_label_timespans'][0]
    )

    matrix_metadata = Planner.make_metadata(
            prod_definitions[-1],
            reconstructed_feature_dict,
            label_name,
            label_type,
            cohort_name,
            'production',
            feature_start_time,
            user_metadata,
    )
    
    matrix_metadata['matrix_id'] = str(as_of_date) +  f'_model_id_{model_id}' + '_risklist'

    matrix_uuid = filename_friendly_hash(matrix_metadata)
    
    matrix_builder.build_matrix(
        as_of_times=[as_of_date],
        label_name=label_name,
        label_type=label_type,
        feature_dictionary=reconstructed_feature_dict,
        matrix_metadata=matrix_metadata,
        matrix_uuid=matrix_uuid,
        matrix_type="production",
    )
    
    # 6. Predict the risk score for production
    predictor = Predictor(
        model_storage_engine=project_storage.model_storage_engine(),
        db_engine=db_engine,
        rank_order='best'
    )

    predictor.predict(
        model_id=model_id,
        matrix_store=matrix_storage_engine.get_store(matrix_uuid),
        misc_db_parameters={},
        train_matrix_columns=matrix_storage_engine.get_store(train_matrix_uuid).columns()
    )
    

class Retrainer:
    """Given a model_group_id and prediction_date, retrain a model using the all the data till prediction_date 
    Args:
        db_engine (sqlalchemy.engine)
        project_path (string)
        model_group_id (string)
    """
    def __init__(self, db_engine, project_path, model_group_id):
        self.db_engine = db_engine
        upgrade_db(db_engine=self.db_engine)
        self.project_storage = ProjectStorage(project_path)
        self.model_group_id = model_group_id
        self.matrix_storage_engine = self.project_storage.matrix_storage_engine()
        self.run_id, self.experiment_config = experiment_config_from_model_group_id(self.db_engine, self.model_group_id)
        self.training_label_timespan = self.experiment_config['temporal_config']['training_label_timespans'][0]
        self.test_label_timespan = self.experiment_config['temporal_config']['test_label_timespans'][0]
        self.feature_start_time=self.experiment_config['temporal_config']['feature_start_time']
        self.label_name = self.experiment_config['label_config']['name']
        self.cohort_name = self.experiment_config['cohort_config']['name']
        self.user_metadata = self.experiment_config['user_metadata']
        self.model_group_info = get_model_group_info(self.db_engine, self.model_group_id)
        
        self.feature_dictionary_creator = FeatureDictionaryCreator(
            features_schema_name='triage_production', db_engine=self.db_engine
        )
        self.label_generator = LabelGenerator(
            label_name=self.experiment_config['label_config'].get("name", None),
            query=self.experiment_config['label_config']["query"],
            replace=True,
            db_engine=self.db_engine,
        )        
        
        self.labels_table_name = "labels_{}_{}_production".format(
            self.experiment_config['label_config'].get('name', 'default'),
            filename_friendly_hash(self.experiment_config['label_config']['query'])
        )

        self.feature_generator = FeatureGenerator(
            db_engine=self.db_engine,
            features_schema_name="triage_production",
            feature_start_time=self.feature_start_time,
        )

        self.model_trainer = ModelTrainer(
            experiment_hash=None,
            model_storage_engine=ModelStorageEngine(self.project_storage),
            db_engine=self.db_engine,
            replace=True,
            run_id=self.run_id,
        )
    
    def get_temporal_config_for_retrain(self, prediction_date):
        temporal_config = self.experiment_config['temporal_config'].copy()
        temporal_config['feature_end_time'] = datetime.strftime(prediction_date, "%Y-%m-%d")
        temporal_config['label_start_time'] = datetime.strftime(
                prediction_date - 
                convert_str_to_relativedelta(self.training_label_timespan) - 
                convert_str_to_relativedelta(self.test_label_timespan), 
                "%Y-%m-%d")
        temporal_config['label_end_time'] = datetime.strftime(
                prediction_date + convert_str_to_relativedelta(self.test_label_timespan), 
                "%Y-%m-%d")
        temporal_config['model_update_frequency'] = self.test_label_timespan
        
        return temporal_config

    def generate_all_labels(self, as_of_date):
        self.label_generator.generate_all_labels(
                labels_table=self.labels_table_name, 
                as_of_dates=[as_of_date], 
                label_timespans=[self.training_label_timespan]
        )

    def generate_entity_date_table(self, as_of_date, entity_date_table_name):
        cohort_table_generator = EntityDateTableGenerator(
            db_engine=self.db_engine,
            query=self.experiment_config['cohort_config']['query'],
            entity_date_table_name=entity_date_table_name
        )
        cohort_table_generator.generate_entity_date_table(as_of_dates=[dt_from_str(as_of_date)])
       
    def get_collate_aggregations(self, as_of_date, state_table):
        collate_aggregations = self.feature_generator.aggregations(
            feature_aggregation_config=self.experiment_config['feature_aggregations'],
            feature_dates=[as_of_date],
            state_table=state_table
        )
        return collate_aggregations

    def get_feature_dict_and_imputation_task(self, collate_aggregations, model_id):
        (train_matrix_uuid, matrix_metadata) = train_matrix_info_from_model_id(self.db_engine, model_id)
        reconstructed_feature_dict = FeatureGroup()
        imputation_table_tasks = OrderedDict()

        for aggregation in collate_aggregations:
            feature_group, feature_names = get_feature_names(aggregation, matrix_metadata)
            reconstructed_feature_dict[feature_group] = feature_names

            # Make sure that the features imputed in training should also be imputed in production
            
            features_imputed_in_train = get_feature_needs_imputation_in_train(aggregation, feature_names)
            
            features_imputed_in_production = get_feature_needs_imputation_in_production(aggregation, self.db_engine)

            total_impute_cols = set(features_imputed_in_production) | set(features_imputed_in_train)
            total_nonimpute_cols = set(f for f in set(feature_names) if '_imp' not in f) - total_impute_cols
            
            task_generator = self.feature_generator._generate_imp_table_tasks_for
            
            imputation_table_tasks.update(task_generator(
                aggregation,
                impute_cols=list(total_impute_cols),
                nonimpute_cols=list(total_nonimpute_cols)
                )
            )
        return reconstructed_feature_dict, imputation_table_tasks
    
    def retrain(self, prediction_date):
        """Retrain a model by going back one split from prediction_date, so the as_of_date for training would be (prediction_date - training_label_timespan)
        
        Args:
            prediction_date(str) 
        """
        prediction_date = dt_from_str(prediction_date)
        temporal_config = self.get_temporal_config_for_retrain(prediction_date)
        timechopper = Timechop(**temporal_config)
        chops = timechopper.chop_time()
        assert len(chops) == 1
        chops_train_matrix = chops[0]['train_matrix']
        retrain_definition = {
            'first_as_of_time': chops_train_matrix['first_as_of_time'],
            'last_as_of_time': chops_train_matrix['last_as_of_time'],
            'matrix_info_end_time': chops_train_matrix['matrix_info_end_time'],
            'as_of_times': chops_train_matrix['as_of_times'],
            'training_label_timespan': chops_train_matrix['training_label_timespan'],
            'max_training_history': chops_train_matrix['max_training_history'],
        }
        as_of_date = datetime.strftime(chops_train_matrix['last_as_of_time'], "%Y-%m-%d")
        
        cohort_table_name = f"triage_production.cohort_{self.experiment_config['cohort_config']['name']}_retrain"
        # 1. Generate all labels
        self.generate_all_labels(as_of_date)

        # 2. Generate cohort
        self.generate_entity_date_table(as_of_date, cohort_table_name)

        # 3. Generate feature aggregations
        collate_aggregations = self.get_collate_aggregations(as_of_date, cohort_table_name)
        feature_aggregation_table_tasks = self.feature_generator.generate_all_table_tasks(
            collate_aggregations,
            task_type='aggregation'
        )
        self.feature_generator.process_table_tasks(feature_aggregation_table_tasks)

        # 4. Reconstruct feature disctionary from feature_names and generate imputation
        feature_imputation_table_tasks = self.feature_generator.generate_all_table_tasks(
            collate_aggregations,
            task_type='imputation'
        )
        self.feature_generator.process_table_tasks(feature_imputation_table_tasks)
        
        feature_dict = self.feature_dictionary_creator.feature_dictionary(
            feature_table_names=feature_imputation_table_tasks.keys(),
            index_column_lookup=self.feature_generator.index_column_lookup(collate_aggregations),
        )
        
        feature_group_creator = FeatureGroupCreator({"all": [True]})
        feature_group_mixer = FeatureGroupMixer(["all"])
        feature_group_dict = feature_group_mixer.generate(
            feature_group_creator.subsets(feature_dict) 
        )[0]

        # 5. Build new matrix
        db_config = {
            "features_schema_name": "triage_production",
            "labels_schema_name": "public",
            "cohort_table_name": cohort_table_name,
            "labels_table_name": self.labels_table_name,
        }

        matrix_builder = MatrixBuilder(
            db_config=db_config,
            matrix_storage_engine=self.matrix_storage_engine,
            engine=self.db_engine,
            experiment_hash=None,
            replace=True,
        )
        new_matrix_metadata = Planner.make_metadata(
            matrix_definition=retrain_definition,
            feature_dictionary=feature_group_dict,
            label_name=self.label_name,
            label_type='binary',
            cohort_name=self.cohort_name,
            matrix_type='train',
            feature_start_time=self.feature_start_time,
            user_metadata=self.user_metadata,
        )
        
        new_matrix_metadata['matrix_id'] = "_".join(
            [
                self.label_name,
                'binary',
                str(as_of_date),
                'retrain',
                ]
        )

        matrix_uuid = filename_friendly_hash(new_matrix_metadata)
        matrix_builder.build_matrix(
            as_of_times=[as_of_date],
            label_name=self.label_name,
            label_type='binary',
            feature_dictionary=feature_group_dict,
            matrix_metadata=new_matrix_metadata,
            matrix_uuid=matrix_uuid,
            matrix_type="train",
        )
        retrained_model_comment = 'retrain_' + str(datetime.now())

        misc_db_parameters = {
            'train_end_time': dt_from_str(as_of_date),
            'test': False,
            'train_matrix_uuid': matrix_uuid, 
            'training_label_timespan': self.training_label_timespan,
            'model_comment': retrained_model_comment,
        }
        
        # get the random seed fromthe last split 
        last_split_train_matrix_uuid, last_split_matrix_metadata = train_matrix_info_from_model_id(
            self.db_engine, 
            model_id=self.model_group_info['model_id_last_split']
        )

        random_seed = self.model_trainer.get_or_generate_random_seed( 
            model_group_id=self.model_group_id, 
            matrix_metadata=last_split_matrix_metadata, 
            train_matrix_uuid=last_split_train_matrix_uuid
        )
        
        # create retrained model hash
        retrained_model_hash = self.model_trainer._model_hash(
                self.matrix_storage_engine.get_store(matrix_uuid).metadata,
                class_path=self.model_group_info['model_type'],
                parameters=self.model_group_info['hyperparameters'],
                random_seed=random_seed,
            ) 

        retrained_model_id = self.model_trainer.process_train_task(
            matrix_store=self.matrix_storage_engine.get_store(matrix_uuid), 
            class_path=self.model_group_info['model_type'], 
            parameters=self.model_group_info['hyperparameters'], 
            model_hash=retrained_model_hash, 
            misc_db_parameters=misc_db_parameters, 
            random_seed=random_seed, 
            retrain=True,
            model_group_id=self.model_group_id
        )

        self.retrained_model_hash = retrieve_model_hash_from_id(self.db_engine, retrained_model_id)
        self.retrained_matrix_uuid = matrix_uuid
        self.retrained_model_id = retrained_model_id
        return {'retrained_model_comment': retrained_model_comment, 'retrained_model_id': retrained_model_id}

    def predict(self, prediction_date):
        """Predict forward by creating a matrix using as_of_date = prediction_date and applying the retrained model on it

        Args:
            prediction_date(str)
        """
        cohort_table_name = f"triage_production.cohort_{self.experiment_config['cohort_config']['name']}_predict"

        # 1. Generate cohort
        self.generate_entity_date_table(prediction_date, cohort_table_name)

        # 2. Generate feature aggregations
        collate_aggregations = self.get_collate_aggregations(prediction_date, cohort_table_name)
        self.feature_generator.process_table_tasks(
            self.feature_generator.generate_all_table_tasks(
                collate_aggregations,
                task_type='aggregation'
            )
        )
        # 3. Reconstruct feature disctionary from feature_names and generate imputation
        reconstructed_feature_dict, imputation_table_tasks = self.get_feature_dict_and_imputation_task(
                collate_aggregations, 
                self.retrained_model_id
        )
        self.feature_generator.process_table_tasks(imputation_table_tasks)
 
        # 4. Build matrix
        db_config = {
            "features_schema_name": "triage_production",
            "labels_schema_name": "public",
            "cohort_table_name": cohort_table_name,
        }

        matrix_builder = MatrixBuilder(
            db_config=db_config,
            matrix_storage_engine=self.matrix_storage_engine,
            engine=self.db_engine,
            experiment_hash=None,
            replace=True,
        )
        # Use timechop to get the time definition for production
        # temporal_config = self.experiment_config["temporal_config"]
        temporal_config = self.get_temporal_config_for_retrain(dt_from_str(prediction_date))
        timechopper = Timechop(**temporal_config)
        prod_definitions = timechopper.define_test_matrices(
            train_test_split_time=dt_from_str(prediction_date), 
            test_duration=temporal_config['test_durations'][0],
            test_label_timespan=temporal_config['test_label_timespans'][0]
        )
        last_split_definition = prod_definitions[-1]
        matrix_metadata = Planner.make_metadata(
            matrix_definition=last_split_definition,
            feature_dictionary=reconstructed_feature_dict,
            label_name=self.label_name,
            label_type='binary',
            cohort_name=self.cohort_name,
            matrix_type='production',
            feature_start_time=self.feature_start_time,
            user_metadata=self.user_metadata,
        )
    
        matrix_metadata['matrix_id'] = str(prediction_date) +  f'_model_id_{self.retrained_model_id}' + '_risklist'

        matrix_uuid = filename_friendly_hash(matrix_metadata)
    
        matrix_builder.build_matrix(
            as_of_times=[prediction_date],
            label_name=self.label_name,
            label_type='binary',
            feature_dictionary=reconstructed_feature_dict,
            matrix_metadata=matrix_metadata,
            matrix_uuid=matrix_uuid,
            matrix_type="production",
        )
        
        # 5. Predict the risk score for production
        predictor = Predictor(
            model_storage_engine=self.project_storage.model_storage_engine(),
            db_engine=self.db_engine,
            rank_order='best'
        )

        predictor.predict(
            model_id=self.retrained_model_id,
            matrix_store=self.matrix_storage_engine.get_store(matrix_uuid),
            misc_db_parameters={},
            train_matrix_columns=self.matrix_storage_engine.get_store(self.retrained_matrix_uuid).columns(),
        )
        
        self.predict_matrix_uuid = matrix_uuid

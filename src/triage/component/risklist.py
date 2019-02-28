from triage.component.results_schema import upgrade_db
from triage.component.architect.cohort_table_generators import CohortTableGenerator, DEFAULT_ACTIVE_STATE
from triage.component.architect.features import FeatureGenerator
from triage.component.architect.builders import MatrixBuilder
from triage.component.catwalk.predictors import Predictor
from triage.component import metta
from triage.util.conf import dt_from_str

from collections import OrderedDict
import json
import re


def get_required_info_from_config(db_engine, model_id):
    """Get all information needed to make the risk list from model_id
    Args:
            db_engine (sqlalchemy.db.engine)
            model_id (int) The id of a given model in the database

    Returns: (dict) a dictionary of all information needed for making the risk list

    """
    get_experiment_query = """
        select experiments.config, matrices.matrix_metadata, matrix_uuid
        from model_metadata.experiments
        join model_metadata.experiment_matrices using (experiment_hash)
        join model_metadata.matrices using (matrix_uuid)
        join model_metadata.models on (models.train_matrix_uuid = matrices.matrix_uuid)
        where model_id = %s
    """
    results = list(db_engine.execute(get_experiment_query, model_id))
    experiment_config = results[0]['config']
    label_config = experiment_config['label_config']
    original_matrix_uuid = results[0]['matrix_uuid']
    matrix_metadata = json.loads(results[0]['matrix_metadata'])
    feature_names = matrix_metadata['feature_names']
    feature_config = experiment_config['feature_aggregations']
    cohort_config = experiment_config['cohort_config']
    timechop_config = experiment_config['temporal_config']
    feature_start_time = timechop_config['feature_start_time']

    model_info = {}
    model_info['cohort_config'] = cohort_config
    model_info['feature_config'] = feature_config
    model_info['feature_names'] = feature_names
    model_info['feature_start_time'] = feature_start_time
    model_info['original_matrix_uuid'] = original_matrix_uuid
    model_info['label_config'] = label_config

    return model_info


def generate_risk_list(db_engine, matrix_storage_engine, model_storage_engine, model_id, as_of_date):
    """Generate the risk list based model_id and as_of_date

    Args:
            db_engine (sqlalchemy.db.engine)
            model_id (int) The id of a given model in the database
            matrix_storage_engine (catwalk.storage.matrix_storage_engine)
            model_storage_engine (catwalk.storage.model_storage_engine)

    """
    upgrade_db(db_engine=db_engine)
    # 1. Get feature and cohort config from database
    model_info = get_required_info_from_config(db_engine, model_id)

    # 2. Generate cohort
    cohort_table_name = f"production.cohort_{model_info['cohort_config']['name']}"
    cohort_table_generator = CohortTableGenerator(
        db_engine=db_engine,
        query=model_info['cohort_config']['query'],
        cohort_table_name=cohort_table_name
    )
    cohort_table_generator.generate_cohort_table([dt_from_str(as_of_date)])

    # 3. Generate feature aggregations
    feature_generator = FeatureGenerator(
        db_engine=db_engine,
        features_schema_name="production",
        feature_start_time=model_info['feature_start_time'],
    )
    collate_aggregations = feature_generator.aggregations(
        feature_aggregation_config=model_info['feature_config'],
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
    reconstructed_feature_dictionary = {}
    imputation_table_tasks = OrderedDict()
    with db_engine.begin() as conn:
        for aggregation in collate_aggregations:
            feature_prefix = aggregation.prefix
            feature_group = aggregation.get_table_name(imputed=True).split('.')[1].replace('"', '')
            feature_names_in_group = [f for f in model_info['feature_names'] if re.match(f'\A{feature_prefix}', f)]
            reconstructed_feature_dictionary[feature_group] = feature_names_in_group

            # Make sure that the features imputed in training should also be imputed in production
            features_imputed_in_train = [f for f in set(feature_names_in_group) if f + '_imp' in feature_names_in_group]
            results = conn.execute(aggregation.find_nulls())
            null_counts = results.first().items()

            features_imputed_in_production = [col for (col, val) in null_counts if val > 0]
            total_impute_cols = set(features_imputed_in_production) | set(features_imputed_in_train)
            total_nonimpute_cols = set(f for f in set(feature_names_in_group) if '_imp' not in f) - total_impute_cols
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
        "features_schema_name": "production",
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

    matrix_metadata = {
        'as_of_times': [as_of_date],
        'matrix_id': str(as_of_date) + '_prediction',
        'state': DEFAULT_ACTIVE_STATE,
        'test_duration': '1y',
        'matrix_type': 'production',
        'label_timespan': None,
        'label_name': model_info['label_config']['name'],
        'indices': ["entity_id", "as_of_date"],
        'feature_start_time': model_info['feature_start_time'],
    }

    matrix_uuid = metta.generate_uuid(matrix_metadata)

    matrix_builder.build_matrix(
        as_of_times=[as_of_date],
        label_name=model_info['label_config']['name'],
        label_type=None,
        feature_dictionary=reconstructed_feature_dictionary,
        matrix_metadata=matrix_metadata,
        matrix_uuid=matrix_uuid,
        matrix_type="production",
    )

    # 6. Predict the risk score for production
    predictor = Predictor(
        model_storage_engine=model_storage_engine,
        db_engine=db_engine
    )

    predictor.predict(
        model_id=model_id,
        matrix_store=matrix_storage_engine.get_store(matrix_uuid),
        misc_db_parameters={},
        train_matrix_columns=matrix_storage_engine.get_store(model_info['original_matrix_uuid']).columns()
    )

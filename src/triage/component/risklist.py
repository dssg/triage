from triage.component.results_schema import upgrade_db
from triage.component.architect.cohort_table_generators import CohortTableGenerator, DEFAULT_ACTIVE_STATE
from triage.component.architect.features import FeatureGenerator, FeatureGroupCreator, FeatureGroupMixer, FeatureDictionaryCreator
from triage.component.architect.builders import MatrixBuilder
from triage.component.catwalk.predictors import Predictor
from triage.component import metta
from triage.util.conf import dt_from_str

import json
import re


def generate_risk_list(db_engine, matrix_storage_engine, model_storage_engine, model_id, as_of_date):
    upgrade_db(db_engine=db_engine)
    # 1. get feature and cohort config from database
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
    original_matrix_uuid = results[0]['matrix_uuid']
    matrix_metadata = json.loads(results[0]['matrix_metadata'])
    feature_config = experiment_config['feature_aggregations']
    cohort_config = experiment_config['cohort_config']
    timechop_config = experiment_config['temporal_config']
    feature_start_time = timechop_config['feature_start_time']
    feature_group = matrix_metadata['feature_groups']
    print(feature_group)
    # Convert feature_group (list of string) to dictionary
    f_dict = {}
    for fg in feature_group:
        key, v = re.split(r'\W+', fg)
        f_dict[key] = v
    feature_group = f_dict

    cohort_table_name = f"production.cohort_{cohort_config['name']}"
    cohort_table_generator = CohortTableGenerator(
        db_engine=db_engine,
        query=cohort_config['query'],
        cohort_table_name=cohort_table_name
    )
    feature_generator = FeatureGenerator(
        db_engine=db_engine,
        features_schema_name="production",
        feature_start_time=feature_start_time,
    )
    feature_dictionary_creator = FeatureDictionaryCreator(
        features_schema_name="production", db_engine=db_engine
    )
    feature_group_creator = FeatureGroupCreator(feature_group)
    cohort_table_generator.generate_cohort_table([dt_from_str(as_of_date)])
    collate_aggregations = feature_generator.aggregations(
        feature_aggregation_config=feature_config,
        feature_dates=[as_of_date],
        state_table=cohort_table_name
    )
    feature_generator.process_table_tasks(
        feature_generator.generate_all_table_tasks(
            collate_aggregations,
            task_type='aggregation'
        )
    )
    imputation_table_tasks = feature_generator.generate_all_table_tasks(
        collate_aggregations,
        task_type='imputation'
    )
    feature_generator.process_table_tasks(imputation_table_tasks)
    feature_dictionary = feature_dictionary_creator.feature_dictionary(
        feature_table_names=imputation_table_tasks.keys(),
        index_column_lookup=feature_generator.index_column_lookup(
            collate_aggregations
        ),
    )

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

    feature_groups = feature_group_creator.subsets(feature_dictionary)
    print(feature_groups)
    master_feature_dict = FeatureGroupMixer(["all"]).generate(feature_groups)[0]
    print(master_feature_dict)
    for f in master_feature_dict['zip_code_features_aggregation_imputed']:
        print(f)
    matrix_metadata = {
        'as_of_times': [as_of_date],
        'matrix_id': str(as_of_date) + '_prediction',
        'state': DEFAULT_ACTIVE_STATE,
        'test_duration': '1y',
        'matrix_type': 'production',
        'label_timespan': None,
        'indices': ["entity_id", "as_of_date"],
        'feature_start_time': feature_start_time,
    }

    matrix_uuid = metta.generate_uuid(matrix_metadata)

    matrix_builder.build_matrix(
        as_of_times=[as_of_date],
        label_name=None,
        label_type=None,
        feature_dictionary=master_feature_dict,
        matrix_metadata=matrix_metadata,
        matrix_uuid=matrix_uuid,
        matrix_type="production",
    )

    predictor = Predictor(
        model_storage_engine=model_storage_engine,
        db_engine=db_engine
    )


    predictor.predict(
        model_id=model_id,
        matrix_store=matrix_storage_engine.get_store(matrix_uuid),
        misc_db_parameters={},
        train_matrix_columns=matrix_storage_engine.get_store(original_matrix_uuid).columns()
    )


from triage.component.results_schema import upgrade_db
from triage.component.architect.cohort_table_generators import CohortTableGenerator
from triage.component.architect.features import FeatureGenerator, FeatureGroupCreator, FeatureGroupMixer, FeatureDictionaryCreator
from triage.util.conf import dt_from_str
import json

def generate_risk_list(db_engine, model_id, as_of_date):
    upgrade_db(db_engine=db_engine)
    # 1. get feature and cohort config from database
    get_experiment_query = """
        select experiments.config, matrices.matrix_metadata
        from model_metadata.experiments
        join model_metadata.experiment_matrices using (experiment_hash)
        join model_metadata.matrices using (matrix_uuid)
        join model_metadata.models on (models.train_matrix_uuid = matrices.matrix_uuid)
        where model_id = %s
    """
    results = list(db_engine.execute(get_experiment_query, model_id))
    experiment_config = results[0]['config']
    matrix_metadata = json.loads(results[0]['matrix_metadata'])
    feature_config = experiment_config['feature_aggregations']
    cohort_config = experiment_config['cohort_config']
    timechop_config = experiment_config['temporal_config']
    feature_start_time = timechop_config['feature_start_time']
    feature_group = matrix_metadata['feature_groups']
    print(type(feature_group))
    print(feature_group)
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
    feature_group_creator = FeatureGroupCreator(feature_group[0])

    cohort_table_generator.generate_cohort_table([dt_from_str(as_of_date)])
    collate_aggregations = feature_generator.aggregations(
        feature_aggregation_config=feature_config,
        feature_dates=[as_of_date],
        state_table=cohort_table_name
    )
    feature_generator.process_table_tasks(feature_generator.generate_all_table_tasks(collate_aggregations, task_type='aggregation'))
    imputation_table_tasks = feature_generator.generate_all_table_tasks(collate_aggregations, task_type='imputation')
    feature_generator.process_table_tasks(imputation_table_tasks)
    feature_dictionary = feature_dictionary_creator.feature_dictionary(
        feature_table_names=imputation_table_tasks.keys(),
        index_column_lookup=feature_generator.index_column_lookup(
            collate_aggregations
        ),
    )
    smaller_dict = feature_group_creator.subsets(feature_dictionary)
    print(feature_dictionary)

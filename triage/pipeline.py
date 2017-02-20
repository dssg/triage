from triage.db import ensure_db
from triage.utils import temporal_splits
from triage.label_generators import LabelGenerator
from triage.feature_generators import FeatureGenerator
from triage.model_trainers import ModelTrainer
from triage.predictors import Predictor
from triage.scoring import ModelScorer
from triage.storage import InMemoryMatrixStore
from triage.timechop import make_entity_dates_table,\
    build_feature_query,\
    write_to_csv,\
    get_feature_names
import logging
import pandas
from dateutil.relativedelta import relativedelta


def timechop_spliterator(split_config):
    return temporal_splits(
        split_config['start_time'],
        split_config['end_time'],
        split_config['update_window'],
        split_config['prediction_windows'],
        split_config['feature_frequency'],
        split_config['test_frequency']
    )


def timechop_foreal_matrices(
    train_labels_table,
    test_labels_table,
    feature_tables,
    split,
    db_engine
):
    make_entity_dates_table(
        db_engine,
        split['train_as_of_dates'],
        feature_tables,
        'public'
    )

    for feature_table in feature_tables:
        feature_names = get_feature_names(feature_table, 'public', db_engine)
        feature_query = build_feature_query(
            feature_table,
            feature_names,
            'public',
            split['train_as_of_dates'],
            db_engine
        )
        write_to_csv(
            feature_query,
            '{}.csv'.format(feature_table),
            db_engine
        )


def timechop_get_matrix(
    labels_table,
    feature_tables,
    as_of_dates,
    outcome_column,
    entity_column,
    start,
    end,
    matrix_name,
    prediction_window,
    db_engine
):
    tbl_name = feature_tables[0].strip('"')
    feature_names = [
        '"{}"'.format(name)
        for name in get_feature_names(
            tbl_name,
            entity_column,
            'public',
            db_engine
        )
    ]
    logging.debug(feature_names)
    as_of_dates_string = ", ".join(
        ["'{}'".format(as_of_date) for as_of_date in as_of_dates]
    )
    query = """select features.{entity_column},
        {feature_names},
        labels.{outcome_column}
        from {tbl_name} features
        join {labels_table} labels on (labels.entity_id = features.{entity_column})
        where date in ({as_of_dates_string})
    """.format(
        feature_names=','.join(feature_names),
        outcome_column=outcome_column,
        tbl_name=tbl_name,
        labels_table=labels_table,
        entity_column=entity_column,
        as_of_dates_string=as_of_dates_string
    )
    logging.debug(query)
    matrix = pandas.read_sql(
        query,
        con=db_engine
    ).set_index(entity_column)
    if matrix.empty:
        raise ValueError('Matrix empty')

    return InMemoryMatrixStore(
        matrix, {
            'start_time': start,
            'end_time': end,
            'label_name': outcome_column,
            'feature_names': feature_names,
            'matrix_id': '{}_{}_{}'.format(
                matrix_name,
                start,
                end
            ),
            'as_of_dates': as_of_dates,
            'prediction_window': prediction_window
        }
    )


class Pipeline(object):
    def __init__(self, config, db_engine, model_storage_class, project_path):
        self.config = config
        self.db_engine = db_engine
        self.model_storage_engine =\
            model_storage_class(project_path=project_path)
        self.project_path = project_path
        ensure_db(self.db_engine)

    def run(self):
        # 1. generate temporal splits
        for split in timechop_spliterator(self.config['temporal_splits']):
            logging.info('Processing split %s', split)

            # 2. create labels
            logging.debug('---------------------')
            logging.debug('---------LABEL GENERATION------------')
            logging.debug('---------------------')
            label_generator = LabelGenerator(
                events_table=self.config['events_table'],
                db_engine=self.db_engine
            )

            labels_table = 'labels'
            self.db_engine.execute(
                'drop table if exists {}'.format(labels_table)
            )
            self.db_engine.execute('''
                create table {} (
                entity_id int,
                outcome_date date,
                outcome bool
            )'''.format(labels_table))

            all_as_of_dates = set(
                split['train_as_of_dates'] + split['test_as_of_dates']
            )
            prediction_delta = relativedelta(months=split['prediction_window'])
            for as_of_date in all_as_of_dates:
                logging.info('Creating labels for %s', as_of_date)
                label_generator.generate(
                    start_date=as_of_date,
                    end_date=as_of_date + prediction_delta,
                    labels_table=labels_table
                )

            # 3. generate features
            logging.info('Generating features for %s', all_as_of_dates)
            feature_tables = FeatureGenerator(
                db_engine=self.db_engine
            ).generate(
                feature_aggregations=self.config['feature_aggregations'],
                feature_dates=all_as_of_dates,
            )

            # 4. create training and test sets
            logging.info('Creating matrices from %s', feature_tables)
            logging.debug('---------------------')
            logging.debug('---------MATRIX GENERATION------------')
            logging.debug('---------------------')
            train_store = timechop_get_matrix(
                labels_table,
                feature_tables,
                split['train_as_of_dates'],
                'outcome',
                self.config['entity_column_name'],
                split['train_start'],
                split['train_end'],
                'train',
                split['prediction_window'],
                self.db_engine
            )

            if len(train_store.labels().unique()) == 1:
                logging.warning('''Train Matrix for split %s had only one
                unique value, no point in training this model. Skipping
                ''', split)
                continue

            trainer = ModelTrainer(
                project_path=self.project_path,
                model_storage_engine=self.model_storage_engine,
                matrix_store=train_store,
                db_engine=self.db_engine
            )

            predictor = Predictor(
                project_path=self.project_path,
                model_storage_engine=self.model_storage_engine,
                db_engine=self.db_engine
            )
            model_scorer = ModelScorer(
                metric_groups=self.config['scoring'],
                db_engine=self.db_engine
            )

            logging.info('Training models')
            model_ids = trainer.train_models(
                grid_config=self.config['grid_config'],
                misc_db_parameters=dict(test=False)
            )
            logging.info('Done training models')

            for as_of_date in split['test_as_of_dates']:
                logging.info('Testing and scoring as_of_date %s', as_of_date)
                test_store = timechop_get_matrix(
                    labels_table,
                    feature_tables,
                    [as_of_date],
                    'outcome',
                    self.config['entity_column_name'],
                    as_of_date,
                    as_of_date,
                    'test',
                    split['prediction_window'],
                    self.db_engine
                )
                for model_id in model_ids:
                    logging.info('Testing model id %s', model_id)
                    predictions, predictions_proba = predictor.predict(
                        model_id,
                        test_store,
                        misc_db_parameters=dict()
                    )

                    model_scorer.score(
                        predictions_proba,
                        predictions,
                        test_store.labels(),
                        model_id,
                        as_of_date
                    )

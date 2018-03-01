import logging

from triage.experiments import ExperimentBase


class SingleThreadedExperiment(ExperimentBase):
    def build_matrices(self):
        logging.info('Creating sparse states')
        self.generate_sparse_states()
        logging.info('Creating labels')
        self.generate_labels()
        logging.info('Creating feature aggregation tables')
        self.feature_generator.process_table_tasks(self.feature_aggregation_table_tasks)
        logging.info('Creating feature imputation tables')
        self.feature_generator.process_table_tasks(self.feature_imputation_table_tasks)
        logging.info('Building all matrices')
        self.matrix_builder.build_all_matrices(self.matrix_build_tasks)

    def catwalk(self):
        for split_num, split in enumerate(self.full_matrix_definitions):
            self.log_split(split_num, split)
            train_store = self.matrix_store(split['train_uuid'])
            if train_store.empty:
                logging.warning('''Train matrix for split %s was empty,
                no point in training this model. Skipping
                ''', split['train_uuid'])
                continue
            if len(train_store.labels().unique()) == 1:
                logging.warning('''Train Matrix for split %s had only one
                unique value, no point in training this model. Skipping
                ''', split['train_uuid'])
                continue

            logging.info('Training models')
            model_ids = self.trainer.train_models(
                grid_config=self.config['grid_config'],
                misc_db_parameters=dict(
                    test=False,
                    model_comment=self.config.get('model_comment', None),
                ),
                matrix_store=train_store
            )
            logging.info('Done training models')

            for split_def, test_uuid in zip(
                split['test_matrices'],
                split['test_uuids']
            ):
                as_of_times = split_def['as_of_times']
                logging.info(
                    'Testing and scoring as_of_times min: %s max: %s num: %s',
                    min(as_of_times),
                    max(as_of_times),
                    len(as_of_times)
                )
                test_store = self.matrix_store(test_uuid)
                if test_store.empty:
                    logging.warning('''Test matrix for uuid %s
                    was empty, no point in generating predictions. Skipping.
                    ''', test_uuid)
                    continue
                for model_id in model_ids:
                    logging.info('Testing model id %s', model_id)
                    predictions_proba = self.predictor.predict(
                        model_id,
                        test_store,
                        misc_db_parameters=dict(),
                        train_matrix_columns=train_store.columns(),
                    )

                    self.individual_importance_calculator\
                        .calculate_and_save_all_methods_and_dates(
                            model_id,
                            test_store
                        )

                    self.evaluator.evaluate(
                        predictions_proba=predictions_proba,
                        labels=test_store.labels(),
                        model_id=model_id,
                        # for evaluation range, using first to last as of time:
                        evaluation_start_time=split_def['first_as_of_time'],
                        evaluation_end_time=split_def['last_as_of_time'],
                        as_of_date_frequency=split_def['test_as_of_date_frequency']
                    )

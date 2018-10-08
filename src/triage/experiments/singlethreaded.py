from triage.experiments import ExperimentBase


class SingleThreadedExperiment(ExperimentBase):
    def process_query_tasks(self, query_tasks):
        self.feature_generator.process_table_tasks(query_tasks)

    def process_matrix_build_tasks(self, matrix_build_tasks):
        self.matrix_builder.build_all_matrices(matrix_build_tasks)

    def process_train_tasks(self, train_tasks):
        return [
            self.trainer.process_train_task(**train_task) for train_task in train_tasks
        ]

    def process_model_test_tasks(self, test_tasks):
        return [
            self.tester.process_model_test_task(**test_task) for test_task in test_tasks
        ]

from triage.experiments import ExperimentBase
from triage.util.db import run_statements


class SingleThreadedExperiment(ExperimentBase):
    def process_inserts(self, inserts):
        run_statements(inserts, self.db_engine)

    def process_matrix_build_tasks(self, matrix_build_tasks):
        self.matrix_builder.build_all_matrices(matrix_build_tasks)

    def process_train_test_batches(self, batches):
        self.model_train_tester.process_all_batches(batches)

    def process_subset_tasks(self, subset_tasks):
        self.subsetter.process_all_tasks(subset_tasks)

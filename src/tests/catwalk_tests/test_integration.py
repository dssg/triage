from triage.component.catwalk import ModelTrainTester, Predictor, ModelTrainer, ModelEvaluator, IndividualImportanceCalculator, ProtectedGroupsGenerator
from triage.component.catwalk.utils import save_experiment_and_get_hash
from triage.component.catwalk.model_trainers import flatten_grid_config
from triage.component.catwalk.storage import (
    ModelStorageEngine,
    MatrixStore,
    MatrixStorageEngine,
)
from triage.tracking import initialize_tracking_and_get_run_id
from tests.utils import (
    get_matrix_store,
    matrix_metadata_creator,
)

from unittest.mock import patch, MagicMock


def test_ModelTrainTester_generate_tasks(db_engine_with_results_schema, project_storage, sample_timechop_splits, sample_grid_config):
    db_engine = db_engine_with_results_schema
    model_storage_engine = ModelStorageEngine(project_storage)
    matrix_storage_engine = MatrixStorageEngine(project_storage)
    sample_matrix_store = get_matrix_store(project_storage)
    experiment_hash = save_experiment_and_get_hash({}, db_engine)
    run_id = initialize_tracking_and_get_run_id(
        experiment_hash,
        experiment_class_path="",
        random_seed=5,
        experiment_kwargs={},
        db_engine=db_engine_with_results_schema
    )
    # instantiate pipeline objects
    trainer = ModelTrainer(
        experiment_hash=experiment_hash,
        model_storage_engine=model_storage_engine,
        db_engine=db_engine,
        run_id=run_id,
    )
    train_tester = ModelTrainTester(
        matrix_storage_engine=matrix_storage_engine,
        model_trainer=trainer,
        model_evaluator=None,
        individual_importance_calculator=None,
        predictor=None,
        subsets=None,
        protected_groups_generator=None,
    )
    with patch.object(matrix_storage_engine, 'get_store', return_value=sample_matrix_store):
        batches = train_tester.generate_task_batches(
            splits=sample_timechop_splits,
            grid_config=sample_grid_config
        )
        assert len(batches) == 3
        # we expect to have a task for each combination of split and classifier
        flattened_tasks = list(task for batch in batches for task in batch.tasks)
        assert len(flattened_tasks) == \
            len(sample_timechop_splits) * len(list(flatten_grid_config(sample_grid_config)))
        # we also expect each task to match the call signature of process_task
        with patch.object(train_tester, 'process_task', autospec=True):
            for task in flattened_tasks:
                train_tester.process_task(**task)


def setup_model_train_tester(project_storage, replace, additional_bigtrain_classnames=None):
    matrix_storage_engine = MatrixStorageEngine(project_storage)
    train_matrix_store = get_matrix_store(
        project_storage,
        metadata=matrix_metadata_creator(matrix_type="train"),
        write_to_db=False
    )
    test_matrix_store = get_matrix_store(
        project_storage,
        metadata=matrix_metadata_creator(matrix_type="test"),
        write_to_db=False
    )
    sample_train_kwargs = {
        'matrix_store': train_matrix_store,
        'class_path': None,
        'parameters': {},
        'model_hash': None,
        'misc_db_parameters': {}
    }
    train_test_task = {
        'train_kwargs': sample_train_kwargs,
        'train_store': train_matrix_store,
        'test_store': test_matrix_store
    }

    predictor = MagicMock(spec_set=Predictor)
    trainer = MagicMock(spec_set=ModelTrainer)
    evaluator = MagicMock(spec_set=ModelEvaluator)
    individual_importance_calculator = MagicMock(spec_set=IndividualImportanceCalculator)
    protected_groups_generator = MagicMock(spec_set=ProtectedGroupsGenerator)
    train_tester = ModelTrainTester(
        matrix_storage_engine=matrix_storage_engine,
        model_trainer=trainer,
        model_evaluator=evaluator,
        individual_importance_calculator=individual_importance_calculator,
        predictor=predictor,
        subsets=[],
        replace=replace,
        protected_groups_generator=protected_groups_generator,
        additional_bigtrain_classnames=additional_bigtrain_classnames
    )
    return train_tester, train_test_task


def test_ModelTrainTester_process_task_replace_False_needs_evaluations(project_storage):
    train_tester, train_test_task = setup_model_train_tester(project_storage, replace=False)
    train_tester.model_evaluator.needs_evaluations.return_value = True
    train_tester.process_task(**train_test_task)
    assert train_tester.model_evaluator.needs_evaluations.call_count == 2
    assert train_tester.predictor.predict.call_count == 2
    assert train_tester.model_evaluator.evaluate.call_count == 2
    assert train_tester.protected_groups_generator.as_dataframe.call_count == 2


def test_ModelTrainTester_process_task_replace_False_no_evaluations(project_storage):
    train_tester, train_test_task = setup_model_train_tester(project_storage, replace=False)
    train_tester.model_evaluator.needs_evaluations.return_value = False
    train_tester.process_task(**train_test_task)
    assert train_tester.model_evaluator.needs_evaluations.call_count == 2
    assert train_tester.predictor.predict.call_count == 0
    assert train_tester.model_evaluator.evaluate.call_count == 0
    assert train_tester.protected_groups_generator.as_dataframe.call_count == 0


def test_ModelTrainTester_process_task_replace_True(project_storage):
    train_tester, train_test_task = setup_model_train_tester(project_storage, replace=True)
    train_tester.process_task(**train_test_task)
    assert train_tester.model_evaluator.needs_evaluations.call_count == 0
    assert train_tester.predictor.predict.call_count == 2
    assert train_tester.model_evaluator.evaluate.call_count == 2
    assert train_tester.protected_groups_generator.as_dataframe.call_count == 2


def test_ModelTrainTester_process_task_empty_train(project_storage):
    train_tester, train_test_task = setup_model_train_tester(project_storage, replace=True)
    train_store = MagicMock()
    train_store.empty = True
    train_test_task['train_store'] = train_store
    train_tester.process_task(**train_test_task)

    assert train_tester.model_trainer.process_train_task.call_count == 0
    assert train_tester.model_evaluator.needs_evaluations.call_count == 0
    assert train_tester.predictor.predict.call_count == 0
    assert train_tester.model_evaluator.evaluate.call_count == 0
    assert train_tester.protected_groups_generator.as_dataframe.call_count == 0

def test_ModelTrainTester_order_and_batch_tasks(project_storage):
    train_tester, sample_train_test_task = setup_model_train_tester(project_storage, replace=True)
    train_classpaths = [
        'triage.component.catwalk.estimators.classifiers.ScaledLogisticRegression',
        'sklearn.ensemble.RandomForestClassifier',
        'someclass.OtherClassifier'
    ]
    train_test_tasks = [{
            'train_kwargs': {
                'class_path': classpath,
                'parameters': {},
                'model_hash': None,
                'misc_db_parameters': {}
            },
            'train_store': sample_train_test_task['train_store'],
            'test_store': sample_train_test_task['test_store']
        }
        for classpath in train_classpaths
    ]
    batches = train_tester.order_and_batch_tasks(train_test_tasks)
    assert len(batches) == 3
    assert len(batches[0].tasks) == 1
    assert batches[0].tasks[0]['train_kwargs']['class_path'] == 'triage.component.catwalk.estimators.classifiers.ScaledLogisticRegression'
    assert len(batches[1].tasks) == 1
    assert batches[1].tasks[0]['train_kwargs']['class_path'] == 'sklearn.ensemble.RandomForestClassifier'
    assert len(batches[2].tasks) == 1
    assert batches[2].tasks[0]['train_kwargs']['class_path'] == 'someclass.OtherClassifier'


def test_ModelTrainTester_order_and_batch_tasks_allows_additional(project_storage):
    train_tester, sample_train_test_task = setup_model_train_tester(
        project_storage,
        replace=True,
        additional_bigtrain_classnames=['someclass.OtherClassifier']
    )
    train_classpaths = [
        'triage.component.catwalk.estimators.classifiers.ScaledLogisticRegression',
        'sklearn.ensemble.RandomForestClassifier',
        'someclass.OtherClassifier'
    ]
    train_test_tasks = [{
            'train_kwargs': {
                'class_path': classpath,
                'parameters': {},
                'model_hash': None,
                'misc_db_parameters': {}
            },
            'train_store': sample_train_test_task['train_store'],
            'test_store': sample_train_test_task['test_store']
        }
        for classpath in train_classpaths
    ]
    batches = train_tester.order_and_batch_tasks(train_test_tasks)
    assert len(batches) == 3
    assert len(batches[0].tasks) == 1
    assert batches[0].tasks[0]['train_kwargs']['class_path'] == 'triage.component.catwalk.estimators.classifiers.ScaledLogisticRegression'
    assert len(batches[1].tasks) == 2
    assert batches[1].tasks[0]['train_kwargs']['class_path'] == 'sklearn.ensemble.RandomForestClassifier'
    assert batches[1].tasks[1]['train_kwargs']['class_path'] == 'someclass.OtherClassifier'
    assert len(batches[2].tasks) == 0

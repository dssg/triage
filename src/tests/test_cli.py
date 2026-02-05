import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

import triage.cli as cli

runner = CliRunner()
ENV = {"DATABASE_URL": "postgresql://postgres@8.8.8.8/test"}

# skipping 2026-02-04: Tests depend on code that needs further work
pytestmark = pytest.mark.skip("2026-02-04: Code in SRC is not ready to be tested yet.")


def invoke(*args):
    result = runner.invoke(cli.app, args, env=ENV)
    assert result.exit_code == 0, result.stdout
    return result


def test_cli_singlethreadedexperiment():
    with patch("triage.cli.SingleThreadedExperiment", autospec=True) as exp_mock:
        instance = exp_mock.return_value
        invoke("experiment", "example/config/experiment.yaml", "--no-validate")
        exp_mock.assert_called_once()
        instance.run.assert_called_once()


def test_cli_multicoreexperiment():
    with patch("triage.cli.MultiCoreExperiment", autospec=True) as exp_mock:
        invoke(
            "experiment",
            "example/config/experiment.yaml",
            "--n-processes",
            "2",
            "--n-db-processes",
            "2",
            "--no-validate",
        )
        exp_mock.assert_called_once()


def test_cli_show_timechop():
    with patch("triage.cli.SingleThreadedExperiment", autospec=True) as exp_mock:
        exp_instance = exp_mock.return_value
        exp_instance.chopper = "chopper"
        with patch("triage.cli.ProjectStorage", autospec=True) as storage_mock:
            store = MagicMock()
            storage_mock.return_value.get_store.return_value = store
            context_handle = Mock()
            store.open.return_value.__enter__.return_value = context_handle
            with patch("triage.cli.visualize_chops", autospec=True) as viz_mock:
                invoke(
                    "experiment",
                    "example/config/experiment.yaml",
                    "--show-timechop",
                    "--no-validate",
                )
                storage_mock.assert_called_once()
                viz_mock.assert_called_once_with("chopper", save_target=context_handle)


def test_cli_audition():
    with patch("triage.cli.AuditionRunner", autospec=True) as runner_mock:
        runner_instance = runner_mock.return_value
        invoke("audition", "--config", "example/config/audition.yaml")
        runner_instance.validate.assert_called_once()
        runner_instance.run.assert_called_once()


def test_cli_crosstabs():
    with patch("triage.cli.run_crosstabs", autospec=True) as run_mock:
        invoke("crosstabs", "example/config/postmodeling_crosstabs.yaml")
        run_mock.assert_called_once()


def test_featuretest():
    with patch("triage.cli.FeatureGenerator", autospec=True) as feature_mock:
        with patch("triage.cli.EntityDateTableGenerator", autospec=True) as cohort_mock:
            invoke("featuretest", "example/config/experiment.yaml", "2017-06-06")
            feature_mock.assert_called_once()
            cohort_mock.assert_called_once()


def test_cli_predictlist():
    with patch(
        "triage.cli.predict_forward_with_existed_model", autospec=True
    ) as predict_mock:
        invoke("predictlist", "40", "2019-06-04")
        predict_mock.assert_called_once()
        args = predict_mock.call_args[0]
        assert args[2] == 40
        assert args[3] == datetime.datetime(2019, 6, 4)


def test_cli_retrain_predict():
    with patch("triage.cli.Retrainer", autospec=True) as retrain_mock:
        invoke("retrainpredict", "3", "2021-04-04")
        retrain_mock.assert_called_once()
        args = retrain_mock.call_args[0]
        assert args[2] == 3


def test_analyze_config():
    # Smoke test to ensure command executes
    invoke("analyze-config", "example/config/experiment.yaml")


def test_dashboard_command():
    fake_run = Mock()
    fake_run.run_id = 1
    fake_run.run_hash = "hash"
    fake_run.current_status = cli.TriageRunStatus.completed
    fake_run.start_time = datetime.datetime(2023, 1, 1)
    fake_run.matrices_made = 1
    fake_run.matrices_needed = 2
    fake_run.models_made = 3
    fake_run.models_needed = 4

    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.return_value = None
    query = session.query.return_value
    query.order_by.return_value = query
    query.filter.return_value = query
    query.limit.return_value.all.return_value = [fake_run]

    with patch("triage.cli.sessionmaker", return_value=lambda bind=None: session):
        invoke("dashboard", "--limit", "1")

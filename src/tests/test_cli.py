import argcmdr
import triage.cli as cli
from unittest.mock import Mock, patch
import os


# we do not need a real database URL but one SQLalchemy thinks looks like a real one
@patch.dict(os.environ, {'DATABASE_URL': 'postgresql://postgres@8.8.8.8/test'})
def try_command(*argv):
    try:
        argcmdr.main(cli.Triage, argv=('--tb',) + argv)
    except SystemExit as exc:
        raise AssertionError(exc)


def test_cli_singlethreadedexperiment():
    with patch('triage.cli.SingleThreadedExperiment', autospec=True) as mock:
        try_command('experiment', 'example/config/experiment.yaml')
        mock.assert_called_once()


def test_cli_multicoreexperiment():
    with patch('triage.cli.MultiCoreExperiment', autospec=True) as mock:
        try_command('experiment', 'example/config/experiment.yaml', '--n-processes', '2')
        mock.assert_called_once()


def test_cli_show_timechop():
    with patch('triage.cli.SingleThreadedExperiment', autospec=True) as exp_mock:
        exp_instance_mock = Mock()
        exp_mock.return_value = exp_instance_mock
        exp_instance_mock.configure_mock(chopper='chopper')
        with patch('triage.cli.ProjectStorage', autospec=True) as ps_mock:
            with patch('triage.cli.visualize_chops', autospec=True) as viz_mock:
                try_command('experiment', 'example/config/experiment.yaml', '--show-timechop')
                ps_mock.assert_called_once()
                viz_mock.assert_called_once()
                assert viz_mock.call_args[1]


def test_cli_audition():
    with patch('triage.cli.AuditionRunner', autospec=True) as mock:
        try_command('audition', '-c', 'example/config/audition.yaml')
        mock.assert_called_once()


def test_cli_crosstabs():
    with patch('triage.cli.run_crosstabs', autospec=True) as mock:
        try_command('crosstabs', 'example/config/postmodeling_crosstabs.yaml')
        mock.assert_called_once()


def test_featuretest():
    with patch('triage.cli.FeatureGenerator', autospec=True) as featuremock:
        with patch('triage.cli.EntityDateTableGenerator', autospec=True) as cohortmock:
            try_command('featuretest', 'example/config/experiment.yaml', '2017-06-06')
            featuremock.assert_called_once()
            cohortmock.assert_called_once()

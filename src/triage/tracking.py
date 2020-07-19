import sys
import datetime
import platform
import getpass
import os
import requests
import subprocess
import verboselogs, logging
logger = verboselogs.VerboseLogger(__name__)
from functools import wraps
from triage.util.db import scoped_session, get_for_update
from triage.util.introspection import classpath
from triage import __version__

try:
    try:
        from pip._internal.operations import freeze as pip_freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze as pip_freeze
except ImportError:
    pip_freeze = None


from triage.component.results_schema import ExperimentRun, ExperimentRunStatus


def infer_git_hash():
    """Attempt to infer the git hash of the repository in the current working directory

    Returns: Either the 'git rev-parse HEAD' output or None
    """
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except Exception as exc:
        logger.spam("Unable to infer git hash")
        git_hash = None
    return git_hash


def infer_triage_version():
    return __version__

def infer_python_version():
    """ Returns python version """
    return sys.version.replace("\r", "").replace("\n", "")

def infer_installed_libraries():
    """Attempt to infer the installed libraries by running pip freeze and formatting as a list

    Returns: Either a list, or None
    """
    if pip_freeze is not None:
        installed_libraries = pip_freeze.freeze()
    else:
        logger.spam("Unable to pip freeze, cannot list installed libraries")
        installed_libraries = []
    return installed_libraries


def infer_ec2_instance_type():
    """Attempt to infer the instance type of the ec2 instance by querying Amazon's endpoint

    Returns: Either the ec2 instance type as returned by Amazon's endpoint, or None
    """
    try:
        ec2_instance_type = requests.get(
            'http://169.254.169.254/latest/meta-data/instance-type',
            timeout=0.01
        ).text
    except requests.exceptions.RequestException:
        logger.spam(
            "Unable to retrieve metadata about ec2 instance, will not set ec2 instance type"
        )
        ec2_instance_type = None
    return ec2_instance_type


def infer_log_location():
    """Attempt to infer the location of the log file of the root logger

    Returns: Either the baseFilename of the first FileHandler on the root logger, or None
    """
    root_logger_handlers = [
        handler
        for handler in logging.getLoggerClass().root.handlers
        if isinstance(handler, logging.FileHandler)
    ]
    if root_logger_handlers:
        log_location = root_logger_handlers[0].baseFilename
    else:
        logger.spam("No FileHandler found in root logger, cannot record logging filename")
        log_location = None
    return log_location


def initialize_tracking_and_get_run_id(
    experiment_hash,
    experiment_class_path,
    random_seed,
    experiment_kwargs,
    db_engine
):
    """Create a row in the ExperimentRun table with some initial info and return the created run_id

    Args:
        experiment_hash (str) An experiment hash that exists in the experiments table
        experiment_class_path (str) The name of the experiment subclass used
        random_seed (int) Random seed used to run the experiment
        experiment_kwargs (dict) Any runtime Experiment keyword arguments that should be saved
        db_engine (sqlalchemy.engine)
    """
    # Any experiment kwargs that are types (e.g. MatrixStorageClass) can't
    # be serialized, so just use the class name if so
    cleaned_experiment_kwargs = {
        k: (classpath(v) if isinstance(v, type) else v)
        for k, v in experiment_kwargs.items()
    }
    run = ExperimentRun(
        start_time=datetime.datetime.now(),
        git_hash=infer_git_hash(),
        triage_version=infer_triage_version(),
        python_version=infer_python_version(),
        experiment_hash=experiment_hash,
        last_updated_time=datetime.datetime.now(),
        current_status=ExperimentRunStatus.started,
        installed_libraries=infer_installed_libraries(),
        platform=platform.platform(),
        os_user=getpass.getuser(),
        working_directory=os.getcwd(),
        ec2_instance_type=infer_ec2_instance_type(),
        log_location=infer_log_location(),
        experiment_class_path=experiment_class_path,
        random_seed = random_seed,
        experiment_kwargs=cleaned_experiment_kwargs,
    )
    run_id = None
    with scoped_session(db_engine) as session:
        session.add(run)
        session.commit()
        run_id = run.run_id
    if not run_id:
        raise ValueError("Failed to retrieve run_id from saved row")
    return run_id


def get_run_for_update(db_engine, run_id):
    """Yields an ExperimentRun at the given run_id for update

    Will kick the last_update_time timestamp of the row each time.

    Args:
        db_engine (sqlalchemy.engine)
        run_id (int) The identifier/primary key of the run
    """
    return get_for_update(db_engine, ExperimentRun, run_id)


def experiment_entrypoint(entrypoint_func):
    """Decorator to control tracking of an experiment run at the wrapped method

    To update the database, it requires the instance of the wrapped method to have a
    db_engine and run_id.

    Upon method entry, will update the ExperimentRun row with the wrapped method name.
    Upon method exit, will update the ExperimentRun row with the status (either failed or completed)
    """
    @wraps(entrypoint_func)
    def with_entrypoint(self, *args, **kwargs):
        entrypoint_name = entrypoint_func.__name__
        with get_run_for_update(self.db_engine, self.run_id) as run_obj:
            if not run_obj.start_method:
                run_obj.start_method = entrypoint_name
        try:
            return_value = entrypoint_func(self, *args, **kwargs)
        except Exception as exc:
            with get_run_for_update(self.db_engine, self.run_id) as run_obj:
                run_obj.current_status = ExperimentRunStatus.failed
                run_obj.stacktrace = str(exc)
            raise exc

        with get_run_for_update(self.db_engine, self.run_id) as run_obj:
            run_obj.current_status = ExperimentRunStatus.completed

        return return_value

    return with_entrypoint


def increment_field(field, run_id, db_engine):
    """Increment an ExperimentRun's named field.

    Expects that the field is an integer in the database.

    Will also kick the last_updated_time timestamp.

    Args:
        field (str) The name of the field
        run_id (int) The identifier/primary key of the run
        db_engine (sqlalchemy.engine)
    """
    with scoped_session(db_engine) as session:
        # Use an update query instead of a session merge so it happens in one atomic query
        # and protect against race conditions
        session.query(ExperimentRun).filter_by(run_id=run_id).update({
            field: getattr(ExperimentRun, field) + 1,
            'last_updated_time': datetime.datetime.now()
        })


def record_matrix_building_started(run_id, db_engine):
    """Mark the current timestamp as the time at which matrix building started

    Args:
        run_id (int) The identifier/primary key of the run
        db_engine (sqlalchemy.engine)
    """
    with get_run_for_update(db_engine, run_id) as run_obj:
        run_obj.matrix_building_started = datetime.datetime.now()


def record_model_building_started(run_id, db_engine):
    """Mark the current timestamp as the time at which model building started

    Args:
        run_id (int) The identifier/primary key of the run
        db_engine (sqlalchemy.engine)
    """
    with get_run_for_update(db_engine, run_id) as run_obj:
        run_obj.model_building_started = datetime.datetime.now()


def built_matrix(run_id, db_engine):
    """Increment the matrix build counter for the ExperimentRun

    Args:
        run_id (int) The identifier/primary key of the run
        db_engine (sqlalchemy.engine)
    """
    increment_field('matrices_made', run_id, db_engine)


def skipped_matrix(run_id, db_engine):
    """Increment the matrix skip counter for the ExperimentRun

    Args:
        run_id (int) The identifier/primary key of the run
        db_engine (sqlalchemy.engine)
    """
    increment_field('matrices_skipped', run_id, db_engine)


def errored_matrix(run_id, db_engine):
    """Increment the matrix error counter for the ExperimentRun

    Args:
        run_id (int) The identifier/primary key of the run
        db_engine (sqlalchemy.engine)
    """
    increment_field('matrices_errored', run_id, db_engine)


def built_model(run_id, db_engine):
    """Increment the model build counter for the ExperimentRun

    Args:
        run_id (int) The identifier/primary key of the run
        db_engine (sqlalchemy.engine)
    """
    increment_field('models_made', run_id, db_engine)


def skipped_model(run_id, db_engine):
    """Increment the model skip counter for the ExperimentRun

    Args:
        run_id (int) The identifier/primary key of the run
        db_engine (sqlalchemy.engine)
    """
    increment_field('models_skipped', run_id, db_engine)


def errored_model(run_id, db_engine):
    """Increment the model error counter for the ExperimentRun

    Args:
        run_id (int) The identifier/primary key of the run
        db_engine (sqlalchemy.engine)
    """
    increment_field('models_errored', run_id, db_engine)

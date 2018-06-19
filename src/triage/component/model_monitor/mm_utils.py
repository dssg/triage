import os
import inspect
import logging
import datetime

DATE_FORMAT = "%Y-%m-%d"


def get_logger(log_level='DEBUG'):
    # set up logging
    logger = logging.getLogger('model_monitor')
    logger.setLevel(getattr(logging, log_level))

    today_dt = datetime.datetime.today().strftime(DATE_FORMAT)
    log_fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "logs",
                             "model-monitor-{}.log".format(today_dt))

    log_fhandler = logging.FileHandler(log_fpath)
    log_fhandler.setLevel(getattr(logging, log_level))

    log_shandler = logging.StreamHandler()
    log_shandler.setLevel(getattr(logging, log_level))

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_fhandler.setFormatter(log_formatter)
    log_shandler.setFormatter(log_formatter)

    logger.addHandler(log_fhandler)
    logger.addHandler(log_shandler)

    logger.debug("Log initialized at {}.".format(datetime.datetime.now()))
    return logger


def get_default_args(obj):
    signature = inspect.signature(obj)
    return {k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty}

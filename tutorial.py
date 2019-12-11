import argparse
from pathlib import Path

from argcmdr import local, LocalRoot, Local
from plumbum import local as plumlocal

ROOT_PATH = Path(__file__).parent.resolve()

class Tutorial(LocalRoot):
    """Commands to manage Dirtyducks infrastructure"""
    pass


@Tutorial.register
class Infrastructure(Local):

    def status(context):
        """shows the infrastructure's status"""
        return context.local['docker-compose']['ps']

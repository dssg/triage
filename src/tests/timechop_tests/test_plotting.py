from unittest.mock import patch
from unittest import TestCase
import yaml
import matplotlib

matplotlib.use("Agg")
from triage.component.timechop import Timechop # noqa
from triage.component.timechop.plotting import visualize_chops # noqa


class VisualizeChopTest(TestCase):
    @property
    def chopper(self):
        # create a valid Timechop chopper
        # least brittle current way of doing this is by loading the
        # example_experiment_config.yaml file, because that is a
        # diligently updated file. If Timechop config changes, the
        # example config should change too
        with open("example/config/experiment.yaml") as fd:
            experiment_config = yaml.full_load(fd)
        return Timechop(**(experiment_config["temporal_config"]))

    # hard to make many assertions, but we can make sure it gets to the end
    # and shows the contents.

    # we do one such test case to work out each combination of boolean arguments
    def test_default_args(self):
        with patch("triage.component.timechop.plotting.plt.show") as show_patch:
            visualize_chops(self.chopper)
            assert show_patch.called

    def test_no_as_of_times(self):
        with patch("triage.component.timechop.plotting.plt.show") as show_patch:
            visualize_chops(self.chopper, show_as_of_times=False)
            assert show_patch.called

    def test_no_boundaries(self):
        with patch("triage.component.timechop.plotting.plt.show") as show_patch:
            visualize_chops(self.chopper, show_boundaries=False)
            assert show_patch.called

    def test_no_boundaries_or_as_of_times(self):
        with patch("triage.component.timechop.plotting.plt.show") as show_patch:
            visualize_chops(self.chopper, show_as_of_times=False, show_boundaries=False)
            assert show_patch.called

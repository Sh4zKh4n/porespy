import os
import subprocess
import logging
import porespy as ps

rootdir = os.path.split(os.path.split(ps.__file__)[0])[0]
examples_dir = os.path.join(rootdir, 'examples')
filters_dir = os.path.join(examples_dir, 'filters')
metrics_dir = os.path.join(examples_dir, 'metrics')
netex_dir = os.path.join(examples_dir, 'networks')


class ExamplesTest():

    def setup_class(self):
        pass

    def _run_shell_command(self, command_line_args):
        try:
            proc = subprocess.run(command_line_args, timeout=600)
        except (OSError, subprocess.CalledProcessError) as exception:
            logging.info('Exception occured: ' + str(exception))
            logging.info('Subprocess failed')
            return False
        else:
            # no exception was raised
            logging.info('Subprocess finished')
        return proc.returncode == 0

    def _notebook_run(self, path):
        """Execute a notebook via nbconvert and collect output.
           :returns (parsed nb object, execution errors)
        """
        dirname, __ = os.path.split(path)
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=360",
                "--output", "temp_output.ipynb", path]
        rc = self._run_shell_command(args)
        print(path, rc)
        print('-'*30)
        if rc:
            os.remove(os.path.join(dirname, "temp_output.ipynb"))
        return rc

    def test_filters_porosimetry(self):
        nbook = os.path.join(filters_dir, 'porosimetry.ipynb')
        rc = self._notebook_run(nbook)
        assert rc

    def test_filters_snow_partitioning(self):
        nbook = os.path.join(filters_dir, 'snow_partitioning.ipynb')
        rc = self._notebook_run(nbook)
        assert rc

    def test_metrics_chord_length_distribution(self):
        nbook = os.path.join(metrics_dir, 'chord_length_distribution.ipynb')
        rc = self._notebook_run(nbook)
        assert rc

    def test_metrics_porosity_profiles(self):
        nbook = os.path.join(metrics_dir, 'porosity_profiles.ipynb')
        rc = self._notebook_run(nbook)
        assert rc

    def test_metrics_regionprops_3d(self):
        nbook = os.path.join(metrics_dir, 'regionprops_3d.ipynb')
        rc = self._notebook_run(nbook)
        assert rc

    def test_metrics_two_point_correlation(self):
        nbook = os.path.join(metrics_dir, 'two_point_correlation.ipynb')
        rc = self._notebook_run(nbook)
        assert rc

    def test_network_extraction_snow_basic(self):
        nbook = os.path.join(netex_dir, 'snow_basic.ipynb')
        rc = self._notebook_run(nbook)
        assert rc

    def test_network_extraction_snow_advanced(self):
        nbook = os.path.join(netex_dir, 'snow_advanced.ipynb')
        rc = self._notebook_run(nbook)
        assert rc


if __name__ == '__main__':
    t = ExamplesTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()

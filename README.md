# OSCILLATIONS REPOSITORY

This repository contains all codes used in the paper "An Oscillating Reaction Network with an exact closed form solution in the time domain". The plots for the paper can be generated using the script``scripts/mk_plots``. The underlying data for these plots can be generated using ``scripts/mk_data``. See "Using the repository" for the setup to run these scripts.

# Using the repository
The following instructions work for MACOS and LINUX. Windows requires a slight variation on these instructions as described [here](https://docs.python.org/3/library/venv.html).

The packages used are in ``requirements.txt``. To create a virtual environment ``osc`` with the required packages, navigate to the top level folder of the repository (which should be ``Oscillations`` unless you have changed it) and do the following:

* ``python3 -m venv osc``
* ``source osc/bin/activate``
* ``pip install --upgrade pip``
* ``pip install -r requirements.txt``
* ``deactivate``

To run codes, navigate the top level folder and:

* ``source osc/bin/activate``
* run a python script
* ``deactivate``


# Key python modules
The following modules are in ``src/Oscillations``.

* ``solver.py`` implements a symbolic solution to the system of differential equations. The step of the solution that calculates the particular solution is computationally intensive and so it is only performed if the flag ``IS_CHECK`` is set to ``True``.

* ``designer.py`` implements the ``designOscillator`` algorithm.

* ``evaluator.py`` calculates the metrics for design errors: feasibility design error, amplitude design error, and phase design error.

# Developer notes

* Unit tests are in ``tests``

* ``github actions`` are used to provide continuous integration
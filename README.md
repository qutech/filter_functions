# `filter_functions`: A package for efficient numerical calculation of generalized filter functions
[![Coverage Status](https://coveralls.io/repos/github/qutech/filter_functions/badge.svg?branch=master)](https://coveralls.io/github/qutech/filter_functions?branch=master)
[![Build Status](https://travis-ci.org/qutech/filter_functions.svg?branch=master)](https://travis-ci.org/qutech/filter_functions)
[![Documentation Status](https://readthedocs.org/projects/filter-functions/badge/?version=latest)](https://filter-functions.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/filter-functions.svg)](https://badge.fury.io/py/filter-functions)

Simply put, filter functions characterize a pulse's susceptibility to noise at a given frequency and can thus be used to gain insight into the decoherence of the system. The formalism allows for efficient calculation of several quantities of interest such as average gate fidelity. Moreover, the filter function of a composite pulse can be easily derived from those of the constituent pulses, allowing for efficient assembly and characterization of pulse sequences.

Previously, filter functions have only been computed analytically for select pulses such as dynamical decoupling sequences [1], [2]. With this project we aim to provide a toolkit for calculating and inspecting filter functions for arbitrary pulses including pulses without analytic form such as one might get from numerical pulse optimization algorithms. 

The package is built to interface with [QuTiP](http://qutip.org/), a widely-used quantum toolbox for Python, and comes with extensive documentation and a test suite.

## Installation
To install the package from PyPI, run `pip install filter_functions`. It is recommended to install QuTiP before by following the [instructions on their website](http://qutip.org/docs/latest/installation.html) rather than installing it through `pip`. To install the package from source run `python setup.py develop` to install using symlinks or `python setup.py install` without.

To install the optional dependencies (`tqdm` and `requests` for a fancy progress bar), run `pip install -e .[fancy_progressbar]` from the root directory.

## Documentation
You can find the documentation on [Readthedocs](https://filter-functions.readthedocs.io/en/latest/). It is built from Jupyter notebooks that can also be run interactively and are located [here](doc/source/examples). The notebooks explain how to use the package and thus make sense to follow chronologically as a first step. Furthermore, there are also a few example scripts in the [examples](examples) folder.

The documentation including the example notebooks and an automatically generated API documentation can be built by running `make <format>` inside the *doc* directory where `<format>` is for example `html`.

Building the documentation requires the following additional dependencies: `nbsphinx`, `numpydoc`, `sphinx_rtd_theme`, `jupyter_client`, `ipython`, `ipykernel`, as well as `pandoc`. The last can be installed via conda (`conda install pandoc`) or downloaded from [Github](https://github.com/jgm/pandoc/releases/) and the rest automatically by running `pip install -e .[doc]`.

## References
[1]: Cywinski, L., Lutchyn, R. M., Nave, C. P., & Das Sarma, S. (2008). How to enhance dephasing time in superconducting qubits. Physical Review B - Condensed Matter and Materials Physics, 77(17), 1–11. [https://doi.org/10.1103/PhysRevB.77.174509](https://doi.org/10.1103/PhysRevB.77.174509)

[2]: Green, T. J., Sastrawan, J., Uys, H., & Biercuk, M. J. (2013). Arbitrary quantum control of qubits in the presence of universal noise. New Journal of Physics, 15(9), 095004. [https://doi.org/10.1088/1367-2630/15/9/095004](https://doi.org/10.1088/1367-2630/15/9/095004)

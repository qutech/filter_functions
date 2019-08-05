# `filter_functions`: A package for efficient numeric calculation of generalized filter functions

Simply put, filter functions characterize a pulse's susceptibility to noise at a given frequency and can thus be used to gain insight into the decoherence of the system. The formalism allows for efficient calculation of several quantities of interest such as average gate fidelity. Moreover, the filter function of a composite pulse can be easily derived from those of the constituent pulses, allowing for efficient assembly and characterization of pulse sequences.

Previously, filter functions have only been computed analytically for select pulses such as dynamical decoupling sequences [1], [2]. With this project we aim to provide a toolkit for calculating and inspecting filter functions for arbitrary pulses including pulses without analytic form such as one might get from numeric pulse optimization algorithms. 

The package is built to interface with [QuTiP](http://qutip.org/), a widely-used quantum toolbox for Python, and comes with extensive documentation and a test suite.

## Installation
Run `python setup.py develop` to install using symlinks or `python setup.py install` without. It is recommended to install QuTiP before by following the [instructions on their website](http://qutip.org/docs/latest/installation.html) rather than installing it through `pip`.

To install the optional dependencies (`tqdm` and `requests` for a fancy progress bar), run `pip install -e .[fancy_progressbar]`.

## Documentation
You can find example Jupyter notebooks [here](doc/source/examples) and example scripts [here](examples). The notebooks explain how to use the package and thus make sense to follow chronologically as a first step. The documentation including the example notebooks and an automatically generated API documentation can be built by running `make <format>` inside the *doc* directory where `<format>` is for example `html`.

Building the documentation requires the following additional dependencies: `nbsphinx`, `numpydoc`, `sphinx_rtd_theme`, `jupyter_client`, `ipython`, `ipykernel`, as well as `pandoc`. The last can be installed via conda (`conda install pandoc`) or downloaded from [Github](https://github.com/jgm/pandoc/releases/) and the rest automatically by running `pip install -e .[doc]`.

## References
[1]: Cywinski, L., Lutchyn, R. M., Nave, C. P., & Das Sarma, S. (2008). How to enhance dephasing time in superconducting qubits. Physical Review B - Condensed Matter and Materials Physics, 77(17), 1–11. [https://doi.org/10.1103/PhysRevB.77.174509](https://doi.org/10.1103/PhysRevB.77.174509)

[2]: Green, T. J., Sastrawan, J., Uys, H., & Biercuk, M. J. (2013). Arbitrary quantum control of qubits in the presence of universal noise. New Journal of Physics, 15(9), 095004. [https://doi.org/10.1088/1367-2630/15/9/095004](https://doi.org/10.1088/1367-2630/15/9/095004)

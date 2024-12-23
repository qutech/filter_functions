# `filter_functions`: A package for efficient numerical calculation of generalized filter functions to describe the effect of noise on quantum gate operations
[![codecov](https://codecov.io/gh/qutech/filter_functions/branch/master/graph/badge.svg)](https://codecov.io/gh/qutech/filter_functions)
[![Build status](https://github.com/qutech/filter_functions/actions/workflows/main.yml/badge.svg)](https://github.com/qutech/filter_functions/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/filter-functions/badge/?version=latest)](https://filter-functions.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/filter-functions.svg)](https://pypi.org/project/filter-functions/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4575000.svg)](https://doi.org/10.5281/zenodo.4575000)

## Introduction
Simply put, filter functions characterize a quantum system's susceptibility to noise at a given frequency during a control operation and can thus be used to gain insight into its decoherence. The formalism allows for efficient calculation of several quantities of interest such as average gate fidelity and even the entire quantum process up to a unitary rotation. Moreover, the filter function of a composite pulse can be easily derived from those of the constituent pulses, allowing for efficient assembly and characterization of pulse sequences.

Initially, filter functions have been introduced to model dynamical decoupling sequences [5, 6]. With this project we aim to provide a toolkit for calculating and inspecting filter functions for arbitrary pulses including pulses without analytic form such as one might get from numerical pulse optimization algorithms. These filter functions can be used to compute process descriptions, fidelities and other quantities of interest from arbitrary classical noise spectral densities. For the efficient and convenient treatment of gate sequences, concatenation rules that allow the filter function of a sequence to be computed from those of its constituents are implemented.

The `filter_functions` package is built to interface with [QuTiP](http://qutip.org/), a widely-used quantum toolbox for Python, as well as [qopt](https://github.com/qutech/qopt) and comes with extensive documentation and a test suite.

As a very brief introduction, consider a Hadamard gate implemented by a pi/2 Y-gate followed by a NOT-gate using simple square pulses. We can calculate and plot the dephasing filter function of the gate with the following code:

```python
import filter_functions as ff
import qutip as qt
from math import pi

H_c = [[qt.sigmax()/2,   [0, pi], 'X'],
       [qt.sigmay()/2, [pi/2, 0], 'Y']]     # control Hamiltonian
H_n = [[qt.sigmaz()/2,    [1, 1], 'Z']]     # constant coupling to z noise
dt = [1, 1]                                 # time steps

hadamard = ff.PulseSequence(H_c, H_n, dt)   # Central object representing a control pulse
omega = ff.util.get_sample_frequencies(hadamard)
F = hadamard.get_filter_function(omega)

from filter_functions import plotting
plotting.plot_filter_function(hadamard)     # Filter function cached from before
```

![Hadamard dephasing filter function](./doc/source/_static/hadamard.png)

An alternative way of obtaining the Hadamard `PulseSequence` is by concatenating the composing pulses:

```python
Y2 = ff.PulseSequence([[qt.sigmay()/2, [pi/2], 'Y']],
                      [[qt.sigmaz()/2,    [1], 'Z']],
                      [1])
X = ff.PulseSequence([[qt.sigmax()/2, [pi], 'X']],
                     [[qt.sigmaz()/2,  [1], 'Z']],
                     [1])

Y2.cache_filter_function(omega)
X.cache_filter_function(omega)

hadamard = Y2 @ X           # equivalent: ff.concatenate((Y2, X))
hadamard.is_cached('filter function')
# True  (filter function cached during concatenation)
```

To compute, for example, the infidelity of the gate in the presence of an arbitrary classical noise spectrum, we can simply call `infidelity()`:

```python
spectrum = 1e-2/omega
infidelity = ff.infidelity(hadamard, spectrum, omega)
# array([0.0025])  (one contribution per noise operator)
```

## Installation
To install the package from PyPI, run `pip install filter_functions`. If you require the optional features provided by QuTiP (visualizing Bloch sphere trajectories), it is recommended to install QuTiP before by following the [instructions on their website](http://qutip.org/docs/latest/installation.html) rather than installing it through `pip`. To install the package from source run `python setup.py develop` to install using symlinks or `python setup.py install` without.

To install dependencies of optional extras (`matplotlib` for plotting, `QuTiP` for Bloch sphere visualization), run `pip install -e .[extra]` where `extra` is one or more of `plotting`, `bloch_sphere_visualization` from the root directory. To install all dependencies, including those needed to build the documentation and run the tests, use the extra `all`.

## Documentation
You can find the documentation on [Readthedocs](https://filter-functions.readthedocs.io/en/latest/). It is built from Jupyter notebooks that can also be run interactively and are located [here](doc/source/examples). The notebooks explain how to use the package and thus make sense to follow chronologically as a first step. Furthermore, there are also a few example scripts in the [examples](examples) folder.

The documentation including the example notebooks and an automatically generated API documentation can be built by running `make <format>` inside the *doc* directory where `<format>` is for example `html`.

Interactively using the documentation requires `jupyter`, and building a static version additionally requires `nbsphinx`, `numpydoc`, `sphinx_rtd_theme`, as well as `pandoc`. The last can be installed via conda (`conda install pandoc`) or downloaded from [Github](https://github.com/jgm/pandoc/releases/) and the rest automatically by running `pip install -e .[doc]`.

## Citing
If this software has benefited your research, please consider citing:

### Formalism
[1]: T. Hangleiter, P. Cerfontaine, and H. Bluhm. Filter-function formalism and software package to compute quantum processes of gate sequences for classical non-Markovian noise. Phys. Rev. Res. **3**, 043047 (2021). [10.1103/PhysRevResearch.3.043047](https://doi.org/10.1103/PhysRevResearch.3.043047). [arXiv:2103.02403](https://arxiv.org/abs/2103.02403).

[2]: P. Cerfontaine, T. Hangleiter, and H. Bluhm. Filter Functions for Quantum Processes under Correlated Noise. Phys. Rev. Lett. **127**, 170403 (2021). [10.1103/PhysRevLett.127.170403](https://doi.org/10.1103/PhysRevLett.127.170403). [arXiv:2103.02385](https://arxiv.org/abs/2103.02385).

### Gradients
[3]: I. N. M. Le, J. D. Teske, T. Hangleiter, P. Cerfontaine, and Hendrik Bluhm. Analytic Filter Function Derivatives for Quantum Optimal Control. Phys. Rev. Applied **17**, 024006 (2022). [10.1103/PhysRevApplied.17.024006](https://doi.org/10.1103/PhysRevApplied.17.024006). [arXiv:2103.09126](https://arxiv.org/abs/2103.09126).

### Software
[4]: T. Hangleiter, I. N. M. Le, and J. D. Teske, "filter_functions: A package for efficient numerical calculation of generalized filter functions to describe the effect of noise on quantum gate operations," (2021). [10.5281/zenodo.4575000](http://doi.org/10.5281/zenodo.4575000).

## Additional References
[5]: L. Cywinski, R. M. Lutchyn, C. P. Nave, and S. Das Sarma. How to enhance dephasing time in superconducting qubits. Phys. Rev. B **77**, 174509 (2008). [10.1103/PhysRevB.77.174509](https://doi.org/10.1103/PhysRevB.77.174509).

[6]: T. J Green., J. Sastrawan, H. Uys, and M. J. Biercuk. Arbitrary quantum control of qubits in the presence of universal noise. *New J. Phys.* **15**, 095004 (2013). [10.1088/1367-2630/15/9/095004](https://doi.org/10.1088/1367-2630/15/9/095004).

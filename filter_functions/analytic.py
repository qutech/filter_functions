# -*- coding: utf-8 -*-
# =============================================================================
#     filter_functions
#     Copyright (C) 2019 Quantum Technology Group, RWTH Aachen University
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#     Contact email: tobias.hangleiter@rwth-aachen.de
# =============================================================================
r"""
This file provides functions for the analytical solutions to some of the
dynamical decoupling sequences. Note that the filter functions given
here differ by a factor of 1/omega**2 from those defined in this package
due to different conventions. See for example [Cyw08]_. Depending on the
definition of the noise Hamiltonian one might also get different
results. The functions here agree for

.. math::

    B_\alpha\equiv\sigma_z/2.

Functions
---------
:func:`FID`
    Free Induction Decay / Ramsey pulse
:func:`SE`
    Spin Echo
:func:`PDD`
    Periodic Dynamical Decoupling
:func:`CPMG`
    Carr-Purcell-Meiboom-Gill Sequence
:func:`CDD`
    Concatenated Dynamical Decoupling
:func:`UDD`
    Uhrig Dynamical Decoupling

References
----------
.. [Cyw08]
    Cywiński, Ł., Lutchyn, R. M., Nave, C. P., & Das Sarma, S. (2008).
    How to enhance dephasing time in superconducting qubits. Physical
    Review B - Condensed Matter and Materials Physics, 77(17), 1–11.
    https://doi.org/10.1103/PhysRevB.77.174509
"""
import numpy as np


def FID(z):
    return 2*np.sin(z/2)**2


def SE(z):
    return 8*np.sin(z/4)**4


def PDD(z, n):
    if n % 2 == 0:
        return 2*np.tan(z/(2*n + 2))**2*np.cos(z/2)**2
    else:
        return 2*np.tan(z/(2*n + 2))**2*np.sin(z/2)**2


def CPMG(z, n):
    if n % 2 == 0:
        return 8*np.sin(z/4/n)**4*np.sin(z/2)**2/np.cos(z/2/n)**2
    else:
        return 8*np.sin(z/4/n)**4*np.cos(z/2)**2/np.cos(z/2/n)**2


def CDD(z, g):
    return 2**(2*g + 1)*np.sin(z/2**(g + 1))**2 *\
        np.product([np.sin(z/2**(k + 1))**2 for k in range(1, g+1)], axis=0)


def UDD(z, n):
    return np.abs(np.sum([(-1)**k*np.exp(1j*z/2*np.cos(np.pi*k/(n + 1)))
                          for k in range(-n-1, n+1)], axis=0))**2/2

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
"""
This file provides methods for the analytical solutions to some of the
dynamical decoupling sequences. The additional factor of two compared to the
reference is due to a different convention. Also note that the filter functions
given here differ by a factor of d/omega**2 from those defined in this package,
again due to convention. See for example [Cyw08]_. The factor d is moved out of
the filter function into the infidelity integral here, whereas the factor of
omega**2 stems from a different definition of the Fourier transform.

Functions
---------
:meth:`FID`
    Free Induction Decay / Ramsey pulse
:meth:`SE`
    Spin Echo
:meth:`PDD`
    Periodic Dynamical Decoupling
:meth:`CPMG`
    Carr-Purcell-Meiboom-Gill Sequence
:meth:`CDD`
    Concatenated Dynamical Decoupling
:meth:`UDD`
    Uhrig Dynamical Decoupling

References
----------
.. [Cyw08]
    Cywiński, Ł., Lutchyn, R. M., Nave, C. P., & Das Sarma, S. (2008). How to
    enhance dephasing time in superconducting qubits. Physical Review B -
    Condensed Matter and Materials Physics, 77(17), 1–11.
    https://doi.org/10.1103/PhysRevB.77.174509
"""
import numpy as np


def FID(z):
    return 2*2*np.sin(z/2)**2


def SE(z):
    return 2*8*np.sin(z/4)**4


def PDD(z, n):
    if n % 2 == 0:
        return 2*2*np.tan(z/(2*n + 2))**2*np.cos(z/2)**2
    else:
        return 2*2*np.tan(z/(2*n + 2))**2*np.sin(z/2)**2


def CPMG(z, n):
    if n % 2 == 0:
        return 2*8*np.sin(z/4/n)**4*np.sin(z/2)**2/np.cos(z/2/n)**2
    else:
        return 2*8*np.sin(z/4/n)**4*np.cos(z/2)**2/np.cos(z/2/n)**2


def CDD(z, l):
    return 2*2**(2*l + 1)*np.sin(z/2**(l + 1))**2 *\
        np.product([np.sin(z/2**(k + 1))**2 for k in range(1, l+1)], axis=0)


def UDD(z, n):
    return 2*np.abs(np.sum([(-1)**k*np.exp(1j*z/2*np.cos(np.pi*k/(n + 1)))
                            for k in range(-n-1, n+1)], axis=0))**2/2

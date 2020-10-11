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
"""
This module provides some functions related to superoperators and
quantum maps.

Functions
---------
:func:`liouville_representation`
    Calculate the Liouville representation of a unitary with respect to
    a basis
:func:`liouville_to_choi`
    Convert from Liouville to Choi matrix representation.
:func:`liouville_is_CP`
    Check if superoperator in Liouville representation is completely
    positive.
:func:`liouville_is_cCP`
    Check if superoperator in Liouville representation is conditional
    CP.

"""
from typing import Optional, Tuple, Union

import numpy as np
from numpy import linalg as nla
from numpy import ndarray

from . import basis as _b


def liouville_representation(U: ndarray, basis: _b.Basis) -> ndarray:
    r"""
    Get the Liouville representaion of the unitary U with respect to the
    basis.

    Parameters
    ----------
    U: ndarray, shape (..., d, d)
        The unitary.
    basis: Basis, shape (d**2, d, d)
        The basis used for the representation, e.g. a Pauli basis.

    Returns
    -------
    R: ndarray, shape (..., d**2, d**2)
        The Liouville representation of U.

    Notes
    -----
    The Liouville representation of a unitary quantum operation
    :math:`\mathcal{U}:\rho\rightarrow U\rho U^\dagger` is given by

    .. math::

        \mathcal{U}_{ij} = \mathrm{tr}(C_i U C_j U^\dagger)

    with :math:`C_i` elements of the basis spanning
    :math:`\mathbb{C}^{d\times d}` with :math:`d` the dimension of the
    Hilbert space.
    """
    U = np.asanyarray(U)
    if basis.btype == 'GGM' and basis.d > 12:
        # Can do closed form expansion and overhead compensated
        path = ['einsum_path', (0, 1), (0, 1)]
        conjugated_basis = np.einsum('...ba,ibc,...cd->...iad', U.conj(), basis, U, optimize=path)
        # If the basis is hermitian, the result will be strictly real so we can
        # drop the imaginary part
        R = _b.ggm_expand(conjugated_basis).real
    else:
        path = ['einsum_path', (0, 1), (0, 1), (0, 1)]
        R = np.einsum('...ba,ibc,...cd,jda', U.conj(), basis, U, basis,
                      optimize=path).real

    return R


def liouville_to_choi(superoperator: ndarray, basis: _b.Basis) -> ndarray:
    r"""Convert from Liouville to Choi matrix representation.

    Parameters
    ----------
    superoperator: ndarray, shape (..., d**2, d**2)
        The Liouville representation of a superoperator.
    basis: Basis, shape (d**2, d, d)
        The operator basis defining the Liouville representation.

    Notes
    -----
    The Choi matrix is given by

    .. math::

        \mathrm{choi}(\mathcal{S})
            &= (\mathbb{I}\otimes\mathcal{S})
                (|\Omega\rangle\langle\Omega|) \\
            &= \sum_{ij} E_{ij}\otimes\mathcal{S}(E_{ij}) \\
            &= \sum_{ij}\mathcal{S}_{ij} C_j^T\otimes C_i

    where :math:`|\Omega\rangle` is a maximally entangled state,
    :math:`E_{ij} = |i\rangle\langle j|`, and :math:`C_i` are the basis
    elements that define the Liouville representation
    :math:`\mathcal{S}_{ij}` [Mer13]_.

    Returns
    -------
    choi: ndarray, shape (..., d**2, d**2)
        The Choi matrix representation of the superoperator.

    References
    ----------

    .. [Mer13]
        Merkel, S. T. et al. Self-consistent quantum process tomography.
        Physical Review A - Atomic, Molecular, and Optical Physics, 87,
        062119 (2013). https://doi.org/10.1103/PhysRevA.87.062119

    See Also
    --------
    liouville_representation: Calculate Liouville representation of a unitary.
    liouville_is_CP: Test if a superoperator is completely positive (CP).
    liouville_is_cCP: Test if a superoperator is conditional CP.
    """
    choi = np.einsum('...ij,jba,icd->...acbd', superoperator, basis, basis,
                     optimize=['einsum_path', (0, 1), (0, 1)]).reshape(superoperator.shape)
    return choi


def liouville_is_CP(
        superoperator: ndarray,
        basis: _b.Basis,
        return_eig: Optional[bool] = False,
        atol: Optional[float] = None
) -> Union[bool, Tuple[bool, Tuple[ndarray, ndarray]]]:
    r"""Test if a Liouville superoperator is completely positive (CP).

    Parameters
    ----------
    superoperator: ndarray, shape (..., d**2, d**2)
        The superoperator in Liouville representation to be checked for
        CPness.
    basis: Basis, shape (d**2, d, d)
        The operator basis defining the Liouville representation.
    return_eig: bool, optional
        Return the tuple of eigenvalues and eigenvectors of the Choi
        matrix. The default is False.
    atol: float, optional
        Absolute tolerance for the complete positivity.

    Returns
    -------
    CP: bool, (shape (...,))
        The (array, if broadcasted) of bools indicating if superoperator
        is CP.
    (D, V): Tuple[ndarray, ndarray]
        The eigenvalues and eigenvectors of the Choi matrix (only if
        return_eig is True).

    Notes
    -----
    A superoperator :math:`\mathcal{S}` is completely positive (CP) if
    and only if its Choi matrix representation is positive semidefinite:

    .. math::

        \mathcal{S}\text{ is CP }\Leftrightarrow
        \mathrm{choi}(\mathcal{S})\geq 0.

    See Also
    --------
    liouville_representation: Calculate Liouville representation of a unitary.
    Liouville_to_choi: Convert from Liouville to Choi matrix representation.
    liouville_is_cCP: Test if a superoperator is conditional CP.
    """

    choi = liouville_to_choi(superoperator, basis)
    D, V = nla.eigh(choi)

    CP = (D >= -(atol or basis._atol)).all(axis=-1)

    if return_eig:
        return CP, (D, V)

    return CP


def liouville_is_cCP(
        superoperator: ndarray,
        basis: _b.Basis,
        return_eig: Optional[bool] = False,
        atol: Optional[float] = None
) -> Union[bool, Tuple[bool, Tuple[ndarray, ndarray]]]:
    r"""Test if a Liouville superoperator is conditional completely positive.

    Parameters
    ----------
    superoperator: ndarray, shape (..., d**2, d**2)
        The superoperator in Liouville representation to be checked for
        cCPness
    basis: Basis, shape (d**2, d, d)
        The operator basis defining the Liouville representation.
    return_eig: bool, optional
        Return the tuple of eigenvalues and eigenvectors of the Choi
        matrix projected on the complement of the maximally entangled
        state. The default is False.
    atol: float, optional
        Absolute tolerance for the complete positivity.

    Returns
    -------
    cCP: bool, (shape (...,))
        The (array, if broadcasted) of bools indicating if superoperator
        is cCP
    (D, V): Tuple[ndarray, ndarray]
        The eigenvalues and eigenvectors of the projected Choi matrix
        (only if return_eig is True).

    Notes
    -----
    A superoperator :math:`\mathcal{S}` is conditional completely
    positive (cCP) if and only if its Choi matrix projected on the
    complement of the maximally entangled state is positive
    semidefinite:

    .. math::

        \mathcal{S}\text{ is cCP }\Leftrightarrow
        Q\mathrm{choi}(\mathcal{S})Q\geq 0

    with :math:`Q = \mathbb{I} - |\Omega\rangle\langle\Omega|`.

    See Also
    --------
    liouville_representation: Calculate Liouville representation of a unitary.
    Liouville_to_choi: Convert from Liouville to Choi matrix representation.
    liouville_is_CP: Test if a superoperator is CP.
    """
    d2 = superoperator.shape[-1]
    d = int(np.sqrt(d2))

    # Maximally entangled state
    Omega = np.zeros(d2, dtype=float)
    Omega[::d+1] = 1/np.sqrt(d)
    Omega = np.multiply.outer(Omega, Omega)

    # Projector onto complement of Omega
    Q = np.eye(Omega.shape[-1]) - Omega

    choi = liouville_to_choi(superoperator, basis)
    D, V = nla.eigh(Q @ choi @ Q)

    cCP = (D >= -(atol or basis._atol)).all(axis=-1)

    if return_eig:
        return cCP, (D, V)

    return cCP

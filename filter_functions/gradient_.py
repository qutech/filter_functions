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
#     This module is an extension of the filter_functions package written by
#     Tobias Hangleiter. Its implementation was center of the Bachelor thesis
#     'Filter Function Derivative for Quantum Optimal Control' by Isabel Le.
#
#     Contact email: isabel.le@rwth-aachen.de
#
#     The code has been extended by Julian Teske such that the noise
#     susceptibilities or noise coefficients can depend on the control
#     amplitudes as well.
#
#     Contact email: j.teske@fz-juelich.de
# =============================================================================
"""
This module implements functions to calculate filter function and
infidelity derivatives.

Throughout this documentation the following notation will be used:
    - n_dt denotes the number of time steps,
    - n_cops the number of all control operators,
    - n_ctrl the number of accessible control operators (if identifiers
      are provided, otherwise n_ctrl=n_cops),
    - n_nops the number of noise operators,
    - n_omega the number of frequency samples, and
    - d the dimension of the system.

Functions
---------
:func:`liouville_derivative`
    Calculate the derivatives of the control propagators in Liouville
    representation.
:func:`control_matrix_at_timestep_derivative`
    Calculate the control matrices and corresponding derivatives.
:func:`calculate_derivative_of_control_matrix_from_scratch`
    Calculate the derivative of the control matrix from scratch.
:func:`calculate_canonical_filter_function_derivative`
    Compute the filter function derivative from the control matrix.
:func:`infidelity_derivative`
    Calculate the infidelity derivative.
"""
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import opt_einsum as oe
from numpy import ndarray
from scipy.integrate import trapz

from . import numeric, superoperator, util
from .basis import Basis
from .types import Coefficients, Operator

__all__ = ['liouville_derivative', 'control_matrix_at_timestep_derivative',
           'calculate_derivative_of_control_matrix_from_scratch',
           'calculate_filter_function_derivative', 'infidelity_derivative']


def _derivative_integral_(E, eigvals, dt, deriv_integral):
    """Calculate I_nmpq, but return I_pqnm"""
    # Any combination of \omega_p-\omega_q (dR_g)
    dE = np.subtract.outer(eigvals, eigvals)
    mask_dE = np.abs(dE) < 1e-7
    # Any combination of \omega+\omega_n-\omega_m (dR_g)
    EdE = np.add.outer(E, dE)
    mask_EdE = np.abs(EdE) < 1e-7
    # Any combination of \omega+\omega_n-\omega_m+\omega_p-\omega_q (dR_g)
    EdEdE = np.add.outer(EdE, dE[~mask_dE])

    # calculation if omega_diff = 0
    tmp1 = np.divide(util.cexp(EdE*dt), EdE, where=~mask_EdE)
    tmp2 = tmp1 - np.divide(1, EdE, where=~mask_EdE)

    tmp1 *= -1j * dt
    tmp1 += np.divide(tmp2, EdE, where=~mask_EdE)
    tmp1[mask_EdE] = dt**2 / 2

    # tmp1 = util.cexp(EdE*dt) / EdE
    # tmp2 = tmp1 - 1 / EdE
    # tmp1 = -1j * dt * tmp1 + tmp2 / EdE

    deriv_integral[:, mask_dE] = tmp1[:, None]

    # # calculation if omega_diff != 0
    # tmp1 = np.divide(1 - util.cexp(EdEdE*dt), EdEdE, where=~mask_ndE_nEdE)
    # tmp1[mask_ndE_EdE] = -1j * dt
    # tmp1 += tmp2[:, None, None]
    # tmp1 = np.divide(tmp1, dE, where=~mask_dE, out=tmp1)
    # deriv_integral[:, ~mask_dE] = tmp1[:, ~mask_dE]
    # calculation if omega_diff != 0

    # pq on last axis in EdEdE, pq on 1,2nd axes in deriv_integral
    deriv_integral[:, ~mask_dE] = ((1 - util.cexp(EdEdE*dt)) / EdEdE / dE[~mask_dE]).transpose(0, 3, 1, 2)
    deriv_integral[:, ~mask_dE] += np.divide.outer(tmp2, dE[~mask_dE]).transpose(0, 3, 1, 2)
    return deriv_integral


def _derivative_integral(E, eigvals, dt, out):
    dE = np.subtract.outer(eigvals, eigvals)
    mask_dE = np.abs(dE) < 1e-7
    EdE = np.add.outer(E, dE)
    mask_EdE = np.abs(EdE) < 1e-7
    EdEdE = np.add.outer(EdE, dE[~mask_dE])
    mask_EdEdE = np.abs(EdEdE) < 1e-7

    # Omega_pq == 0
    tmp1 = np.divide(util.cexp(EdE*dt), EdE, where=~mask_EdE)
    tmp2 = tmp1 - np.divide(1, EdE, where=~mask_EdE)
    tmp2[mask_EdE] = 1j * dt

    tmp1 *= -1j * dt
    tmp1 += np.divide(tmp2, EdE, where=~mask_EdE)
    tmp1[mask_EdE] = dt**2 / 2

    out[:, mask_dE] = tmp1[:, None]

    # Omega_pq != 0
    tmp1 = np.divide(1 - util.cexp(EdEdE*dt), EdEdE, where=~mask_EdEdE)
    tmp1[mask_EdEdE] = -1j * dt
    tmp1 += tmp2[..., None]

    out[:, ~mask_dE] = (tmp1 / dE[~mask_dE]).transpose(0, 3, 1, 2)

    return out


def liouville_derivative(
        dt: Coefficients,
        propagators: ndarray,
        basis: Basis,
        eigvecs: ndarray,
        eigvals: ndarray,
        transf_control_operators: ndarray) -> ndarray:
    r"""
    Calculate the derivatives of the control propagators in Liouville
    representation.

    Parameters
    ----------
    dt : array_like, shape (n_dt)
        Sequence duration, i.e. for the :math:`g`-th pulse
        :math:`t_g - t_{g-1}`.
    propagators : array_like, shape (n_dt+1, d, d)
        The propagators :math:`Q_g = P_g P_{g-1}\cdots P_0` as a (d, d) array
        with *d* the dimension of the Hilbert space.
    basis : Basis, shape (d**2, d, d)
        The basis elements, in which the pulse control matrix will be expanded.
    eigvecs : array_like, shape (n_dt, d, d)
        Eigenvector matrices for each time pulse segment *g* with the first
        axis counting the pulse segment, i.e.
        ``HV == array([V_0, V_1, ...])``.
    eigvals : array_like, shape (n_dt, d)
        Eigenvalue vectors for each time pulse segment *g* with the first
        axis counting the pulse segment, i.e.
        ``HD == array([D_0, D_1, ...])``.
    transf_control_operators : array_like, shape (n_dt, c_ctrl, d, d)
        The control operators transformed into the eigenspace of the control
        Hamiltonian. The drift operators are ignored, if identifiers for
        accessible control operators are provided.

    Returns
    -------
    dL: array_like, shape (n_dt, n_ctrl, n_dt, d**2, d**2)
        The derivative of the control propagators in Liouville representation
        :math:`\frac{\partial \mathcal{Q}_{jk}^{(g)}}
        {\partial u_h(t_{g^\prime})}`.
        The array's indexing has shape :math:`(g,h,g^\prime,j,k)`.

    Notes
    -----
    The derivatives of the control propagators in Liouville representation are
    calculated according to

    .. math::

        \frac{\partial\mathcal{Q}_{jk}^{(g-1)}}{\partial u_h(t_{g^\prime})} &=
        \Theta_{g-1}(g^\prime) \mathrm{tr}\Big(
            \frac{\partial U_c^\dagger(t_{g-1},0)}{\partial u_h(t_{g^\prime})}
            C_j U_c(t_{g-1},0) C_k\Big)\\
            &+\Theta_{g-1}(g^\prime)\mathrm{tr}\Big(U_c^\dagger(t_{g-1},0)C_j
                                                \frac{\partial U_c(t_{g-1},0)}
                                                {\partial u_h(t_{g^\prime})}C_k
                                                \Big),

    where :math:`\Theta_{g-1}(g^\prime)` being 1 if :math:`g^\prime < g-1` and
    zero otherwise.

    """
    n = len(dt)
    d = basis.shape[-1]

    # Allocate memory
    A_mat = np.empty((d, d), dtype=complex)
    partial_U = np.empty((n, transf_control_operators.shape[1], d, d),
                         dtype=complex)
    deriv_U = np.zeros((n, n, transf_control_operators.shape[1], d, d),
                       dtype=complex)

    for g in (range(n)):
        omega_diff = np.subtract.outer(eigvals[g], eigvals[g])
        mask = (abs(omega_diff) < 1e-10)
        A_mat[mask] = dt[g]  # if the integral diverges
        A_mat[~mask] = (util.cexp(omega_diff[~mask]*dt[g]) - 1) \
            / (1j * omega_diff[~mask])
        # Calculate dU(t_g,t_{g-1})/du_h within one time step
        partial_U[g] = -1j * np.einsum(
            'ia,ja,jk,hkl,kl,ml->him', propagators[g + 1],
            propagators[g].conj(), eigvecs[g], transf_control_operators[g],
            A_mat, eigvecs[g].conj(), optimize=['einsum_path', (3, 4), (0, 1),
                                                (0, 3), (0, 1), (0, 1)])
        # Calculate the whole propagator derivative
        for g_prime in range(g+1):  # condition g' <= g-1
            deriv_U[g, g_prime] = np.einsum(
                'ij,kj,hkl,lm->him', propagators[g + 1],
                propagators[g_prime + 1].conj(), partial_U[g_prime],
                propagators[g_prime], optimize=['einsum_path', (0, 1),
                                                (0, 1), (0, 1)])

    # Now calculate derivative of Liouville representation
    # Calculate traces first to save memory
    sum1 = np.einsum(
        'tshba,jbc,tcd,kda->thsjk', deriv_U.conj(), basis, propagators[1:],
         basis, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    sum2 = np.einsum(
        'tba,jbc,tshcd,kda->thsjk', propagators[1:].conj(), basis, deriv_U,
        basis, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    liouville_deriv = sum1 + sum2

    return liouville_deriv


def liouville_derivative_faster(
        dt: Coefficients,
        propagators: ndarray,
        basis: Basis,
        eigvecs: ndarray,
        eigvals: ndarray,
        transf_control_operators: ndarray) -> ndarray:
    """."""
    n = len(dt)
    d = basis.shape[-1]

    # Allocate memory
    A_mat = np.empty((d, d), dtype=complex)
    omega_diff = np.empty((d, d), dtype=float)
    partial_U = np.empty((n, transf_control_operators.shape[1], d, d),
                         dtype=complex)
    deriv_U = np.zeros((n, n, transf_control_operators.shape[1], d, d),
                       dtype=complex)

    for g in (range(n)):
        omega_diff = np.subtract.outer(eigvals[g], eigvals[g], out=omega_diff)
        mask = omega_diff == 0
        A_mat[mask] = dt[g]  # if the integral diverges
        A_mat[~mask] = (util.cexp(omega_diff[~mask]*dt[g]) - 1) / (1j * omega_diff[~mask])
        # Calculate dU(t_g,t_{g-1})/du_h within one time step
        partial_U[g] = -1j * (propagators[g+1] @ propagators[g].conj().T @ eigvecs[g] @
                              (transf_control_operators[g] * A_mat) @ eigvecs[g].conj().T)
        # # Calculate the whole propagator derivative
        deriv_U[g, :g+1] = (propagators[g+1] @ propagators[1:g+2].conj().swapaxes(-1, -2) @
                             partial_U[:g+1].swapaxes(0, 1) @ propagators[:g+1]).swapaxes(0, 1)

    # Now calculate derivative of Liouville representation
    # Calculate traces first to save memory
    liouville_deriv = np.einsum(
        'tshba,jbc,tcd,kda->thsjk', deriv_U.conj(), basis, propagators[1:],
         basis, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])

    return 2*liouville_deriv.real


def liouville_derivative_vectorized(
        dt: Coefficients,
        propagators: ndarray,
        basis: Basis,
        eigvecs: ndarray,
        eigvals: ndarray,
        VHV: ndarray
) -> ndarray:
    """"."""
    n, d = eigvecs.shape[:2]
    # omega_i - omega_j
    omega_diff = eigvals[:, :, None] - eigvals[:, None]
    dt_broadcast = np.broadcast_to(dt[:, None, None], omega_diff.shape)
    # mask = omega_diff == 0
    mask = np.broadcast_to(np.eye(d, dtype=bool), omega_diff.shape)
    A_mat = np.empty(omega_diff.shape, dtype=complex)
    A_mat[mask] = dt_broadcast[mask]
    # A_mat[~mask] = (util.cexp((omega_diff[~mask]*dt_broadcast[~mask])) - 1) / (1j * omega_diff[~mask])
    A_mat[~mask] = np.expm1(1j*(omega_diff[~mask]*dt_broadcast[~mask])) / (1j * omega_diff[~mask])
    U_deriv = -1j * (propagators[1:]
                     @ propagators[:-1].conj().swapaxes(-1, -2)
                     @ eigvecs
                     @ (A_mat * VHV.swapaxes(0, 1))
                     @ eigvecs.conj().swapaxes(-1, -2))

    # Calculate the whole propagator derivative
    propagators_deriv = np.zeros((VHV.shape[1], n-1, n, d, d), dtype=complex)
    for g in range(n-1):
        propagators_deriv[:, g, :g+1] = (propagators[g+1]
                                         @ propagators[1:g+2].conj().swapaxes(-1, -2)
                                         @ U_deriv[:, :g+1]
                                         @ propagators[:g+1])

    # liouville_deriv = np.einsum('htsba,tjkba->thsjk', propagators_deriv.conj(),
    #                             (basis @ propagators[1:-1, None])[:, :, None] @ basis).real
    liouville_deriv = oe.contract('htsba,jbc,tcd,kda->thsjk',
                                  propagators_deriv.conj(), basis, propagators[1:-1], basis,
                                  optimize=[(0, 2), (0, 2), (0, 1)]).real
    liouville_deriv *= 2

    return liouville_deriv


def liouville_derivative_vectorized_completely(
        dt: Coefficients,
        propagators: ndarray,
        basis: Basis,
        eigvecs: ndarray,
        eigvals: ndarray,
        VHV: ndarray
) -> ndarray:
    n, d = eigvecs.shape[:2]
    # omega_i - omega_j
    omega_diff = eigvals[:, :, None] - eigvals[:, None]
    dt_broadcast = np.broadcast_to(dt[:, None, None], omega_diff.shape)
    # mask = omega_diff == 0
    mask = np.broadcast_to(np.eye(d, dtype=bool), omega_diff.shape)
    A_mat = np.empty(omega_diff.shape, dtype=complex)
    A_mat[mask] = dt_broadcast[mask]
    A_mat[~mask] = np.expm1(1j*(omega_diff[~mask]*dt_broadcast[~mask])) / (1j * omega_diff[~mask])
    U_deriv = -1j * (propagators[1:]
                     @ propagators[:-1].conj().swapaxes(-1, -2)
                     @ eigvecs
                     @ (A_mat * VHV.swapaxes(0, 1))
                     @ eigvecs.conj().swapaxes(-1, -2))

    # opt_einsum performs some magic with intermediate terms here, faster than explicit loop
    propagators_deriv = oe.contract('tab,scb,hscd,sde->htsae', propagators[1:-1],
                                    propagators[1:].conj(), U_deriv, propagators[:-1],
                                    optimize=[(1, 2), (1, 2), (0, 1)])
    # Derivative is zero for times in the future of the control step
    propagators_deriv[:, ~np.tri(n - 1, n, dtype=bool)] = 0

    liouville_deriv = oe.contract('htsba,jbc,tcd,kda->thsjk',
                                  propagators_deriv.conj(), basis, propagators[1:-1], basis,
                                  optimize=[(0, 2), (0, 2), (0, 1)]).real
    # oe.contract('xtba,jbc,tcd,kda->xtjk', propagators_deriv.transpose(0,2,1,3,4).reshape(5*100,99,4,4, order='F').conj(), basis, propagators[1:-1], basis, optimize=p).real.reshape(5,100,99,16,16,order='F')
    liouville_deriv *= 2

    return liouville_deriv


def liouville_derivative_vectorized_loop(
        dt: Coefficients,
        propagators: ndarray,
        basis: Basis,
        eigvecs: ndarray,
        eigvals: ndarray,
        c_opers_transformed: ndarray
) -> ndarray:
    """"."""
    n, d = eigvecs.shape[:2]
    # omega_i - omega_j
    omega_diff = eigvals[:, :, None] - eigvals[:, None]
    dt_broadcast = np.broadcast_to(dt[:, None, None], omega_diff.shape)
    # mask = omega_diff == 0
    mask = np.broadcast_to(np.eye(d, dtype=bool), omega_diff.shape)
    integral = np.empty(omega_diff.shape, dtype=complex)
    integral[mask] = dt_broadcast[mask]
    # integral[~mask] = (util.cexp((omega_diff[~mask]*dt_broadcast[~mask])) - 1) / (1j * omega_diff[~mask])
    integral[~mask] = (np.expm1(1j*(omega_diff[~mask]*dt_broadcast[~mask]))
                       / (1j * omega_diff[~mask]))

    return -1j * (propagators[1:]
                  @ propagators[:-1].conj().swapaxes(-1, -2)
                  @ eigvecs
                  @ (integral * c_opers_transformed.swapaxes(0, 1))
                  @ eigvecs.conj().swapaxes(-1, -2))


def liouville_derivative_vectorized_completely_loop(propagator_deriv, basis, propagator, expr,
                                                    out):
    liouville_deriv = expr(propagator_deriv.conj(), basis, propagator, basis, out=out)
    liouville_deriv.real *= 2
    return liouville_deriv


def propagators_derivative_vectorized_completely(
        dt: Coefficients,
        propagators: ndarray,
        basis: Basis,
        eigvecs: ndarray,
        eigvals: ndarray,
        c_opers_transformed: ndarray
) -> ndarray:

    n, d = eigvecs.shape[:2]
    # omega_i - omega_j
    omega_diff = eigvals[:, :, None] - eigvals[:, None]
    dt_broadcast = np.broadcast_to(dt[:, None, None], omega_diff.shape)
    # mask = omega_diff == 0
    mask = np.broadcast_to(np.eye(d, dtype=bool), omega_diff.shape)
    A_mat = np.empty(omega_diff.shape, dtype=complex)
    A_mat[mask] = dt_broadcast[mask]
    A_mat[~mask] = np.expm1(1j*(omega_diff[~mask]*dt_broadcast[~mask])) / (1j * omega_diff[~mask])
    U_deriv = -1j * (propagators[1:]
                     @ propagators[:-1].conj().swapaxes(-1, -2)
                     @ eigvecs
                     @ (A_mat * c_opers_transformed.swapaxes(0, 1))
                     @ eigvecs.conj().swapaxes(-1, -2))

    # opt_einsum performs some magic with intermediate terms here, faster than explicit loop
    propagators_deriv = oe.contract('tab,scb,hscd,sde->htsae', propagators[1:-1],
                                    propagators[1:].conj(), U_deriv, propagators[:-1],
                                    optimize=[(1, 2), (1, 2), (0, 1)])
    # Derivative is zero for times in the future of the control step
    propagators_deriv[:, ~np.tri(n - 1, n, dtype=bool)] = 0
    return propagators_deriv


def propagator_derivative(
        dt: Coefficients,
        propagators: ndarray,
        eigvecs: ndarray,
        eigvals: ndarray,
        transf_control_operators: ndarray) -> ndarray:
    """"."""
    n, d = eigvecs.shape[:2]
    # omega_i - omega_j
    omega_diff = eigvals[:, :, None] - eigvals[:, None]
    dt_broadcast = np.broadcast_to(dt[:, None, None], omega_diff.shape)
    # mask = omega_diff == 0
    mask = np.broadcast_to(np.eye(d, dtype=bool), omega_diff.shape)
    A_mat = np.empty(omega_diff.shape, dtype=complex)
    A_mat[mask] = dt_broadcast[mask]
    # A_mat[~mask] = (util.cexp((omega_diff[~mask]*dt_broadcast[~mask])) - 1) / (1j * omega_diff[~mask])
    A_mat[~mask] = np.expm1(1j*(omega_diff[~mask]*dt_broadcast[~mask])) / (1j * omega_diff[~mask])
    partial_U = -1j * (propagators[1:] @ propagators[:-1].conj().swapaxes(-1, -2)
                       @ eigvecs
                       @ (transf_control_operators.swapaxes(0, 1) * A_mat)
                       @ eigvecs.conj().swapaxes(-1, -2))

    # Calculate the whole propagator derivative
    derivative = np.zeros((transf_control_operators.shape[1], n, n, d, d), dtype=complex)
    for g in range(n):
        derivative[:, g, :g+1] = (propagators[g+1] @ propagators[1:g+2].conj().swapaxes(-1, -2) @
                                  partial_U[:, :g+1] @ propagators[:g+1])

    return derivative


def propagator_derivative_factor(
        dt: Coefficients,
        propagators: ndarray,
        eigvecs: ndarray,
        eigvals: ndarray,
        VHV: ndarray) -> ndarray:
    """"."""
    n, d = eigvecs.shape[:2]
    # omega_i - omega_j
    omega_diff = eigvals[:, :, None] - eigvals[:, None]
    dt_broadcast = np.broadcast_to(dt[:, None, None], omega_diff.shape)
    # mask = omega_diff == 0
    mask = np.broadcast_to(np.eye(d, dtype=bool), omega_diff.shape)
    A_mat = np.empty(omega_diff.shape, dtype=complex)
    A_mat[mask] = dt_broadcast[mask]
    # A_mat[~mask] = (util.cexp((omega_diff[~mask]*dt_broadcast[~mask])) - 1) / (1j * omega_diff[~mask])
    A_mat[~mask] = np.expm1(1j*(omega_diff[~mask]*dt_broadcast[~mask])) / (1j * omega_diff[~mask])
    partial_U = -1j * (propagators[1:] @ propagators[:-1].conj().swapaxes(-1, -2)
                       @ eigvecs
                       @ (VHV.swapaxes(0, 1) * A_mat)
                       @ eigvecs.conj().swapaxes(-1, -2))

    # Calculate the whole propagator derivative
    derivative = np.zeros((VHV.shape[1], n, n, d, d), dtype=complex)
    for g in range(n):
        derivative[:, g, :g+1] = (propagators[1:g+2].conj().swapaxes(-1, -2)
                                  @ partial_U[:, :g+1]
                                  @ propagators[:g+1])

    return derivative


def control_matrix_at_timestep_derivative(
        omega: Coefficients,
        dt: Coefficients,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        transf_control_operators: ndarray,
        s_derivs: Optional[Sequence[Coefficients]] = None) -> dict:
    r"""
    Calculate the control matrices and corresponding derivatives.

    Calculate control matrices at each time step and the corresponding partial
    derivatives of those with respect to control strength at each time step.

    Parameters
    ----------
    omega : array_like, shape (n_omega)
        Frequencies, at which the pulse control matrix is to be evaluated.
    dt : array_like, shape (n_dt)
        Sequence duration, i.e. for the :math:`g`-th pulse
        :math:`t_g - t_{g-1}`.
    eigvals : array_like, shape (n_dt, d)
        Eigenvalue vectors for each time pulse segment *g* with the first
        axis counting the pulse segment, i.e.
        ``HD == array([D_0, D_1, ...])``.
    eigvecs : array_like, shape (n_dt, d, d)
        Eigenvector matrices for each time pulse segment *g* with the first
        axis counting the pulse segment, i.e.
        ``HV == array([V_0, V_1, ...])``.
    basis : Basis, shape (d**2, d, d)
        The basis elements, in which the pulse control matrix will be expanded.
    n_opers : array_like, shape (n_nops, d, d)
        Noise operators :math:`B_\alpha`.
    n_coeffs : array_like, shape (n_nops, n_dt)
        The sensitivities of the system to the noise operators given by
        *n_opers* at the given time step.
    transf_control_operators : array_like, shape (n_dt, n_ctrl, d, d)
        The control operators transformed into the eigenspace of the control
        Hamiltonian. The drift operators are ignored, if identifiers for
        accessible control operators are provided.
    s_derivs : array_like, shape (n_nops, n_ctrl, n_dt)
        The derivatives of the noise susceptibilities by the control amplitudes.
        Defaults to None.

    Returns
    -------
    ctrlmat_data : dict {'R_g': R_g, 'dR_g': gradient}
        * **R_g** *(array_like, shape (n_dt, n_nops, d**2, n_omega))*
        The control matrix at each time step
        :math:`\mathcal{R}_{\alpha j}^{(g)}(\omega)` is identified with R_g.
        The array's indexing has shape :math:`(g,\alpha,j,\omega)`.

        * **dR_g** *(array_like, shape (n_dt, n_nops, d**2, n_ctrl, n_omega))*
        The corresponding derivative with respect to the control strength
        :math:`\frac{\partial\mathcal{R}_{\alpha j}^{(g)}(\omega)}
        {\partial u_h(t_{g^\prime})}` is identified with dR_g. The array's
        indexing has shape :math:`(g^\prime,\alpha,j,h,\omega)`. Here, only one
        time axis is needed, since the derivative is zero for
        :math:`g\neq g^\prime`.


    Notes
    -----
    The control matrix at each time step is evaluated according to

    .. math::

            \mathcal{R}_{\alpha j}^{(g)}(\omega) = s_\alpha^{(g)}\mathrm{tr}
            \left([\bar{B}_\alpha \circ I_1^{(g)}(\omega)] \bar{C}_j \right),

    where

    .. math::

        I_{1,nm}^{(g)}(\omega) = \frac{\exp(\mathrm{i}(\omega + \omega_n^{(g)}
                                            - \omega_m^{(g)}) \Delta t_g) - 1}
        {\mathrm{i}(\omega + \omega_n^{(g)} - \omega_m^{(g)})}

    The derivative of the control matrix with respect to the control strength
    at different time steps is calculated according to

    .. math::

        \frac{\partial \mathcal{R}_{\alpha j}^{(g)}(\omega)}
        {\partial u_h(t_{g^\prime})} =
        \mathrm{i}\delta_{gg^\prime} s_\alpha^{(g)} \mathrm{tr}
        \left( \bar{B}_{\alpha} \cdot \mathbb{M} \right)
        + \frac{\partial s_\alpha^{(g)}}{u_h (t_{g^\prime})} \text{tr}
        \left( (\overline{B}_{\alpha} \circ I_1^{(g)}{}(\omega)) \cdot
        \overline{C}_{j}) \right).

    We assume that the noise susceptibility :math:`s` only depends locally
    on the time i.e. :math:`\partial_{u(t_g)} s(t_{g^\prime})
    = \delta_{gg^\prime} \partial_{u(t_g)} s(t_g)`
    If denoting :math:`\Delta\omega_{ij} = \omega_i^{(g)} - \omega_j^{(g)}`
    the integral part is encapsulated in

    .. math::

        \mathbb{M}_{mn} = \left[ \bar{C}_j, \mathbb{I}^{(mn)}
                                \circ \bar{H}_h \right]_{mn},

    with

    .. math::

        \mathbb{I}_{pq}^{(mn)} &= \delta_{pq} \left(
            \frac{\Delta t_g \cdot
                  \exp(\mathrm{i}(\omega + \Delta\omega_{nm})\Delta t_g)}
            {\mathrm{i}(\omega + \Delta\omega_{nm})}
            + \frac{\exp(\mathrm{i}(\omega + \Delta\omega_{nm})\Delta t_g) - 1}
            {(\omega + \Delta\omega_{nm})^2}\right)\\
            &+  \frac{1-\delta_{pq}}{\mathrm{i}\Delta\omega_{pq}} \cdot
            \frac{\exp(\mathrm{i}(\omega + \Delta\omega_{nm}
                                  + \Delta\omega_{pq})\Delta t_g) - 1}
            {\mathrm{i}(\omega + \Delta\omega_{nm} + \Delta\omega_{pq})}\\
            &- \frac{1-\delta_{pq}}{\mathrm{i}\Delta\omega_{pq}} \cdot
            \frac{\exp(\mathrm{i}(\omega + \Delta\omega_{nm})\Delta t_g) - 1}
            {\mathrm{i}(\omega + \Delta\omega_{nm})}
    """
    d = eigvecs.shape[-1]
    n_dt = len(dt)
    E = omega

    # Precompute some transformation into eigenspace of control Hamiltonian
    path = ['einsum_path', (0, 1), (0, 1)]
    VBV = np.einsum('gji,ajk,gkl->gail', eigvecs.conj(), n_opers, eigvecs,
                    optimize=path)
    VCV = np.einsum('tnm,jnk,tkl->tjml', eigvecs.conj(), basis, eigvecs,
                    optimize=path)

    # Allocate memory
    R_g = np.empty((n_dt, len(n_opers), len(basis), len(E)), dtype=complex)
    R_g_d_s = np.empty(
        (n_dt, len(n_opers), len(basis), len(transf_control_operators[0]),
         len(E)), dtype=complex)
    # For calculating dR_g: d,d = p,q, d,d = m,n
    integral_deriv = np.empty((n_dt, len(E), d, d, d, d), dtype=complex)

    for g in range(n_dt):
        # Any combination of \omega_m-\omega_n (R_g), \omega_p-\omega_q (dR_g)
        dE = np.subtract.outer(eigvals[g], eigvals[g])
        # Any combination of \omega+\omega_m-\omega_n (R_g) or
        # \omega-\omega_m+\omega_n (dR_g)
        EdE = np.add.outer(E, dE)

        # 1) Calculation of the control matrix R_g at each time step
        integral_Rg = np.empty((len(E), d, d), dtype=complex)
        # Mask the integral to avoid convergence problems
        mask_Rg = np.abs(EdE*dt[g]) <= 1e-7
        integral_Rg[mask_Rg] = dt[g]
        integral_Rg[~mask_Rg] = (util.cexp(EdE[~mask_Rg]*dt[g]) - 1) \
            / (1j*(EdE[~mask_Rg]))

        R_g[g] = np.einsum('a,bcd,adc,hdc->abh', n_coeffs[:, g], VCV[g],
                           VBV[g], integral_Rg,
                           optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

        # Add the dependency of the susceptibilities
        # s_derivs has shape (n_nops, n_ctrl, n_dt)
        # VCV has shape (n_dt, d**2, d, d)
        # VBV has shape (n_dt, n_nops, d, d)
        # integral_Rg has shape (n_omega, d, d)
        # R_g_d_s shall be of shape (n_dt, n_nops, d**2, n_ctrl, n_omega)
        if s_derivs is not None:
            R_g_d_s[g] = np.einsum(
                'ae,bcd,adc,hdc->abeh', s_derivs[:, :, g], VCV[g], VBV[g],
                integral_Rg, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

        # 2) Calculation of derivatives of control matrices at each time step
        mask_deriv = (abs(dE) < 1e-15)  # case: \omega_p-\omega_q = 0
        # calculation if omega_diff = 0
        n_case = sum(sum(mask_deriv))
        a = dt[g]*util.cexp(EdE*dt[g]) / (1j * EdE) \
            + (util.cexp(EdE*dt[g]) - 1) / (EdE)**2
        integral_deriv[g, :, mask_deriv] = np.concatenate(([[a]*n_case]),
                                                          axis=0)
        # calculation if omega_diff != 0
        b1 = - (util.cexp(np.add.outer(EdE, dE[~mask_deriv])*dt[g]) - 1) \
            / (np.add.outer(EdE, dE[~mask_deriv])) / dE[~mask_deriv]
        b2 = + np.divide.outer(((util.cexp(EdE*dt[g]) - 1) / EdE), dE[~mask_deriv])
        integral_deriv[g, :, ~mask_deriv] = (b1 + b2).transpose(3, 0, 1, 2)

    # Computation of the derivative/ gradient
    I_circ_H = np.einsum('toijnm,thij->tohijnm', integral_deriv,
                         transf_control_operators)
    M_mat = (np.einsum('tjmk,tohknnm->tojhmn', VCV, I_circ_H)
             - np.einsum('tohmknm,tjkn->tojhmn', I_circ_H, VCV))
    gradient = 1j * np.einsum('at,tamn,tojhnm->tajho', n_coeffs, VBV, M_mat,
                              optimize=['einsum_path', (1, 2), (0, 1)])
    if s_derivs is not None:
        gradient += R_g_d_s
    ctrlmat_data = {'R_g': R_g, 'dR_g': gradient}
    return ctrlmat_data


def control_matrix_at_timestep_derivative_refactored(
        omega: Coefficients,
        dt: Coefficients,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        c_opers_transformed: ndarray,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        n_coeffs_deriv: Optional[Sequence[Coefficients]] = None,
        intermediates: Optional[Dict[str, ndarray]] = None
) -> Tuple[ndarray, ndarray]:
    """"."""
    d = eigvecs.shape[-1]
    n_dt = len(dt)
    n_ctrl = len(c_opers_transformed[0])
    n_nops = len(n_opers)
    E = omega

    # Precompute some transformation into eigenspace of control Hamiltonian
    path = ['einsum_path', (0, 1), (0, 1)]
    VBV = np.einsum('gji,ajk,gkl->gail', eigvecs.conj(), n_opers, eigvecs, optimize=path)
    VCV = np.einsum('tnm,jnk,tkl->tjml', eigvecs.conj(), basis, eigvecs, optimize=path)

    # Allocate memory
    R_g = np.empty((n_dt, n_nops, d**2, len(E)), dtype=complex)
    buffer = np.empty((n_nops, d**2, len(E)), dtype=complex)
    R_g_d_s = np.empty((n_dt, n_nops, d**2, n_ctrl, len(E)), dtype=complex)
    # For calculating dR_g: d,d = p,q, d,d = m,n
    integral_deriv = np.empty((n_dt, len(E), d, d, d, d), dtype=complex)
    exp_buf, integral_Rg = np.empty((2, len(E), d, d), dtype=complex)

    expr = oe.contract_expression('bcd,adc,hdc->abh', VCV[0].shape, VBV[0].shape,
                                  integral_Rg.shape, optimize=[(0, 1), (0, 1)])
    for g in range(n_dt):
        integral_Rg = numeric._first_order_integral(E, eigvals[g], dt[g], exp_buf, integral_Rg)

        buffer = expr(VCV[g], VBV[g], integral_Rg, out=buffer)
        R_g[g] = n_coeffs[:, g, None, None] * buffer

        # Add the dependency of the susceptibilities
        # n_coeffs_deriv has shape (n_nops, n_ctrl, n_dt)
        # VCV has shape (n_dt, d**2, d, d)
        # VBV has shape (n_dt, n_nops, d, d)
        # integral_Rg has shape (n_omega, d, d)
        # R_g_d_s shall be of shape (n_dt, n_nops, d**2, n_ctrl, n_omega)
        if n_coeffs_deriv is not None:
            R_g_d_s[g] = n_coeffs_deriv[:, None, :, g, None] * buffer[:, :, None]

        integral_deriv[g] = _derivative_integral(E, eigvals[g], dt[g], integral_deriv[g])

    # Computation of the derivative/ gradient
    # I_circ_H = np.einsum('toijnm,thij->tohijnm', integral_deriv, c_opers_transformed)
    I_circ_H = integral_deriv[:, :, None] * c_opers_transformed[:, None, ..., None, None]
    if d == 2:
        # Life a bit simpler
        mask = np.eye(d, dtype=bool)
        M_mat = np.zeros((n_dt, len(E), d**2, n_ctrl, d, d), dtype=complex)
        M_mat[:, :, 1:] = np.einsum('tjmk,tohknnm->tojhmn', VCV[:, 1:], I_circ_H)
        M_mat[:, :, 1:, :, mask] -= M_mat[:, :, 1:, :, mask][..., ::-1]
        M_mat[:, :, 1:, :, ~mask] *= 2
        # M_mat -= M_mat.conj().swapaxes(-1, -2)
    else:
        M_mat = np.einsum('tjmk,tohknnm->tojhmn', VCV, I_circ_H)
        M_mat -= np.einsum('tjkn,tohmknm->tojhmn', VCV, I_circ_H)

    ctrlmat_g_deriv = 1j*np.einsum('tamn,tojhnm->tajho', n_coeffs.T[..., None, None] * VBV, M_mat)

    if n_coeffs_deriv is not None:
        ctrlmat_g_deriv += R_g_d_s

    return R_g, ctrlmat_g_deriv


def control_matrix_at_timestep_derivative_faster(
        omega: Coefficients,
        dt: Coefficients,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        c_opers_transformed: ndarray,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        n_coeffs_deriv: Optional[Sequence[Coefficients]] = None
) -> Tuple[ndarray, ndarray]:
    """"."""
    d = eigvecs.shape[-1]
    d2 = d**2
    n_dt = len(dt)
    n_ctrl = len(c_opers_transformed[0])
    n_nops = len(n_opers)
    n_omega = len(omega)
    E = omega

    # Precompute some transformation into eigenspace of control Hamiltonian
    n_opers_transformed = numeric._transform_hamiltonian(eigvecs, n_opers, n_coeffs).swapaxes(0, 1)
    basis_transformed = numeric._transform_by_unitary(eigvecs[:, None], basis[None],
                                                      out=np.empty((n_dt, d2, d, d), complex))

    # Allocate memory
    ctrlmat_g = np.empty((n_dt, n_nops, d2, len(E)), dtype=complex)
    ctrlmat_g_deriv_s = np.empty((n_dt, n_nops, d2, n_ctrl, len(E)), dtype=complex)
    # For calculating dR_g: d,d = p,q, d,d = m,n
    deriv_integral = np.empty((n_dt, len(E), d, d, d, d), dtype=complex)
    exp_buf, integral = np.empty((2, len(E), d, d), dtype=complex)

    expr = oe.contract_expression('bcd,adc,hdc->abh', basis.shape, basis.shape,
                                  integral.shape, optimize=[(0, 1), (0, 1)])
    for g in range(n_dt):
        integral = numeric._first_order_integral(E, eigvals[g], dt[g], exp_buf, integral)
        deriv_integral[g] = _derivative_integral(E, eigvals[g], dt[g], deriv_integral[g])

        ctrlmat_g[g] = expr(basis_transformed[g], n_opers_transformed[g], integral,
                            out=ctrlmat_g[g])
        if n_coeffs_deriv is not None:
            # equivalent contraction: 'ah,a,ako->akho', but this faster
            ctrlmat_g_deriv_s[g] = (
                (n_coeffs_deriv[..., g] / n_coeffs[:, g, None])[:, None, :, None]
                * ctrlmat_g[g, :, :, None]
            )

    # Basically a tensor product
    # K = np.einsum('taij,thkl->tahikjl', VBV, VHV)
    # L = K.transpose(0, 1, 2, 4, 3, 6, 5)
    K = util.tensor(n_opers_transformed[:, :, None], c_opers_transformed[:, None])
    L = util.tensor_transpose(K, (1, 0), [[d, d], [d, d]])
    k = np.diagonal(K.reshape(n_dt, n_nops, n_ctrl, d, d, d, d), 0, -2, -3)
    l = np.diagonal(L.reshape(n_dt, n_nops, n_ctrl, d, d, d, d), 0, -2, -3)
    i1 = np.diagonal(deriv_integral, 0, -2, -3)
    i2 = np.diagonal(deriv_integral, 0, -1, -4)
    # Let magic ensue. Reshaping in F-major is faster than not (~ factor 2-4)
    M = np.einsum(
        'tahpm,topm->tahop',
        l.reshape(n_dt, n_nops, n_ctrl, d2, d, order='F'),
        i1.reshape(n_dt, n_omega, d2, d, order='F')
    ).reshape(n_dt, n_nops, n_ctrl, n_omega, d, d, order='F')
    if d == 2:
        # Life a bit simpler
        mask = np.eye(d, dtype=bool)
        M[..., mask] -= M[..., mask][..., ::-1]
        M[..., ~mask] *= 2
    else:
        M -= np.einsum(
            'tahpn,topn->tahop',
            k.swapaxes(-2, -3).reshape(n_dt, n_nops, n_ctrl, d2, d, order='F'),
            i2.reshape(n_dt, n_omega, d2, d, order='F')
        ).reshape(n_dt, n_nops, n_ctrl, n_omega, d, d, order='F').swapaxes(-1, -2)

    # Expand in basis transformed to eigenspace
    ctrlmat_g_deriv = np.einsum('tjnk,tahokn->tajho', 1j*basis_transformed, M)

    if n_coeffs_deriv is not None:
        ctrlmat_g_deriv += ctrlmat_g_deriv_s

    return ctrlmat_g, ctrlmat_g_deriv


def _control_matrix_at_timestep_derivative_loop_1(
        omega: Coefficients,
        dt: Coefficients,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis_transformed,
        c_opers_transformed: ndarray,
        n_opers_transformed,
        n_coeffs: Sequence[Coefficients],
        n_coeffs_deriv: Sequence[Coefficients],
        ctrlmat_step,
        deriv_integral,
        exp_buf,
        integral,
        expr
) -> Tuple[ndarray, ndarray]:
    """"."""
    d = len(eigvecs)
    d2 = d**2
    n_ctrl = len(c_opers_transformed)
    n_nops = len(n_opers_transformed)
    n_omega = len(omega)

    integral = numeric._first_order_integral(omega, eigvals, dt, exp_buf, integral)
    deriv_integral = _derivative_integral(omega, eigvals, dt, deriv_integral)
    ctrlmat_step = expr(basis_transformed, n_opers_transformed, integral, out=ctrlmat_step)
    if n_coeffs_deriv is not None:
        # equivalent contraction: 'ah,a,ako->akho', but this faster
        ctrlmat_step_deriv_s = ((n_coeffs_deriv / n_coeffs[:, None])[:, None, :, None]
                                * ctrlmat_step[:, :, None])

    # Basically a tensor product
    # K = np.einsum('taij,thkl->tahikjl', VBV, VHV)
    # L = K.transpose(0, 1, 2, 4, 3, 6, 5)
    K = util.tensor(n_opers_transformed[:, None], c_opers_transformed[None])
    L = util.tensor_transpose(K, (1, 0), [[d, d], [d, d]])
    k = np.diagonal(K.reshape(n_nops, n_ctrl, d, d, d, d), 0, -2, -3)
    l = np.diagonal(L.reshape(n_nops, n_ctrl, d, d, d, d), 0, -2, -3)
    i1 = np.diagonal(deriv_integral, 0, -2, -3)
    i2 = np.diagonal(deriv_integral, 0, -1, -4)
    # Let magic ensue. Reshaping in F-major is faster than not (~ factor 2-4)
    M = np.einsum(
        'ahpm,opm->ahop',
        l.reshape(n_nops, n_ctrl, d2, d, order='F'),
        i1.reshape(n_omega, d2, d, order='F')
    ).reshape(n_nops, n_ctrl, n_omega, d, d, order='F')
    if d == 2:
        # Life a bit simpler
        mask = np.eye(d, dtype=bool)
        M[..., mask] -= M[..., mask][..., ::-1]
        M[..., ~mask] *= 2
    else:
        M -= np.einsum(
            'ahpn,opn->ahop',
            k.swapaxes(-2, -3).reshape(n_nops, n_ctrl, d2, d, order='F'),
            i2.reshape(n_omega, d2, d, order='F')
        ).reshape(n_nops, n_ctrl, n_omega, d, d, order='F').swapaxes(-1, -2)

    # Expand in basis transformed to eigenspace
    ctrlmat_step_deriv = np.einsum('jnk,ahokn->ajho', 1j*basis_transformed, M)

    if n_coeffs_deriv is not None:
        ctrlmat_step_deriv += ctrlmat_step_deriv_s

    return ctrlmat_step, ctrlmat_step_deriv


def _control_matrix_at_timestep_derivative_loop_2(
        omega: Coefficients,
        dt: Coefficients,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis_transformed,
        c_opers_transformed: ndarray,
        n_opers_transformed,
        n_coeffs: Sequence[Coefficients],
        n_coeffs_deriv: Sequence[Coefficients],
        ctrlmat_step,
        phase_factor,
        deriv_integral,
        exp_buf,
        integral,
        expr,
        out
) -> Tuple[ndarray, ndarray]:
    """"."""
    d = len(eigvecs)
    d2 = d**2
    n_ctrl = len(c_opers_transformed)
    n_nops = len(n_opers_transformed)
    n_omega = len(omega)

    deriv_integral = _derivative_integral(omega, eigvals, dt, out=deriv_integral)

    # Basically a tensor product
    # K = np.einsum('taij,thkl->tahikjl', VBV, VHV)
    # L = K.transpose(0, 1, 2, 4, 3, 6, 5)
    K = util.tensor(n_opers_transformed[:, None], c_opers_transformed[None])
    L = util.tensor_transpose(K, (1, 0), [[d, d], [d, d]])
    k = np.diagonal(K.reshape(n_nops, n_ctrl, d, d, d, d), 0, -2, -3)
    l = np.diagonal(L.reshape(n_nops, n_ctrl, d, d, d, d), 0, -2, -3)
    i1 = np.diagonal(deriv_integral, 0, -2, -3)
    i2 = np.diagonal(deriv_integral, 0, -1, -4)
    # reshaping in F-major is significantly faster than C-major (~ factor 2-4)
    M = np.einsum(
        'ahpm,opm->ahop',
        l.reshape(n_nops, n_ctrl, d2, d, order='F'),
        i1.reshape(n_omega, d2, d, order='F')
    ).reshape(n_nops, n_ctrl, n_omega, d, d, order='F')
    if d == 2:
        # Life a bit simpler
        mask = np.eye(d, dtype=bool)
        M[..., mask] -= M[..., mask][..., ::-1]
        M[..., ~mask] *= 2
    else:
        M -= np.einsum(
            'ahpn,opn->ahop',
            k.swapaxes(-2, -3).reshape(n_nops, n_ctrl, d2, d, order='F'),
            i2.reshape(n_omega, d2, d, order='F')
        ).reshape(n_nops, n_ctrl, n_omega, d, d, order='F').swapaxes(-1, -2)

    # Expand in basis transformed to eigenspace
    ctrlmat_step_deriv = np.einsum('jnk,ahokn->ajho', 1j*basis_transformed, M)
    ctrlmat_step_deriv *= phase_factor

    if n_coeffs_deriv is not None:
        # equivalent contraction: 'ah,a,ako->akho', but this faster
        ctrlmat_step_deriv += ((n_coeffs_deriv / n_coeffs[:, None])[:, None, :, None]
                               * ctrlmat_step[:, :, None])

    return ctrlmat_step_deriv


def noise_operators_at_timestep_derivative(
        omega: Coefficients,
        dt: Coefficients,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        VHV: ndarray,
        s_derivs: Optional[Sequence[Coefficients]] = None,
        show_progressbar: Optional[bool] = None,
        intermediates: Optional[Dict[str, ndarray]] = None
):
    d = eigvecs.shape[-1]
    n_dt = len(dt)
    n_ctrl = len(VHV[0])
    n_nops = len(n_opers)
    E = omega

    if intermediates is None:
        noise_operators = np.empty((n_dt, len(E), n_nops, d, d), dtype=complex)
        VBV = numeric._transform_hamiltonian(eigvecs, n_opers, n_coeffs).swapaxes(0, 1)
        exp_buf, integral = np.empty((2, len(E), d, d), dtype=complex)
    else:
        noise_operators = intermediates['noise_operators_step']
        VBV = intermediates['n_opers_transformed'].swapaxes(0, 1)

    # R_g_d_s = np.empty((n_dt, len(E), n_nops, n_ctrl, d, d), dtype=complex)
    # For calculating dR_g: d,d = p,q, d,d = m,n
    integral_deriv = np.empty((n_dt, len(E), d, d, d, d), dtype=complex)

    expr_1 = oe.contract_expression('akl,okl->oakl', VBV[:, 0].shape, (len(E), d, d))
    expr_2 = oe.contract_expression('ij,...jk,lk',
                                    eigvecs[0].shape, noise_operators[0].shape, eigvecs[0].shape,
                                    optimize=[(0, 1), (0, 1)])
    for g in range(n_dt):
        if intermediates is None:
            integral = numeric._first_order_integral(E, eigvals[g], dt[g], exp_buf, integral)
            noise_operators[g] = expr_1(VBV[g], integral, out=noise_operators[g])
            noise_operators[g] = expr_2(eigvecs[g], noise_operators[g], eigvecs[g].conj(),
                                        out=noise_operators[g])

        # Add the dependency of the susceptibilities
        # s_derivs has shape (n_nops, n_ctrl, n_dt)
        # VCV has shape (n_dt, d**2, d, d)
        # VBV has shape (n_dt, n_nops, d, d)
        # integral_Rg has shape (n_omega, d, d)
        # R_g_d_s shall be of shape (n_dt, n_nops, d**2, n_ctrl, n_omega)
        # if s_derivs is not None:
            # Still need to think about n_coeffs here
            # R_g_d_s[g] = s_derivs[:, None, :, g, None] * noise_operators[g]

        integral_deriv[g] = _derivative_integral(E, eigvals[g], dt[g], integral_deriv[g])

    # (np.einsum('tcmn,tanp,topmn->tocamp', VHV, VBV, I1, optimize=['einsum_path', (0, 1), (0, 1)]) -
    #   np.einsum('tcnp,tamn,tompn->tocamp', VHV, VBV, I2, optimize=['einsum_path', (0, 1), (0, 1)]))
    I1 = np.diagonal(integral_deriv, 0, -4, -1)
    path = [(0, 1), (0, 1)]  # optimization says [(0, 1), (0, 1)]
    # M = oe.contract('tcmn,tanp,topmn->tocamp', VHV, VBV, I1, optimize=path)
    M = oe.contract('tcmn,tanp,tompn->tocamp', VHV, VBV, I1, optimize=path)
    if d == 2:
        mask = np.eye(2, dtype=bool)
        M[..., mask] -= M[..., mask][..., ::-1]
        M[..., ~mask] *= 2
    else:
        I2 = np.diagonal(integral_deriv, 0, -3, -2)
        # M -= oe.contract('tcnp,tamn,tompn->tocamp', VHV, VBV, I2, optimize=path)
        M -= oe.contract('tcnp,tamn,topmn->tocamp', VHV, VBV, I2, optimize=path)

    intermediates = intermediates or dict()
    intermediates.setdefault('noise_operators_step', noise_operators)
    intermediates.setdefault('n_opers_transformed', VBV.swapaxes(0, 1))

    return 1j*M, intermediates


def calculate_derivative_of_control_matrix_from_scratch(
        omega: Coefficients,
        propagators: ndarray,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        t: Coefficients,
        dt: Coefficients,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        c_opers: Sequence[Operator],
        all_identifiers: Sequence[str],
        control_identifiers: Optional[Sequence[str]] = None,
        s_derivs: Optional[Sequence[Coefficients]] = None) -> ndarray:
    r"""
    Calculate the derivative of the control matrix from scratch.

    Parameters
    ----------
    omega : array_like, shape (n_omega,)
        Frequencies, at which the pulse control matrix is to be evaluated.
    propagators : array_like, shape (n_dt+1, d, d)
        The propagators :math:`Q_g = P_g P_{g-1}\cdots P_0` as a (d, d) array
        with *d* the dimension of the Hilbert space.
    Q_Liou : ndarray, shape (n_dt+1, d**2, d**2)
        The Liouville representation of the cumulative control propagators
        U_c(t_g,0).
    eigvals : array_like, shape (n_dt, d)
        Eigenvalue vectors for each time pulse segment *g* with the first
        axis counting the pulse segment, i.e.
        ``HD == array([D_0, D_1, ...])``.
    eigvecs : array_like, shape (n_dt, d, d)
        Eigenvector matrices for each time pulse segment *g* with the first
        axis counting the pulse segment, i.e.
        ``HV == array([V_0, V_1, ...])``.
    basis : Basis, shape (d**2, d, d)
        The basis elements, in which the pulse control matrix will be expanded.
    t : array_like, shape (n_dt+1), optional
        The absolute times of the different segments.
    dt : array_like, shape (n_dt)
        Sequence duration, i.e. for the :math:`g`-th pulse
        :math:`t_g - t_{g-1}`.
    n_opers : array_like, shape (n_nops, d, d)
        Noise operators :math:`B_\alpha`.
    n_coeffs : array_like, shape (n_nops, n_dt)
        The sensitivities of the system to the noise operators given by
        *n_opers* at the given time step.
    c_opers : array_like, shape (n_cops, d, d)
        Control operators :math:`H_k`.
    all_identifiers : array_like, shape (n_cops)
        Identifiers of all control operators.
    control_identifiers : Sequence[str], shape (n_ctrl), Optional
        Sequence of strings with the control identifiers to distinguish between
        accessible control and drift Hamiltonian. The default is None.
    s_derivs : array_like, shape (n_nops, n_ctrl, n_dt)
        The derivatives of the noise susceptibilities by the control amplitudes.
        Defaults to None.

    Raises
    ------
    ValueError
        If the given identifiers *control_identifier* are not subset of the
        total identifiers *all_identifiers* of all control operators.

    Returns
    -------
    dR : array_like, shape (n_nops, d**2, n_dt, n_ctrl, n_omega)
        Partial derivatives of the total control matrix with respect to each
        control direction
        :math:`\frac{\partial R_{\alpha k}(\omega)}{\partial u_h(t_{g'})}`.
        The array's indexing has shape :math:`(\alpha,k,g^\prime,h,\omega)`.

    Notes
    -----
    The derivative of the control matrix is calculated according to

    .. math ::

        \frac{\partial R_{\alpha k}(\omega)}{\partial u_h(t_{g'})} =
            \sum_{g=1}^G \mathrm{e}^{\mathrm{i}\omega t_{g-1}}\cdot\left(\sum_j
                \left[\frac{\partial R_{\alpha j}^{(g)}(\omega)}
                    {\partial u_h(t_{g'})} \cdot \mathcal{Q}_{jk}^{(g-1)}
                +   R_{\alpha j}^{(g)}(\omega)
                \cdot\frac{\partial \mathcal{Q}_{jk}^{(g-1)}}
                {\partial u_h(t_{g'})} \right] \right)

    See Also
    --------
    :func:`liouville_derivative`
    :func:`control_matrix_at_timestep_derivative`
    """
    # Distinction between control and drift operators and only calculate the
    # derivatives in control direction

    path = ['einsum_path', (0, 1), (0, 1)]
    if (control_identifiers is None):
        VHV = np.einsum('tji,hjk,tkl->thil', eigvecs.conj(), c_opers, eigvecs,
                        optimize=path)
    elif (set(control_identifiers) <= set(all_identifiers)):
        dict_id_oper = dict(zip(all_identifiers, c_opers))
        control = [dict_id_oper[element] for element in control_identifiers]
        VHV = np.einsum('tji,hjk,tkl->thil', eigvecs.conj(), control, eigvecs,
                        optimize=path)
    else:
        raise ValueError('Given control identifiers have to be a \
                         subset of (drift+control) Hamiltonian!')

    Q_Liou = superoperator.liouville_representation(propagators, basis)
    # Get derivative of Liouville, control matrix at each time step, derivative
    # derivative of control matrix at each time step
    dL = liouville_derivative(
        dt=dt, propagators=propagators, basis=basis, eigvecs=eigvecs,
        eigvals=eigvals, transf_control_operators=VHV)
    ctrlmat_data = control_matrix_at_timestep_derivative(
        omega=omega,
        dt=dt,
        eigvals=eigvals,
        eigvecs=eigvecs,
        basis=basis,
        n_opers=n_opers,
        n_coeffs=n_coeffs,
        transf_control_operators=VHV,
        s_derivs=s_derivs
    )
    ctrlmat_g = ctrlmat_data['R_g']
    ctrlmat_g_deriv = ctrlmat_data['dR_g']

    # Calculate the derivative of the total control matrix
    exp_factor = util.cexp(np.multiply.outer(t, omega))
    summand1 = np.einsum('to,tajho,tjk->aktho', exp_factor[:-1],
                         ctrlmat_g_deriv, Q_Liou[:-1],
                         optimize=['einsum_path', (1, 2), (0, 1)])
    summand2 = np.einsum('to,tajo,thsjk->aksho', exp_factor[1:-1],
                         ctrlmat_g[1:], dL[:-1],
                         optimize=['einsum_path', (0, 1), (0, 1)])

    dR = summand1 + summand2
    return dR


def calculate_derivative_of_control_matrix_from_scratch_refactored(
        omega: Coefficients,
        propagators: ndarray,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        t: Coefficients,
        dt: Coefficients,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        c_opers: Sequence[Operator],
        all_identifiers: Sequence[str],
        control_identifiers: Optional[Sequence[str]] = None,
        n_coeffs_deriv: Optional[Sequence[Coefficients]] = None) -> ndarray:
    r"""."""
    # Distinction between control and drift operators and only calculate the
    # derivatives in control direction

    if control_identifiers is None:
        control = c_opers
    elif set(control_identifiers) <= set(all_identifiers):
        dict_id_oper = dict(zip(all_identifiers, c_opers))
        control = [dict_id_oper[element] for element in control_identifiers]
    else:
        raise ValueError('Given control identifiers have to be a \
                         subset of (drift+control) Hamiltonian!')

    c_opers_transformed = numeric._transform_hamiltonian(eigvecs, control).swapaxes(0, 1)
    propagators_liouville = superoperator.liouville_representation(propagators[:-1], basis)

    propagators_liouville_deriv = liouville_derivative_vectorized(dt, propagators, basis, eigvecs,
                                                                  eigvals, c_opers_transformed)
    # ctrlmat_g, ctrlmat_g_deriv = control_matrix_at_timestep_derivative_refactored(
    ctrlmat_g, ctrlmat_g_deriv = control_matrix_at_timestep_derivative_faster(
        omega, dt, eigvals, eigvecs, basis, c_opers_transformed, n_opers, n_coeffs, n_coeffs_deriv
    )

    # Calculate the derivative of the total control matrix
    exp_factor = util.cexp(np.multiply.outer(omega, t[:-1]))
    # Equivalent (but slower) einsum contraction for first term:
    # ctrlmat_deriv = np.einsum('ot,tajho,tjk->hotak',
    #                           exp_factor, ctrlmat_g_deriv, propagators_liouville,
    #                           optimize=['einsum_path', (0, 2), (0, 1)])
    ctrlmat_deriv = (ctrlmat_g_deriv.transpose(3, 4, 0, 1, 2)
                     @ (exp_factor[..., None, None] * propagators_liouville))
    ctrlmat_deriv += np.einsum('ot,tajo,thsjk->hosak',
                               exp_factor[:, 1:], ctrlmat_g[1:], propagators_liouville_deriv,
                               optimize=['einsum_path', (0, 1), (0, 1)])

    return ctrlmat_deriv


def calculate_derivative_of_control_matrix_from_scratch_loop_1(
        omega: Coefficients,
        propagators: ndarray,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        t: Coefficients,
        dt: Coefficients,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        c_opers: Sequence[Operator],
        intermediates,
        all_identifiers: Sequence[str],
        control_identifiers: Optional[Sequence[str]] = None,
        n_coeffs_deriv: Optional[Sequence[Coefficients]] = None
) -> ndarray:
    r"""."""
    # Distinction between control and drift operators and only calculate the
    # derivatives in control direction
    if control_identifiers is None:
        control = c_opers
    elif set(control_identifiers) <= set(all_identifiers):
        dict_id_oper = dict(zip(all_identifiers, c_opers))
        control = [dict_id_oper[element] for element in control_identifiers]
    else:
        raise ValueError('Given control identifiers have to be a \
                         subset of (drift+control) Hamiltonian!')

    d = eigvecs.shape[-1]
    d2 = d**2
    n_dt = len(dt)
    n_ctrl = len(control)
    n_nops = len(n_opers)
    n_omega = len(omega)

    # Precompute some transformation into eigenspace of control Hamiltonian
    c_opers_transformed = numeric._transform_hamiltonian(eigvecs, control).swapaxes(0, 1)
    n_opers_transformed = numeric._transform_hamiltonian(eigvecs, n_opers, n_coeffs).swapaxes(0, 1)
    basis_transformed = numeric._transform_by_unitary(eigvecs[:, None], basis[None],
                                                      out=np.empty((n_dt, d2, d, d), complex))
    propagators_liouville = superoperator.liouville_representation(propagators[:-1], basis)

    propagators_liouville_deriv = liouville_derivative_vectorized(dt, propagators, basis, eigvecs,
                                                                  eigvals, c_opers_transformed)
    exp_factor = util.cexp(np.multiply.outer(omega, t[:-1]))

    # Allocate memory
    ctrlmat_deriv = np.empty((n_ctrl, n_omega, n_dt, n_nops, d2), dtype=complex)
    ctrlmat_step = np.empty((n_dt, n_nops, d2, n_omega), dtype=complex)
    deriv_integral = np.empty((n_omega, d, d, d, d), dtype=complex)
    exp_buf, integral = np.empty((2, n_omega, d, d), dtype=complex)
    ctrlmat_expr = oe.contract_expression('bcd,adc,hdc->abh', basis.shape, basis.shape,
                                          integral.shape, optimize=[(0, 1), (0, 1)])
    for g in range(n_dt):
        ctrlmat_step[g], ctrlmat_step_deriv = _control_matrix_at_timestep_derivative_loop_1(
            omega, dt[g], eigvals[g], eigvecs[g], basis_transformed[g], c_opers_transformed[g],
            n_opers_transformed[g], n_coeffs[:, g], n_coeffs_deriv[:, :, g], ctrlmat_step[g],
            deriv_integral, exp_buf, integral, ctrlmat_expr
        )

        # Calculate the derivative of the total control matrix
        ctrlmat_deriv[:, :, g] = ((exp_factor[:, g] * ctrlmat_step_deriv).transpose(2, 3, 0, 1)
                                  @ propagators_liouville[g])

    ctrlmat_deriv += oe.contract('ot,tajo,thsjk->hosak',
                                 exp_factor[:, 1:], ctrlmat_step[1:], propagators_liouville_deriv,
                                 optimize=[(0, 1), (0, 1)])

    return ctrlmat_deriv


def calculate_derivative_of_control_matrix_from_scratch_loop_2(
        omega: Coefficients,
        propagators: ndarray,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        t: Coefficients,
        dt: Coefficients,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        c_opers: Sequence[Operator],
        intermediates,
        all_identifiers: Sequence[str],
        control_identifiers: Optional[Sequence[str]] = None,
        n_coeffs_deriv: Optional[Sequence[Coefficients]] = None
) -> ndarray:
    r"""."""
    # Distinction between control and drift operators and only calculate the
    # derivatives in control direction
    if control_identifiers is None:
        control = c_opers
    elif set(control_identifiers) <= set(all_identifiers):
        dict_id_oper = dict(zip(all_identifiers, c_opers))
        control = [dict_id_oper[element] for element in control_identifiers]
    else:
        raise ValueError('Given control identifiers have to be a \
                         subset of (drift+control) Hamiltonian!')

    d = eigvecs.shape[-1]
    d2 = d**2
    n_dt = len(dt)
    n_ctrl = len(control)
    n_nops = len(n_opers)
    n_omega = len(omega)

    # Precompute some transformation into eigenspace of control Hamiltonian
    c_opers_transformed = numeric._transform_hamiltonian(eigvecs, control).swapaxes(0, 1)
    n_opers_transformed = numeric._transform_hamiltonian(eigvecs, n_opers, n_coeffs).swapaxes(0, 1)
    basis_transformed = numeric._transform_by_unitary(eigvecs[:, None], basis[None],
                                                      out=np.empty((n_dt, d2, d, d), complex))
    # U_deriv = liouville_derivative_vectorized_loop(dt, propagators, basis, eigvecs, eigvals,
    #                                                c_opers_transformed)
    propagators_liouville = superoperator.liouville_representation(propagators[:-1], basis)

    # propagators_liouville_deriv = np.empty((n_ctrl, n_dt, d2, d2), dtype=complex)
    # propagators_liouville_deriv = liouville_derivative_vectorized(dt, propagators, basis, eigvecs,
    #                                                               eigvals, c_opers_transformed)
    propagators_deriv = propagators_derivative_vectorized_completely(dt, propagators, basis,
                                                                     eigvecs, eigvals,
                                                                     c_opers_transformed)
    liouville_deriv = np.empty((n_ctrl, n_dt, d2, d2), dtype=complex)
    # Need to 'remove' the propagators from the control matrix at time step as computed by
    # numeric.calculate_control_matrix_from_scratch
    ctrlmat_step = (intermediates['control_matrix_step'].transpose(3, 0, 1, 2)
                    @ propagators_liouville.conj().swapaxes(-1, -2)).transpose(1, 2, 3, 0)
    # exp_factor = util.cexp(np.multiply.outer(omega, t[:-1]))

    # Allocate memory
    ctrlmat_deriv = np.empty((n_ctrl, n_omega, n_dt, n_nops, d2), dtype=complex)
    deriv_integral = np.empty((n_omega, d, d, d, d), dtype=complex)
    ctrlmat_step_deriv = np.empty((n_nops, d2, n_ctrl, n_omega), dtype=complex)
    exp_buf, integral = np.empty((2, n_omega, d, d), dtype=complex)
    ctrlmat_expr = oe.contract_expression('bcd,adc,hdc->abh', basis.shape, basis.shape,
                                          integral.shape, optimize=[(0, 1), (0, 1)])
    liouville_deriv_expr = oe.contract_expression('hsba,jbc,cd,kda->hsjk',
                                                  propagators_deriv.conj()[:, 0].shape,
                                                  basis.shape, propagators[0].shape, basis.shape,
                                                  optimize=[(0, 2), (0, 2), (0, 1)])
    for g in range(n_dt):
        phase_factor = util.cexp(omega * t[g])
        # Calculate the whole propagator derivative
        # propagators_deriv[:, :g+1] = (propagators[g+1]
        #                               @ propagators[1:g+2].conj().swapaxes(-1, -2)
        #                               @ U_deriv[:, :g+1]
        #                               @ propagators[:g+1])

        # # Now calculate derivative of Liouville representation. Operation is faster using matmul, but
        # # not nice to read. Full einsum contraction would be as follows (we insert new axes to induce
        # # an outer product between basis elements steps using broadcasting):
        # # liouville_deriv = np.einsum('hsba,jbc,cd,kda->hsjk',
        # #                             deriv_U.conj(), basis, propagators[g+1], basis)
        # propagators_liouville_deriv = np.einsum('hsba,jkba->hsjk', propagators_deriv.conj(),
        #                                         (basis @ propagators[g+1])[:, None] @ basis,
        #                                         out=propagators_liouville_deriv)

        ctrlmat_step_deriv = _control_matrix_at_timestep_derivative_loop_2(
            omega, dt[g], eigvals[g], eigvecs[g], basis_transformed[g], c_opers_transformed[g],
            n_opers_transformed[g], n_coeffs[:, g], n_coeffs_deriv[:, :, g], ctrlmat_step[g],
            phase_factor, deriv_integral, exp_buf, integral, ctrlmat_expr, out=ctrlmat_step_deriv
        )

        # Calculate the derivative of the total control matrix (phase already included)
        ctrlmat_deriv[:, :, g] = (ctrlmat_step_deriv.transpose(2, 3, 0, 1)
                                  @ propagators_liouville[g])

        if g < n_dt - 1:
            liouville_deriv = liouville_derivative_vectorized_completely_loop(
                propagators_deriv[:, g], basis, propagators[g+1], liouville_deriv_expr,
                out=liouville_deriv
            )
            ctrlmat_deriv += oe.contract('ajo,hsjk->hosak',
                                         ctrlmat_step[g+1], liouville_deriv.real)

    # ctrlmat_deriv += oe.contract('ot,tajo,thsjk->hosak',
    #                              exp_factor[:, 1:], ctrlmat_g[1:], propagators_liouville_deriv,
    #                              optimize=[(0, 1), (0, 1)])
    # ctrlmat_deriv += oe.contract('otaj,thsjk->hosak', prefactor, propagators_liouville_deriv)

    return ctrlmat_deriv


def calculate_derivative_of_control_matrix_from_scratch_loop_3(
        omega: Coefficients,
        propagators: ndarray,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        t: Coefficients,
        dt: Coefficients,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        c_opers: Sequence[Operator],
        intermediates,
        all_identifiers: Sequence[str],
        control_identifiers: Optional[Sequence[str]] = None,
        n_coeffs_deriv: Optional[Sequence[Coefficients]] = None
) -> ndarray:
    r"""."""
    # Distinction between control and drift operators and only calculate the
    # derivatives in control direction
    if control_identifiers is None:
        control = c_opers
    elif set(control_identifiers) <= set(all_identifiers):
        dict_id_oper = dict(zip(all_identifiers, c_opers))
        control = [dict_id_oper[element] for element in control_identifiers]
    else:
        raise ValueError('Given control identifiers have to be a \
                         subset of (drift+control) Hamiltonian!')

    d = eigvecs.shape[-1]
    d2 = d**2
    n_dt = len(dt)
    n_ctrl = len(control)
    n_nops = len(n_opers)
    n_omega = len(omega)

    # Precompute some transformation into eigenspace of control Hamiltonian
    c_opers_transformed = numeric._transform_hamiltonian(eigvecs, control).swapaxes(0, 1)
    n_opers_transformed = numeric._transform_hamiltonian(eigvecs, n_opers, n_coeffs).swapaxes(0, 1)
    basis_transformed = numeric._transform_by_unitary(eigvecs[:, None], basis[None],
                                                      out=np.empty((n_dt, d2, d, d), complex))
    # U_deriv = liouville_derivative_vectorized_loop(dt, propagators, basis, eigvecs, eigvals,
    #                                                c_opers_transformed)
    propagators_liouville = superoperator.liouville_representation(propagators[:-1], basis)

    # propagators_liouville_deriv = np.empty((n_ctrl, n_dt, d2, d2), dtype=complex)
    # propagators_liouville_deriv = liouville_derivative_vectorized(dt, propagators, basis, eigvecs,
    #                                                               eigvals, c_opers_transformed)
    propagators_deriv = propagators_derivative_vectorized_completely(dt, propagators, basis,
                                                                     eigvecs, eigvals,
                                                                     c_opers_transformed)
    # Contraction 'jbc,tcd,kda->tjbka'
    basis_propagators = np.tensordot(np.tensordot(propagators[1:-1], basis, axes=[1, 2]),
                                     basis, axes=[1, 1])
    propagators_liouville_deriv =  oe.contract('tjnkm,htsnm->tsjkh',
                                               basis_propagators, propagators_deriv)

    # Need to 'remove' the propagators from the control matrix at time step as computed by
    # numeric.calculate_control_matrix_from_scratch
    ctrlmat_step = (intermediates['control_matrix_step'].transpose(3, 0, 1, 2)
                    @ propagators_liouville.conj().swapaxes(-1, -2)).transpose(1, 2, 3, 0)
    # exp_factor = util.cexp(np.multiply.outer(omega, t[:-1]))

    # Allocate memory
    ctrlmat_deriv = np.empty((n_ctrl, n_omega, n_dt, n_nops, d2), dtype=complex)
    deriv_integral = np.empty((n_omega, d, d, d, d), dtype=complex)
    ctrlmat_step_deriv = np.empty((n_nops, d2, n_ctrl, n_omega), dtype=complex)
    exp_buf, integral = np.empty((2, n_omega, d, d), dtype=complex)
    ctrlmat_expr = oe.contract_expression('bcd,adc,hdc->abh', basis.shape, basis.shape,
                                          integral.shape, optimize=[(0, 1), (0, 1)])
    liouville_deriv_expr = oe.contract_expression('tajo,htnm,jnp,tpq,kqm->hoak',
                                                  ctrlmat_step[1:].shape,
                                                  propagators_deriv.conj()[:, :, 0].shape,
                                                  basis.shape, propagators[1:-1].shape,
                                                  basis.shape,
                                                  optimize=[(2, 3), (2, 3), (1, 2), (0, 1)])
    for g in range(n_dt):
        phase_factor = util.cexp(omega * t[g])
        # Calculate the whole propagator derivative
        # propagators_deriv[:, :g+1] = (propagators[g+1]
        #                               @ propagators[1:g+2].conj().swapaxes(-1, -2)
        #                               @ U_deriv[:, :g+1]
        #                               @ propagators[:g+1])

        # # Now calculate derivative of Liouville representation. Operation is faster using matmul, but
        # # not nice to read. Full einsum contraction would be as follows (we insert new axes to induce
        # # an outer product between basis elements steps using broadcasting):
        # # liouville_deriv = np.einsum('hsba,jbc,cd,kda->hsjk',
        # #                             deriv_U.conj(), basis, propagators[g+1], basis)
        # propagators_liouville_deriv = np.einsum('hsba,jkba->hsjk', propagators_deriv.conj(),
        #                                         (basis @ propagators[g+1])[:, None] @ basis,
        #                                         out=propagators_liouville_deriv)

        ctrlmat_step_deriv = _control_matrix_at_timestep_derivative_loop_2(
            omega, dt[g], eigvals[g], eigvecs[g], basis_transformed[g], c_opers_transformed[g],
            n_opers_transformed[g], n_coeffs[:, g], n_coeffs_deriv[:, :, g], ctrlmat_step[g],
            phase_factor, deriv_integral, exp_buf, integral, ctrlmat_expr, out=ctrlmat_step_deriv
        )

        # Calculate the derivative of the total control matrix (phase already included)
        ctrlmat_deriv[:, :, g] = (ctrlmat_step_deriv.transpose(2, 3, 0, 1)
                                  @ propagators_liouville[g])

        propagators_liou_deriv = oe.contract('htba,jbc,tcd,kda->thjk', propagators_deriv[:, :, g],
                                             basis, propagators[1:-1], basis)
        ctrlmat_deriv[:, :, g] += oe.contract('tajo,thjk->hoak', ctrlmat_step[1:],
                                              propagators_liou_deriv)

        # ctrlmat_deriv[:, :, g] += oe.contract('tajo,htnm,tjnkm->hoak', ctrlmat_step[1:],
        #                                       propagators_deriv.conj()[:, :, g], basis_propagators,
        #                                       optimize=[(1, 2), (0, 1)])

        ctrlmat_deriv[:, :, g] += oe.contract('tajo,tjkh->hoak',
                                              ctrlmat_step[1:], propagators_liouville_deriv[:, g])

        # ctrlmat_deriv[:, :, g] += liouville_deriv_expr(ctrlmat_step[1:],
        #                                                 propagators_deriv.conj()[:, :, 0],
        #                                                 basis, propagators[1:-1], basis)

    # ctrlmat_deriv += oe.contract('ot,tajo,thsjk->hosak',
    #                              exp_factor[:, 1:], ctrlmat_step[1:], propagators_liouville_deriv,
    #                              optimize=[(0, 1), (0, 1)])
    # ctrlmat_deriv += oe.contract('otaj,thsjk->hosak', prefactor, propagators_liouville_deriv)

    return ctrlmat_deriv


def calculate_derivative_of_control_matrix_from_scratch_loop_4(
        omega: Coefficients,
        propagators: ndarray,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        t: Coefficients,
        dt: Coefficients,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        c_opers: Sequence[Operator],
        intermediates,
        all_identifiers: Sequence[str],
        control_identifiers: Optional[Sequence[str]] = None,
        n_coeffs_deriv: Optional[Sequence[Coefficients]] = None
) -> ndarray:
    r"""."""
    # Distinction between control and drift operators and only calculate the
    # derivatives in control direction
    if control_identifiers is None:
        control = c_opers
    elif set(control_identifiers) <= set(all_identifiers):
        dict_id_oper = dict(zip(all_identifiers, c_opers))
        control = [dict_id_oper[element] for element in control_identifiers]
    else:
        raise ValueError('Given control identifiers have to be a \
                         subset of (drift+control) Hamiltonian!')

    d = eigvecs.shape[-1]
    d2 = d**2
    n_dt = len(dt)
    n_ctrl = len(control)
    n_nops = len(n_opers)
    n_omega = len(omega)

    # Precompute some transformation into eigenspace of control Hamiltonian
    c_opers_transformed = numeric._transform_hamiltonian(eigvecs, control).swapaxes(0, 1)
    n_opers_transformed = numeric._transform_hamiltonian(eigvecs, n_opers, n_coeffs).swapaxes(0, 1)
    basis_transformed = numeric._transform_by_unitary(eigvecs[:, None], basis[None],
                                                      out=np.empty((n_dt, d2, d, d), complex))
    propagators_liouville = superoperator.liouville_representation(propagators[:-1], basis)

    propagators_liouville_deriv = liouville_derivative_vectorized(dt, propagators, basis, eigvecs,
                                                                  eigvals, c_opers_transformed)
    ctrlmat_step = intermediates['control_matrix_step']
    prefactor = (ctrlmat_step[1:].transpose(-1, 0, 1, 2)
                 @ propagators_liouville[1:].conj().swapaxes(-1, -2))

    # Allocate memory
    ctrlmat_step_deriv = np.empty((n_nops, d2, n_ctrl, n_omega), dtype=complex)
    ctrlmat_deriv = np.empty((n_ctrl, n_omega, n_dt, n_nops, d2), dtype=complex)
    deriv_integral = np.empty((n_omega, d, d, d, d), dtype=complex)
    exp_buf, integral = np.empty((2, n_omega, d, d), dtype=complex)
    ctrlmat_expr = oe.contract_expression('bcd,adc,hdc->abh', basis.shape, basis.shape,
                                          integral.shape, optimize=[(0, 1), (0, 1)])
    for g in range(n_dt):
        ctrlmat_step_deriv = _control_matrix_at_timestep_derivative_loop_2(
            omega, dt[g], eigvals[g], eigvecs[g], basis_transformed[g], c_opers_transformed[g],
            n_opers_transformed[g], n_coeffs[:, g], n_coeffs_deriv[:, :, g], ctrlmat_step[g],
            deriv_integral, exp_buf, integral, ctrlmat_expr, out=ctrlmat_step_deriv
        )

        # Calculate the derivative of the total control matrix
        exp_factor = util.cexp(omega * t[g])
        ctrlmat_deriv[:, :, g] = ((exp_factor * ctrlmat_step_deriv).transpose(2, 3, 0, 1)
                                  @ propagators_liouville[g])

    ctrlmat_deriv += oe.contract('otaj,thsjk->hosak', prefactor, propagators_liouville_deriv)

    return ctrlmat_deriv


def calculate_derivative_of_noise_operators_from_scratch(
        omega: Coefficients,
        propagators: ndarray,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis: Basis,
        t: Coefficients,
        dt: Coefficients,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        c_opers: Sequence[Operator],
        all_identifiers: Sequence[str],
        control_identifiers: Optional[Sequence[str]] = None,
        s_derivs: Optional[Sequence[Coefficients]] = None,
        intermediates: Optional[Dict[str, ndarray]] = None):

    if control_identifiers is None:
        control = c_opers
    elif set(control_identifiers) <= set(all_identifiers):
        dict_id_oper = dict(zip(all_identifiers, c_opers))
        control = [dict_id_oper[element] for element in control_identifiers]
    else:
        raise ValueError('Given control identifiers have to be a \
                         subset of (drift+control) Hamiltonian!')

    d = eigvecs.shape[-1]
    VHV = numeric._transform_hamiltonian(eigvecs, control).swapaxes(0, 1)
    VQ = numeric._propagate_eigenvectors(eigvecs, propagators[:-1])

    exp_factor = util.cexp(np.multiply.outer(t[:-1], omega))
    M, intermediates = noise_operators_at_timestep_derivative(omega, dt, eigvals, eigvecs, basis,
                                                              n_opers, n_coeffs, VHV,
                                                              intermediates=intermediates)

    QVMVQ = numeric._transform_by_unitary(M, VQ[:, None, None, None], out=M)
    QVMVQ *= exp_factor[..., None, None, None, None]

    noise_operators_step_pulse = intermediates['noise_operators_step_pulse']
    noise_operators_step_total = intermediates['noise_operators_step_total']

    propagators_deriv = propagator_derivative(dt, propagators, eigvecs, eigvals, VHV)
    propagators_deriv_factor = propagator_derivative_factor(dt, propagators, eigvecs, eigvals, VHV)

    noise_operators_deriv = np.zeros((len(omega), len(n_opers), d, len(c_opers), len(dt), d),
                                     dtype=complex)
    for g in range(len(dt)):
        noise_operators_deriv[..., :g+1, :] += np.tensordot(noise_operators_step_total[g],
                                                            propagators_deriv_factor[:, g, :g+1],
                                                            axes=[-1, -2])
        noise_operators_deriv[..., :g+1, :] += np.tensordot(noise_operators_step_total[g],
                                                            propagators_deriv_factor[:, g, :g+1].conj(),
                                                            axes=[-2, -2])

    noise_operators_deriv = np.zeros(QVMVQ.shape, dtype=complex)
    noise_operators_deriv = np.einsum(
        'so,sji,soajk,hstkl->tohail',
        exp_factor[:-1], propagators[1:-1].conj(), noise_operators_step_pulse[1:], propagators_deriv[:, :-1],
        optimize=['einsum_path', (0, 1), (0, 2), (0, 1)]
    )
    noise_operators_deriv += np.einsum(
        'so,hstji,soajk,skl->tohail',
        exp_factor[:-1], propagators_deriv[:, :-1].conj(), noise_operators_step_pulse[1:], propagators[1:-1],
        optimize=['einsum_path', (0, 1), (0, 2), (0, 1)]
    )

    noise_operators_deriv = np.zeros(QVMVQ.shape, dtype=complex)
    noise_operators_deriv.real = 2 * np.einsum(
        'to,hstjk,skl,soali->tohaji',
        exp_factor, propagators_deriv[:, :-1], propagators[1:-1], noise_operators_step_pulse[1:],
        optimize=['einsum_path', (1, 2), (1, 2), (0, 1)]
    ).real
    noise_operators_deriv += QVMVQ

    return noise_operators_deriv


def calculate_filter_function_derivative(R: ndarray, deriv_R: ndarray) -> ndarray:
    r"""
    Compute the filter function derivative from the control matrix.

    Parameters
    ----------
    R : array_like, shape (n_nops, d**2, n_omega)
        The control matrix.
    deriv_R: array_like, shape (n_nops, d**2, n_t, n_ctrl, n_omega)
        The derivative of the control matrix.

    Returns
    -------
    deriv_filter_function : ndarray, shape (n_nops, n_dt, n_ctrl, n_omega)
        The regular filter functions' derivatives for variation in each control
        contribution
        :math:`\frac{\partial F_\alpha(\omega)}{\partial u_h(t_{g'})}`.
        The array's indexing has shape :math:`(\alpha,g^\prime,h,\omega)`.

    Notes
    -----
    The filter function derivative is calculated according to

    .. math ::

        \frac{\partial F_\alpha(\omega)}{\partial u_h(t_{g'})}
                    = 2 \mathrm{Re} \left( \sum_k R_{\alpha k}^\ast(\omega)
                    \frac{\partial R_{\alpha k}(\omega)}
                    {\partial u_h(t_{g'})} \right)
    """
    summe = np.einsum('ako,aktho->atho', R.conj(), deriv_R)
    return 2*summe.real


def calculate_filter_function_derivative_refactored(R: ndarray, R_deriv: ndarray) -> ndarray:
    summe = np.einsum('ako,hotak->atho', R.conj(), R_deriv)
    return 2*summe.real


def infidelity_derivative(
        pulse: 'PulseSequence',
        S: Union[Coefficients, Callable],
        omega: Coefficients,
        control_identifiers: Optional[Sequence[str]] = None,
        s_derivs: Optional[Sequence[Coefficients]] = None
) -> ndarray:
    r"""
    Calculate the infidelity derivative.

    Calculate the entanglement infidelity derivative of the ``PulseSequence``
    *pulse*.

    Parameters
    ----------
    pulse : PulseSequence
        The ``PulseSequence`` instance, for which to calculate the infidelity
        for.
    S : array_like
        The two-sided noise power spectral density in units of inverse
        frequencies as an array of shape (n_omega,) or (n_nops, n_omega). In
        the first case, the same spectrum is taken for all noise operators, in
        the second, it is assumed that there are no correlations between
        different noise sources and thus there is one spectrum for each noise
        operator.
    omega : array_like, shape (n_omega)
        The frequencies at which the integration is to be carried out.
    control_identifiers : Sequence[str], shape (n_ctrl)
        Sequence of strings with the control identifiern to distinguish between
        accessible control and drift Hamiltonian.
    s_derivs : array_like, shape (n_nops, n_ctrl, n_dt)
        The derivatives of the noise susceptibilities by the control amplitudes.
        Defaults to None.

    Raises
    ------
    ValueError
        If the provided noise spectral density does not fit expected shape.

    Returns
    -------
    deriv_infid : ndarray, shape (n_nops, n_dt, n_ctrl)
        Array with the derivative of the infidelity for each noise source taken
        for each control direction at each time step
        :math:`\frac{\partial I_e}{\partial u_h(t_{g'})}`. The array's indexing
        has shape :math:`(\alpha,g^\prime,h)`.

    Notes
    -----
    The infidelity's derivative is given by

    .. math::

        \frac{\partial I_e}{\partial u_h(t_{g'})} = \frac{1}{d}
                                            \int_{-\infty}^\infty
                                            \frac{d\omega}{2\pi}
                                            S_\alpha(\omega)
                                            \frac{\partial F_\alpha (\omega)}
                                            {\partial u_h(t_{g'})}

    with :math:`S_{\alpha}(\omega)` the noise spectral density
    and :math:`F_{\alpha}(\omega)` the canonical filter function for
    noise source :math:`\alpha`.

    To convert to the average gate infidelity, use the
    following relation given by Horodecki et al. [Hor99]_ and
    Nielsen [Nie02]_:

    .. math::

        \big\langle\mathcal{I}_\mathrm{avg}\big\rangle = \frac{d}{d+1}
                \big\langle\mathcal{I}_\mathrm{e}\big\rangle.

    References
    ----------
    .. [Hor99]
        Horodecki, M., Horodecki, P., & Horodecki, R. (1999). General
        teleportation channel, singlet fraction, and quasidistillation.
        Physical Review A - Atomic, Molecular, and Optical Physics, 60(3),
        1888–1898. https://doi.org/10.1103/PhysRevA.60.1888

    .. [Nie02]
        Nielsen, M. A. (2002). A simple formula for the average gate
        fidelity of a quantum dynamical operation. Physics Letters,
        Section A: General, Atomic and Solid State Physics, 303(4), 249–252.
        https://doi.org/10.1016/S0375-9601(02)01272-0
    """
    n_ops = len(pulse.n_opers)
    S = np.asarray(S)
    if(S.shape[0] == n_ops):
        S_all = S
    elif(S.shape[0] == 1):
        S_all = np.array([S[0]]*n_ops)
    elif(S.shape[0] == len(omega)):
        S_all = np.array([S]*n_ops)
    else:
        raise ValueError('Not fitting shape of S.')

    filter_function_deriv = pulse.get_filter_function_derivative(omega, control_identifiers,
                                                                 s_derivs)

    integrand = np.einsum('ao,atho->atho', S_all, filter_function_deriv)
    infid_deriv = trapz(integrand, omega) / (2*np.pi*pulse.d)

    return infid_deriv

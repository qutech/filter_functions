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
infidelity derivatives. Currently only auto-correlated noise (i.e. no
cross-correlations) is implemented.

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
:func:`calculate_derivative_of_control_matrix_from_scratch`
    Calculate the derivative of the control matrix from scratch.
:func:`calculate_filter_function_derivative`
    Compute the filter function derivative from the control matrix.
:func:`infidelity_derivative`
    Calculate the infidelity derivative.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import opt_einsum as oe
from numpy import ndarray
from opt_einsum.contract import ContractExpression

from . import numeric, superoperator, util
from .basis import Basis
from .types import Coefficients, Operator

__all__ = ['calculate_derivative_of_control_matrix_from_scratch',
           'calculate_filter_function_derivative', 'infidelity_derivative']


def _derivative_integral(E: Coefficients, eigvals: Coefficients, dt: float,
                         out: ndarray) -> ndarray:
    """
    Compute the integral appearing in the derivative of the control
    matrix. Result (out) has shape (len(E), d, d, d, d).
    """
    # Precompute masks and energy differences
    dE = np.subtract.outer(eigvals, eigvals)
    mask_dE = np.abs(dE) < 1e-7
    EdE = np.add.outer(E, dE)
    mask_EdE = np.abs(EdE) < 1e-7
    EdEdE = np.add.outer(EdE, dE[~mask_dE])
    mask_EdEdE = np.abs(EdEdE) < 1e-7

    # Case Omega_pq == 0
    tmp1 = np.divide(util.cexp(EdE*dt), EdE, where=~mask_EdE)
    tmp2 = tmp1 - np.divide(1, EdE, where=~mask_EdE)
    tmp2[mask_EdE] = 1j * dt

    tmp1 *= -1j * dt
    tmp1 += np.divide(tmp2, EdE, where=~mask_EdE)
    tmp1[mask_EdE] = dt**2 / 2

    out[:, mask_dE] = tmp1[:, None]

    # Case Omega_pq != 0
    tmp1 = np.divide(1 - util.cexp(EdEdE*dt, where=~mask_EdEdE), EdEdE, where=~mask_EdEdE)
    tmp1[mask_EdEdE] = -1j * dt
    tmp1 += tmp2[..., None]

    out[:, ~mask_dE] = (tmp1 / dE[~mask_dE]).transpose(0, 3, 1, 2)
    return out


def _liouville_derivative(dt: Coefficients, propagators: ndarray, basis: Basis, eigvecs: ndarray,
                          eigvals: ndarray, c_opers_transformed: ndarray) -> ndarray:
    r"""
    Calculate the derivatives of the control propagators in Liouville
    representation.

    Parameters
    ----------
    dt: array_like, shape (n_dt)
        Sequence duration, i.e. for the :math:`g`-th pulse
        :math:`t_g - t_{g-1}`.
    propagators: array_like, shape (n_dt+1, d, d)
        The propagators :math:`Q_g = P_g P_{g-1}\cdots P_0` as a (d, d)
        array with *d* the dimension of the Hilbert space.
    basis: Basis, shape (d**2, d, d)
        The basis elements, in which the pulse control matrix will be
        expanded.
    eigvecs: array_like, shape (n_dt, d, d)
        Eigenvector matrices for each time pulse segment *g* with the
        first axis counting the pulse segment, i.e.
        ``HV == array([V_0, V_1, ...])``.
    eigvals: array_like, shape (n_dt, d)
        Eigenvalue vectors for each time pulse segment *g* with the
        first axis counting the pulse segment, i.e.
        ``HD == array([D_0, D_1, ...])``.
    c_opers_transformed: array_like, shape (n_dt, c_ctrl, d, d)
        The control operators transformed into the eigenspace of the
        control Hamiltonian. The drift operators are ignored, if
        identifiers for accessible control operators are provided.

    Returns
    -------
    liouville_deriv: array_like, shape (n_dt, n_ctrl, n_dt, d**2, d**2)
        The derivative of the control propagators in Liouville
        representation :math:`\frac{\partial \mathcal{Q}_{jk}^{(g)}}
        {\partial u_h(t_{g^\prime})}`.

    Notes
    -----
    The derivatives of the control propagators in Liouville
    representation are calculated according to

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
    n, d = eigvecs.shape[:2]
    # omega_i - omega_j
    omega_diff = eigvals[:, :, None] - eigvals[:, None, :]
    dt_broadcast = np.broadcast_to(dt[:, None, None], omega_diff.shape)
    # mask = omega_diff == 0
    mask = np.broadcast_to(np.eye(d, dtype=bool), omega_diff.shape)
    A_mat = np.empty(omega_diff.shape, dtype=complex)
    A_mat[mask] = dt_broadcast[mask]
    A_mat[~mask] = 1j*(1 - util.cexp(omega_diff[~mask]*dt_broadcast[~mask])) / omega_diff[~mask]
    U_deriv = -1j * (propagators[1:]
                     @ propagators[:-1].conj().swapaxes(-1, -2)
                     @ eigvecs
                     @ (A_mat * c_opers_transformed.swapaxes(0, 1))
                     @ eigvecs.conj().swapaxes(-1, -2))

    # Calculate the whole propagator derivative
    propagators_deriv = np.zeros((c_opers_transformed.shape[1], n-1, n, d, d), dtype=complex)
    U_deriv_transformed = np.zeros((c_opers_transformed.shape[1], n-1, d, d), dtype=complex)
    for g in range(n-1):
        U_deriv_transformed[:, g] = (propagators[g+1].conj().swapaxes(-1, -2)
                                     @ U_deriv[:, g]
                                     @ propagators[g])
        propagators_deriv[:, g, :g+1] = propagators[g+1] @ U_deriv_transformed[:, :g+1]

    # Equivalent but usually slower contraction: 'htsba,jbc,tcd,kda->thsjk'
    # Can just take 2*Re(·) when calculating x + x*
    liouville_deriv = np.einsum('htsba,tjkba->thsjk', propagators_deriv.conj(),
                                (basis @ propagators[1:-1, None])[:, :, None] @ basis).real
    liouville_deriv *= 2
    return liouville_deriv


def _control_matrix_at_timestep_derivative(
        omega: Coefficients,
        dt: Coefficients,
        eigvals: ndarray,
        eigvecs: ndarray,
        basis_transformed,
        c_opers_transformed: ndarray,
        n_opers_transformed: ndarray,
        n_coeffs: Sequence[Coefficients],
        n_coeffs_deriv: Sequence[Coefficients],
        phase_factor: ndarray,
        integral: ndarray,
        deriv_integral: ndarray,
        ctrlmat_step: ndarray,
        ctrlmat_expr: ContractExpression,
) -> Tuple[ndarray, ndarray]:
    r"""Calculate the control matrices and corresponding derivatives.

    Calculate control matrices at each time step and the corresponding
    partial derivatives of those with respect to control strength at
    each time step.

    Parameters
    ----------
    omega: array_like, shape (n_omega)
        Frequencies, at which the pulse control matrix is to be
        evaluated.
    dt: array_like, shape (n_dt)
        Sequence duration, i.e. for the :math:`g`-th pulse
        :math:`t_g - t_{g-1}`.
    eigvals: array_like, shape (n_dt, d)
        Eigenvalue vectors for each time pulse segment *g* with the
        first axis counting the pulse segment, i.e.
        ``D == array([D_0, D_1, ...])``.
    eigvecs: array_like, shape (n_dt, d, d)
        Eigenvector matrices for each time pulse segment *g* with the
        first axis counting the pulse segment, i.e.
        ``V == array([V_0, V_1, ...])``.
    basis_transformed: array_like, shape (d**2, d, d)
        The basis elements in which the pulse control matrix will be
        expanded transformed by the eigenvectors.
    c_opers_transformed: array_like, shape (n_ctrl, d, d)
        The control operators transformed into the eigenspace of the
        control Hamiltonian.
    n_opers_transformed: array_like, shape (n_nops, d, d)
        The noise operators transformed into the eigenspace of the
        control Hamiltonian.
    n_coeffs: array_like, shape (n_nops,)
        The sensitivities of the system to the noise operators given by
        *n_opers_transformed* at the given time step.
    n_coeffs_deriv: array_like, shape (n_nops, n_ctrl,)
        The derivatives of the noise susceptibilities by the control
        amplitudes. Defaults to None.
    phase_factor: array_like, shape (n_omega,)
        The phase factor :math:`e^{i\omega t_{g-1}}`.
    integral: array_like, shape (n_omega, d, d)
        The integral during the time step appearing in the regular
        control matrix.
    deriv_integral: array_like, shape (n_omega, d, d, d, d)
        An array to write the integral during the time step appearing in
        the derivative into.
    ctrlmat_step: array_like, shape (n_nops, d**2, n_omega)
        An array to write the control matrix during the time step into.
    ctrlmat_expr: ContractExpression
        An :class:`opt_einsum.contract.ContractExpression` to compute
        the control matrix during the time step.

    Returns
    -------
    ctrlmat_g: ndarray, shape (n_dt, n_nops, d**2, n_omega)
        The individual control matrices of all time steps
    ctrlmat_g_deriv: ndarray, shape (n_dt, n_nops, d**2, n_ctrl, n_omega)
        The corresponding derivative with respect to the control
        strength :math:`\frac{\partial\mathcal{B}_{\alpha j}^{(g)}(\omega)}`

    Notes
    -----
    The control matrix at each time step is evaluated according to

    .. math::

            \mathcal{B}_{\alpha j}^{(g)}(\omega) = s_\alpha^{(g)}\mathrm{tr}
            \left([\bar{B}_\alpha \circ I_1^{(g)}(\omega)] \bar{C}_j \right),

    where

    .. math::

        I_{1,nm}^{(g)}(\omega) = \frac{\exp(\mathrm{i}(\omega + \omega_n^{(g)}
                                            - \omega_m^{(g)}) \Delta t_g) - 1}
        {\mathrm{i}(\omega + \omega_n^{(g)} - \omega_m^{(g)})}

    The derivative of the control matrix with respect to the control
    strength at different time steps is calculated according to

    .. math::

        \frac{\partial \mathcal{B}_{\alpha j}^{(g)}(\omega)}
        {\partial u_h(t_{g^\prime})} =
        \mathrm{i}\delta_{gg^\prime} s_\alpha^{(g)} \mathrm{tr}
        \left( \bar{B}_{\alpha} \cdot \mathbb{M} \right)
        + \frac{\partial s_\alpha^{(g)}}{u_h (t_{g^\prime})} \text{tr}
        \left( (\overline{B}_{\alpha} \circ I_1^{(g)}{}(\omega)) \cdot
        \overline{C}_{j}) \right).

    We assume that the noise susceptibility :math:`s` only depends
    locally on the time i.e. :math:`\partial_{u(t_g)} s(t_{g^\prime})
    = \delta_{gg^\prime} \partial_{u(t_g)} s(t_g)`
    If denoting :math:`\Delta\omega_{ij} = \omega_i^{(g)} -
    \omega_j^{(g)}` the integral part is encapsulated in

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
    d = len(eigvecs)
    d2 = d**2
    n_ctrl = len(c_opers_transformed)
    n_nops = len(n_opers_transformed)
    n_omega = len(omega)

    deriv_integral = _derivative_integral(omega, eigvals, dt, out=deriv_integral)
    ctrlmat_step = ctrlmat_expr(phase_factor, basis_transformed, n_opers_transformed, integral,
                                out=ctrlmat_step)

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

    # Expand in basis transformed to eigenspace. Include phase factor and
    # factor 1j here to make use of optimized contraction order
    ctrlmat_step_deriv = oe.contract('o,jnk,ahokn->ajho', phase_factor, 1j*basis_transformed, M,
                                     optimize=[(1, 2), (0, 1)])

    if n_coeffs_deriv is not None:
        # equivalent contraction: 'ah,a,ako->akho', but this faster
        ctrlmat_step_deriv += ((n_coeffs_deriv / n_coeffs[:, None])[:, None, :, None]
                               * ctrlmat_step[:, :, None])

    return ctrlmat_step, ctrlmat_step_deriv


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
        n_coeffs_deriv: Optional[Sequence[Coefficients]] = None,
        intermediates: Optional[Dict[str, ndarray]] = None
) -> ndarray:
    r"""
    Calculate the derivative of the control matrix from scratch.

    Parameters
    ----------
    omega: array_like, shape (n_omega,)
        Frequencies, at which the pulse control matrix is to be
        evaluated.
    propagators: array_like, shape (n_dt+1, d, d)
        The propagators :math:`Q_g = P_g P_{g-1}\cdots P_0` as a (d, d)
        array with *d* the dimension of the Hilbert space.
    eigvals: array_like, shape (n_dt, d)
        Eigenvalue vectors for each time pulse segment *g* with the
        first axis counting the pulse segment, i.e.
        ``HD == array([D_0, D_1, ...])``.
    eigvecs: array_like, shape (n_dt, d, d)
        Eigenvector matrices for each time pulse segment *g* with the
        first axis counting the pulse segment, i.e.
        ``HV == array([V_0, V_1, ...])``.
    basis: Basis, shape (d**2, d, d)
        The basis elements, in which the pulse control matrix will be
        expanded.
    t: array_like, shape (n_dt+1), optional
        The absolute times of the different segments.
    dt: array_like, shape (n_dt)
        Sequence duration, i.e. for the :math:`g`-th pulse
        :math:`t_g - t_{g-1}`.
    n_opers: array_like, shape (n_nops, d, d)
        Noise operators :math:`B_\alpha`.
    n_coeffs: array_like, shape (n_nops, n_dt)
        The sensitivities of the system to the noise operators given by
        *n_opers* at the given time step.
    c_opers: array_like, shape (n_ctrl, d, d)
        Control operators :math:`H_k` with respect to which the
        derivative is computed.
    n_coeffs_deriv: array_like, shape (n_nops, n_ctrl, n_dt)
        The derivatives of the noise susceptibilities by the control
        amplitudes. Defaults to None.
    intermediates: Dict[str, ndarray], optional
        Optional dictionary containing intermediate results of the
        calculation of the control matrix.

    Returns
    -------
    ctrlmat_deriv: ndarray, shape (n_ctrl, n_omega, n_dt, n_nops, d**2)
        Partial derivatives of the total control matrix with respect to
        each control direction
        :math:`\frac{\partial\mathcal{B}_{\alpha k}(\omega)}{\partial
        u_h(t_{g'})}`.

    Notes
    -----
    The derivative of the control matrix is calculated according to

    .. math ::

        \frac{\partial\mathcal{B}_{\alpha k}(\omega)}{\partial u_h(t_{g'})} =
            \sum_{g=1}^G \mathrm{e}^{\mathrm{i}\omega t_{g-1}}\cdot\left(\sum_j
                \left[\frac{\partial\mathcal{B}_{\alpha j}^{(g)}(\omega)}
                    {\partial u_h(t_{g'})} \cdot \mathcal{Q}_{jk}^{(g-1)}
                +   \mathcal{B}_{\alpha j}^{(g)}(\omega)
                \cdot\frac{\partial \mathcal{Q}_{jk}^{(g-1)}}
                {\partial u_h(t_{g'})} \right] \right)

    See Also
    --------
    _liouville_derivative
    _control_matrix_at_timestep_derivative
    """
    d = eigvecs.shape[-1]
    n_dt = len(dt)
    n_ctrl = len(c_opers)
    n_nops = len(n_opers)
    n_omega = len(omega)

    # Precompute some transformations or grab from cache if possible
    basis_transformed = numeric._transform_by_unitary(eigvecs[:, None], basis[None],
                                                      out=np.empty((n_dt, d**2, d, d), complex))
    c_opers_transformed = numeric._transform_hamiltonian(eigvecs, c_opers).swapaxes(0, 1)
    if not intermediates:
        # None or empty
        n_opers_transformed = numeric._transform_hamiltonian(eigvecs, n_opers,
                                                             n_coeffs).swapaxes(0, 1)
        exp_buf, integral = np.empty((2, n_omega, d, d), dtype=complex)
    else:
        n_opers_transformed = intermediates['n_opers_transformed'].swapaxes(0, 1)

    propagators_liouville = superoperator.liouville_representation(propagators[:-1], basis)
    propagators_liouville_deriv = _liouville_derivative(dt, propagators, basis, eigvecs, eigvals,
                                                        c_opers_transformed)

    deriv_integral = np.empty((n_omega, d, d, d, d), dtype=complex)
    ctrlmat_deriv = np.empty((n_ctrl, n_omega, n_dt, n_nops, d**2), dtype=complex)
    ctrlmat_step = np.empty((n_dt, n_nops, d**2, n_omega), dtype=complex)
    # Optimized expression that is passed to control_matrix_at_timestep_derivative
    # in each iteration
    ctrlmat_expr = oe.contract_expression('o,icd,adc,odc->aio', (len(omega),), basis.shape,
                                          n_opers.shape, (len(omega), d, d),
                                          optimize=[(0, 3), (0, 1), (0, 1)])
    for g in range(n_dt):
        if not intermediates:
            integral = numeric._first_order_integral(omega, eigvals[g], dt[g], exp_buf, integral)
        else:
            integral = intermediates['first_order_integral'][g]

        n_coeff_deriv = n_coeffs_deriv if n_coeffs_deriv is None else n_coeffs_deriv[:, :, g]

        # ctrlmat_step is computed from scratch because the quantity in
        # the cache (from numeric.calculate_control_matrix_from_scratch)
        # contains the Liouville propagators already and it is expensive
        # to remove them by multiplying with the transpose.
        ctrlmat_step[g], ctrlmat_step_deriv = _control_matrix_at_timestep_derivative(
            omega, dt[g], eigvals[g], eigvecs[g], basis_transformed[g], c_opers_transformed[g],
            n_opers_transformed[g], n_coeffs[:, g], n_coeff_deriv, util.cexp(omega*t[g]), integral,
            deriv_integral, ctrlmat_step[g], ctrlmat_expr
        )
        # Phase factor already part of ctrlmat_step_deriv
        ctrlmat_deriv[:, :, g] = (ctrlmat_step_deriv.transpose(2, 3, 0, 1)
                                  @ propagators_liouville[g])

    # opt_einsum a lot faster here
    # Phase factor again already part of ctrlmat_step
    ctrlmat_deriv += oe.contract('tajo,thsjk->hosak',
                                 ctrlmat_step[1:], propagators_liouville_deriv)

    return ctrlmat_deriv


def calculate_filter_function_derivative(ctrlmat: ndarray, ctrlmat_deriv: ndarray) -> ndarray:
    r"""
    Compute the filter function derivative from the control matrix.

    Parameters
    ----------
    ctrlmat: array_like, shape (n_nops, d**2, n_omega)
        The control matrix.
    ctrlmat_deriv: array_like, shape (n_nops, d**2, n_t, n_ctrl, n_omega)
        The derivative of the control matrix.

    Returns
    -------
    filter_function_deriv: ndarray, shape (n_nops, n_dt, n_ctrl, n_omega)
        The regular filter functions' derivatives for variation in each control
        contribution
        :math:`\frac{\partial F_\alpha(\omega)}{\partial u_h(t_{g'})}`.

    Notes
    -----
    The filter function derivative is calculated according to

    .. math ::

        \frac{\partial F_\alpha(\omega)}{\partial u_h(t_{g'})}
                    = 2 \mathrm{Re}\left(\sum_k
                    \mathcal{B}_{\alpha k}^\ast(\omega)
                    \frac{\partial\mathcal{B}_{\alpha k}(\omega)}
                    {\partial u_h(t_{g'})} \right)
    """
    return 2*np.einsum('ako,hotak->atho', ctrlmat.conj(), ctrlmat_deriv).real


def infidelity_derivative(
        pulse: 'PulseSequence',
        spectrum: Coefficients,
        omega: Coefficients,
        control_identifiers: Optional[Sequence[str]] = None,
        n_oper_identifiers: Optional[Sequence[str]] = None,
        n_coeffs_deriv: Optional[Sequence[Coefficients]] = None
) -> ndarray:
    r"""Calculate the entanglement infidelity derivative of the
    ``PulseSequence`` *pulse*.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to calculate the
        infidelity.
    spectrum: array_like, shape ([[n_nops,] n_nops,] omega)
        The two-sided noise power spectral density in units of inverse
        frequencies as an array of shape (n_omega,), (n_nops, n_omega),
        or (n_nops, n_nops, n_omega). In the first case, the same
        spectrum is taken for all noise operators, in the second, it is
        assumed that there are no correlations between different noise
        sources and thus there is one spectrum for each noise operator.
        In the third and most general case, there may be a spectrum for
        each pair of noise operators corresponding to the correlations
        between them. n_nops is the number of noise operators considered
        and should be equal to ``len(n_oper_identifiers)``.
    omega: array_like, shape (n_omega,)
        The frequencies at which the integration is to be carried out.
    control_identifiers: Sequence[str], shape (n_ctrl,)
        Sequence of strings with the control identifiers to
        distinguish between control and drift Hamiltonian. The
        default is None, in which case the derivative is computed
        for all known non-noise operators.
    n_oper_identifiers: Sequence[str], shape (n_nops,)
        Sequence of strings with the noise identifiers for which to
        compute the derivative contribution. The default is None, in
        which case it is computed for all known noise operators.
    n_coeffs_deriv: array_like, shape (n_nops, n_ctrl, n_dt)
        The derivatives of the noise susceptibilities by the control
        amplitudes. The rows and columns should be in the same order
        as the corresponding identifiers above. Defaults to None, in
        which case the coefficients are assumed to be constant and
        hence their derivative vanishing.

        .. warning::

            Internally, control and noise terms of the Hamiltonian
            are stored alphanumerically sorted by their identifiers.
            If the noise and/or control identifiers above are not
            explicitly given, the rows and/or columns of this
            parameter need to be sorted in the same fashion.


    Raises
    ------
    ValueError
        If the provided noise spectral density does not fit expected
        shape.

    Returns
    -------
    infid_deriv: ndarray, shape (n_nops, n_dt, n_ctrl)
        Array with the derivative of the infidelity for each noise
        source taken for each control direction at each time step
        :math:`\frac{\partial I_e}{\partial u_h(t_{g'})}`. Sorted in
        the same fashion as `n_coeffs_deriv` or, if not given,
        alphanumerically by the identifiers.

    Notes
    -----
    The infidelity's derivative is given by

    .. math::

        \frac{\partial I_e}{\partial u_h(t_{g'})} = \frac{1}{d}
                                            \int_{-\infty}^\infty
                                            \frac{d\omega}{2\pi}
                                            S_\alpha(\omega)
                                            \frac{\partial F_\alpha(\omega)}
                                            {\partial u_h(t_{g'})}

    with :math:`S_{\alpha}(\omega)` the noise spectral density
    and :math:`F_{\alpha}(\omega)` the canonical filter function for
    noise source :math:`\alpha`.

    To convert to the average gate infidelity, use the following
    relation given by Horodecki et al. [Hor99]_ and Nielsen [Nie02]_:

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
    spectrum = util.parse_spectrum(spectrum, omega, range(len(pulse.n_opers)))
    filter_function_deriv = pulse.get_filter_function_derivative(omega,
                                                                 control_identifiers,
                                                                 n_oper_identifiers,
                                                                 n_coeffs_deriv)

    integrand = np.einsum('...o,...tho->...tho', spectrum, filter_function_deriv)
    infid_deriv = util.integrate(integrand, omega) / (2*np.pi*pulse.d)

    return infid_deriv

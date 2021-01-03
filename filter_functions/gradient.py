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
This module implements functions to calculate filter function and infidelity
derivatives.

Throughout this documentation the following notation will be used: n_dt denotes
the number of time steps, n_cops the number of all control operators, n_ctrl
the number of accessible control operators (if identifiers are provided,
otherwise n_ctrl=n_cops), n_nops the number of noise operators, n_omega the
number of frequency samples and d the dimension of the system.

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
from typing import Callable, Optional, Sequence, Union

import numpy as np
from numpy import ndarray
from scipy.integrate import trapz

from filter_functions.basis import Basis
from filter_functions.types import Coefficients, Operator
from filter_functions.util import cexp

__all__ = ['liouville_derivative', 'control_matrix_at_timestep_derivative',
           'calculate_derivative_of_control_matrix_from_scratch',
           'calculate_canonical_filter_function_derivative',
           'infidelity_derivative']


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
        A_mat[~mask] = (cexp(omega_diff[~mask]*dt[g]) - 1) \
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
        integral_Rg[~mask_Rg] = (cexp(EdE[~mask_Rg]*dt[g]) - 1) \
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
        a = dt[g]*cexp(EdE*dt[g]) / (1j * EdE) \
            + (cexp(EdE*dt[g]) - 1) / (EdE)**2
        integral_deriv[g, :, mask_deriv] = np.concatenate(([[a]*n_case]),
                                                          axis=0)
        # calculation if omega_diff != 0
        b1 = - (cexp(np.add.outer(EdE, dE[~mask_deriv])*dt[g]) - 1) \
            / (np.add.outer(EdE, dE[~mask_deriv])) / dE[~mask_deriv]
        b2 = + np.divide.outer(((cexp(EdE*dt[g]) - 1) / EdE), dE[~mask_deriv])
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


def calculate_derivative_of_control_matrix_from_scratch(
        omega: Coefficients,
        propagators: ndarray,
        Q_Liou: ndarray,
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
    exp_factor = cexp(np.multiply.outer(t, omega))
    summand1 = np.einsum('to,tajho,tjk->aktho', exp_factor[:-1],
                         ctrlmat_g_deriv, Q_Liou[:-1],
                         optimize=['einsum_path', (1, 2), (0, 1)])
    summand2 = np.einsum('to,tajo,thsjk->aksho', exp_factor[1:-1],
                         ctrlmat_g[1:], dL[:-1],
                         optimize=['einsum_path', (0, 1), (0, 1)])

    dR = summand1 + summand2
    return dR


def calculate_canonical_filter_function_derivative(
        R: ndarray,
        deriv_R: ndarray) -> ndarray:
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

    deriv_F = pulse.get_filter_function_derivative(
        omega=omega, contorl_identifier=control_identifiers, s_derivs=s_derivs)
    d = pulse.d
    integrand = np.einsum('ao,atho->atho', S_all, deriv_F)

    deriv_infid = trapz(integrand, omega) / (2*np.pi*d)

    return deriv_infid

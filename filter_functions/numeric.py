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
This module defines the functions to calculate everything related to filter
functions.

Functions
---------
:func:`calculate_control_matrix_from_atomic`
    Calculate the control matrix from those of atomic pulse sequences
:func:`calculate_control_matrix_from_scratch`
    Calculate the control matrix from scratch
:func:`calculate_control_matrix_periodic`
    Calculate the control matrix for a periodic Hamiltonian
:func:`calculate_cumulant_function`
    Calculate the cumulant function for a given ``PulseSequence`` object.
:func:`calculate_decay_amplitudes`
    Calculate the decay amplitudes, corresponding to first order terms of the
    Magnus expansion
:func:`calculate_filter_function`
    Calculate the filter function from the control matrix
:func:`calculate_pulse_correlation_filter_function`
    Calculate the pulse correlation filter function from the control matrix
:func:`diagonalize`
    Diagonalize a Hamiltonian
:func:`error_transfer_matrix`
    Calculate the error transfer matrix of a pulse up to a unitary rotation
:func:`infidelity`
    Function to compute the infidelity of a pulse defined by a
    ``PulseSequence`` instance for a given noise spectral density and
    frequencies
"""
from collections import deque
from itertools import accumulate, repeat
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import opt_einsum as oe
import sparse
from numpy import linalg as nla
from numpy import ndarray
from scipy import integrate
from scipy import linalg as sla

from . import util
from .basis import Basis
from .types import Coefficients, Operator

__all__ = ['calculate_control_matrix_from_atomic',
           'calculate_control_matrix_from_scratch',
           'calculate_cumulant_function',
           'calculate_decay_amplitudes',
           'calculate_filter_function',
           'calculate_pulse_correlation_filter_function', 'diagonalize',
           'error_transfer_matrix', 'infidelity']


def _first_order_integral(E: ndarray, HD: ndarray, dt: float,
                          int_buf: ndarray, exp_buf: ndarray) -> ndarray:
    r"""Calculate the integral appearing in first order Magnus expansion.

    The integral is evaluated as

    .. math::
        I_{mn}^{(g)}(\omega) = \frac
            {e^{i(\omega + \Omega_{mn}^{(g)})\Delta t_g} - 1}
            {i(\omega + \Omega_{mn}^{(g)})}

    """
    dE = np.subtract.outer(HD, HD)
    # iEdE_nm = 1j*(omega + omega_n - omega_m)
    int_buf.real = 0
    int_buf.imag = np.add.outer(E, dE, out=int_buf.imag)

    # Catch zero-division
    mask = (int_buf.imag != 0)
    # Use expm1 for better convergence with small arguments
    exp_buf = np.expm1(int_buf*dt, out=exp_buf, where=mask)
    int_buf = np.divide(exp_buf, int_buf, out=int_buf, where=mask)
    int_buf[~mask] = dt

    return int_buf


def _second_order_integral(E: ndarray, HD: ndarray, dt: float,
                           int_buf: ndarray, frc_bufs: Tuple[ndarray, ndarray],
                           dE_bufs: Tuple[ndarray, ndarray, ndarray],
                           exp_buf: ndarray,
                           msk_bufs: Tuple[ndarray, ndarray]) -> ndarray:
    r"""Calculate the nested integral of second order Magnus expansion.

    The integral is evaluated as

    .. math::
        I_{ijmn}^{(g)}(\omega) = \begin{cases}
            \frac{1}{\omega + \Omega_{mn}^{(g)}}\left(
                \frac{e^{i(\Omega_{ij}^{(g)} - \omega)\Delta t_g} - 1}
                     {\Omega_{ij}^{(g)} - \omega} -
                \frac{e^{i(\Omega_{ij}^{(g)} + \Omega_{mn}^{(g)})\Delta t_g}
                      - 1}{\Omega_{ij}^{(g)} + \Omega_{mn}^{(g)}}
            \right) &\quad\mathrm{if}\quad \omega + \Omega_{mn}^{(g)}\neq 0, \\
            \frac{1}{\Omega_{ij}^{(g)} - \omega}\left(
                \frac{e^{i(\Omega_{ij}^{(g)} - \omega)\Delta t_g} - 1}
                     {\Omega_{ij}^{(g)} - \omega} -
                i\Delta t_ge^{i(\Omega_{ij}^{(g)} - \omega)\Delta t_g}
            \right) &\quad\mathrm{if}\quad\omega + \Omega_{mn}^{(g)} = 0 \wedge
                                      \Omega_{ij}^{(g)} - \omega\neq 0, \\
            \Delta t_g^2/2 &\quad\mathrm{if}\quad
                \omega + \Omega_{mn}^{(g)} = 0 \wedge
                \Omega_{ij}^{(g)} - \omega = 0.
        \end{cases}

    with :math:`\Omega_{mn}^{(g)} = \omega_m^{(g)} - \omega_n^{(g)}`.

    Test
    ----
    Set t_{g-1} = 1/np.sqrt(2)

    >>> from scipy import integrate
    >>> from tests import testutil
    >>> d, t0 = 2, 1/np.sqrt(2)
    >>> (t1, t2), dt = np.tile(np.linspace(0, 1, 1001), (2, 1)) + t0, 1
    >>> HD = np.random.randn()*np.array([-1, 1])
    >>> dE = np.subtract.outer(HD, HD)
    >>> E = np.linspace(-1, 1, 101)
    >>> ex = 1j*(np.multiply.outer(dE, t2 - t0) +
    ...          np.expand_dims(np.multiply.outer(E, t2), (1, 2)))
    >>> I1 = integrate.cumtrapz(np.exp(ex), t2, initial=0)
    >>> ex = 1j*(np.multiply.outer(dE, t1 - t0) -
    ...          np.expand_dims(np.multiply.outer(E, t1), (1, 2)))
    >>> integrand = (np.expand_dims(np.exp(ex), (3, 4)) *
    ...              np.expand_dims(I1, (1, 2)))
    >>> I2 = integrate.trapz(integrand, t1)
    >>> dE_bufs = (np.empty((d, d, d, d), dtype=float),
    ...            np.empty((len(E), d, d), dtype=float),
    ...            np.empty((len(E), d, d), dtype=float))
    >>> exp_buf = np.empty((len(E), d, d), dtype=complex)
    >>> frc_bufs = (np.empty((len(E), d, d), dtype=complex),
    ...             np.empty((d, d, d, d), dtype=complex))
    >>> int_buf = np.empty((len(E), d, d, d, d), dtype=complex)
    >>> msk_bufs = np.empty((2, len(E), d, d, d, d), dtype=bool)
    >>> I3 = _second_order_integral(E, HD, dt, int_buf, frc_bufs, dE_bufs,
    ...                             exp_buf, msk_bufs)
    >>> np.allclose(I2, I3)

    """
    # frc_buf1 has shape (len(E), *dE.shape), frc_buf2 has shape dE.shape*2
    frc_buf1, frc_buf2 = frc_bufs
    dEdE, EdE, dEE = dE_bufs
    mask_nEdE_dEE, mask_nEdE_ndEE = msk_bufs

    dE = np.subtract.outer(HD, HD)
    dEdE = np.add.outer(dE, dE, out=dEdE)
    EdE = np.add.outer(E, dE, out=EdE)
    dEE = np.subtract.outer(-E, -dE, out=dEE)
    mask_dEdE = np.not_equal(dEdE, 0)
    mask_EdE = np.not_equal(EdE, 0)
    mask_dEE = np.not_equal(dEE, 0)
    mask_nEdE_dEE = np.logical_and(~mask_EdE[:, None, None],
                                   mask_dEE[..., None, None],
                                   out=mask_nEdE_dEE)
    mask_nEdE_ndEE = np.logical_and(~mask_EdE[:, None, None],
                                    ~mask_dEE[..., None, None],
                                    out=mask_nEdE_ndEE)
    mask_EdE_dEE = np.broadcast_to(mask_EdE[:, None, None], int_buf.shape)

    # First term in the brackets
    exp_buf = np.expm1(1j*dEE*dt, out=exp_buf, where=mask_dEE)
    frc_buf1 = np.divide(exp_buf, dEE, out=frc_buf1, where=mask_dEE)
    frc_buf1[~mask_dEE] = 1j*dt

    # Second term in the brackets
    frc_buf2 = np.expm1(1j*dEdE*dt, out=frc_buf2, where=mask_dEdE)
    frc_buf2 = np.divide(frc_buf2, dEdE, out=frc_buf2, where=mask_dEdE)
    frc_buf2[~mask_dEdE] = 1j*dt

    # Broadcast to full (len(E), d, d, d, d) result
    int_buf = np.subtract(frc_buf1[..., None, None], frc_buf2[None, ...],
                          out=int_buf, where=mask_EdE_dEE)

    # Prefactor
    int_buf = np.divide(int_buf, EdE[:, None, None],
                        out=int_buf, where=mask_EdE_dEE)

    # Case where omega + Omega_ij = 0, omega - Omega_mn != 0
    exp_buf = np.add(exp_buf, 1, out=exp_buf, where=mask_dEE)
    exp_buf = np.multiply(exp_buf, dt, out=exp_buf, where=mask_dEE)
    frc_buf1.real = np.add(frc_buf1.real, exp_buf.imag,
                           out=frc_buf1.real, where=mask_dEE)
    frc_buf1.imag = np.subtract(frc_buf1.imag, exp_buf.real,
                                out=frc_buf1.imag, where=mask_dEE)
    frc_buf1 = np.divide(frc_buf1, dEE, out=frc_buf1, where=mask_dEE)

    int_buf[mask_nEdE_dEE] = np.broadcast_to(frc_buf1[..., None, None],
                                             int_buf.shape)[mask_nEdE_dEE]

    # Case where omega + Omega_ij = 0, omega - Omega_mn = 0
    int_buf[mask_nEdE_ndEE] = dt**2 / 2

    return int_buf


@util.parse_optional_parameters({'which': ('total', 'correlations')})
def calculate_control_matrix_from_atomic(
        phases: ndarray, R_g: ndarray, Q_liouville: ndarray,
        show_progressbar: Optional[bool] = None,
        which: str = 'total') -> ndarray:
    r"""
    Calculate the control matrix from the control matrices of atomic segments.

    Parameters
    ----------
    phases: array_like, shape (n_dt, n_omega)
        The phase factors for :math:`l\in\{0, 1, \dots, n-1\}`.
    R_g: array_like, shape (n_dt, n_nops, d**2, n_omega)
        The pulse control matrices for :math:`l\in\{1, 2, \dots, n\}`.
    Q_liouville: array_like, shape (n_dt, n_nops, d**2, d**2)
        The transfer matrices of the cumulative propagators for
        :math:`l\in\{0, 1, \dots, n-1\}`.
    show_progressbar: bool, optional
        Show a progress bar for the calculation.
    which: str, ('total', 'correlations')
        Compute the total control matrix (the sum of all time steps) or the
        correlation control matrix (first axis holds each pulses' contribution)

    Returns
    -------
    R: ndarray, shape ([n_pls,] n_nops, d**2, n_omega)
        The control matrix :math:`\mathcal{R}(\omega)`.

    Notes
    -----
    The control matrix is calculated by evaluating the sum

    .. math::

        \mathcal{R}(\omega) = \sum_{l=1}^n e^{i\omega t_{l-1}}
            \mathcal{R}^{(l)}(\omega)\mathcal{Q}^{(l-1)}.

    See Also
    --------
    calculate_control_matrix_from_scratch: Control matrix from scratch.
    liouville_representation: Liouville representation for a given basis.
    """
    n = len(R_g)
    # Set up a reusable contraction expression. In some cases it is faster to
    # also contract the time dimension in the same expression instead of
    # looping over it, but we don't distinguish here for readability.
    R_expr = oe.contract_expression('ijo,jk->iko',
                                    R_g.shape[1:], Q_liouville.shape[1:])

    # Allocate memory
    if which == 'total':
        R = np.zeros(R_g.shape[1:], dtype=complex)
        for g in util.progressbar_range(n, show_progressbar=show_progressbar,
                                        desc='Calculating control matrix'):
            R += R_expr(phases[g]*R_g[g], Q_liouville[g])
    else:
        # which == 'correlations'
        R = np.zeros_like(R_g)
        for g in util.progressbar_range(n, show_progressbar=show_progressbar,
                                        desc='Calculating control matrix'):
            R[g] = R_expr(phases[g]*R_g[g], Q_liouville[g])

    return R


def calculate_control_matrix_from_scratch(
        HD: ndarray,
        HV: ndarray,
        Q: ndarray,
        omega: Coefficients,
        basis: Basis,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        dt: Coefficients,
        t: Optional[Coefficients] = None,
        show_progressbar: Optional[bool] = False,
        out: Optional = None) -> ndarray:
    r"""
    Calculate the control matrix from scratch, i.e. without knowledge of the
    control matrices of more atomic pulse sequences.

    Parameters
    ----------
    HD: array_like, shape (n_dt, d)
        Eigenvalue vectors for each time pulse segment *l* with the first
        axis counting the pulse segment, i.e.
        ``HD == array([D_0, D_1, ...])``.
    HV: array_like, shape (n_dt, d, d)
        Eigenvector matrices for each time pulse segment *l* with the first
        axis counting the pulse segment, i.e.
        ``HV == array([V_0, V_1, ...])``.
    Q: array_like, shape (n_dt+1, d, d)
        The propagators :math:`Q_l = P_l P_{l-1}\cdots P_0` as a (d, d) array
        with *d* the dimension of the Hilbert space.
    omega: array_like, shape (n_omega,)
        Frequencies at which the pulse control matrix is to be evaluated.
    basis: Basis, shape (d**2, d, d)
        The basis elements in which the pulse control matrix will be expanded.
    n_opers: array_like, shape (n_nops, d, d)
        Noise operators :math:`B_\alpha`.
    n_coeffs: array_like, shape (n_nops, n_dt)
        The sensitivities of the system to the noise operators given by
        *n_opers* at the given time step.
    dt: array_like, shape (n_dt)
        Sequence duration, i.e. for the :math:`l`-th pulse
        :math:`t_l - t_{l-1}`.
    t: array_like, shape (n_dt+1), optional
        The absolute times of the different segments. Can also be computed from
        *dt*.
    show_progressbar: bool, optional
        Show a progress bar for the calculation.
    out: ndarray, optional
        A location into which the result is stored. See :func:`numpy.ufunc`.

    Returns
    -------
    R: ndarray, shape (n_nops, d**2, n_omega)
        The control matrix :math:`\mathcal{R}(\omega)`

    Notes
    -----
    The control matrix is calculated according to

    .. math::

        \mathcal{R}_{\alpha k}(\omega) = \sum_{l=1}^n e^{i\omega t_{l-1}}
            s_\alpha^{(l)}\mathrm{tr}\left(
                [\bar{B}_\alpha^{(l)}\circ I(\omega)] \bar{C}_k^{(l)}
            \right)

    where

    .. math::

        I^{(l)}_{nm}(\omega) &= \int_0^{t_l - t_{l-1}}\mathrm{d}t\,
                                e^{i(\omega+\omega_n-\omega_m)t} \\
                             &= \frac{e^{i(\omega+\omega_n-\omega_m)
                                (t_l - t_{l-1})} - 1}
                                {i(\omega+\omega_n-\omega_m)}, \\
        \bar{B}_\alpha^{(l)} &= V^{(l)\dagger} B_\alpha V^{(l)}, \\
        \bar{C}_k^{(l)} &= V^{(l)\dagger} Q_{l-1} C_k Q_{l-1}^\dagger V^{(l)},

    and :math:`V^{(l)}` is the matrix of eigenvectors that diagonalizes
    :math:`\tilde{\mathcal{H}}_n^{(l)}`, :math:`B_\alpha` the :math:`\alpha`-th
    noise operator :math:`s_\alpha^{(l)}` the noise sensitivity during interval
    :math:`l`, and :math:`C_k` the :math:`k`-th basis element.

    See Also
    --------
    calculate_control_matrix_from_atomic: Control matrix from concatenation.
    """
    if t is None:
        t = np.concatenate(([0], np.asarray(dt).cumsum()))

    d = HV.shape[-1]
    # We're lazy
    E = omega
    n_coeffs = np.asarray(n_coeffs)

    # Precompute noise opers transformed to eigenbasis of each pulse
    # segment and Q^\dagger @ HV
    QdagV = Q[:-1].transpose(0, 2, 1).conj() @ HV
    B = np.empty((len(n_opers), len(dt), d, d), dtype=complex)
    for j, n_oper in enumerate(n_opers):
        B[j] = HV.conj().transpose(0, 2, 1) @ n_oper @ HV

    # Allocate result and buffers for intermediate arrays
    if out is None:
        out = np.zeros((len(n_opers), len(basis), len(E)), dtype=complex)

    exp_buf = np.empty((len(E), d, d), dtype=complex)
    int_buf = np.empty((len(E), d, d), dtype=complex)
    R_path = ['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)]

    for l in util.progressbar_range(len(dt), show_progressbar=show_progressbar,
                                    desc='Calculating control matrix'):

        int_buf = _first_order_integral(E, HD[l], dt[l], int_buf, exp_buf)
        # Faster for d = 2 to also contract over the time dimension instead of
        # loop, but for readability we don't distinguish.
        out += np.einsum('o,j,jmn,omn,knm->jko',
                         util.cexp(E*t[l]), n_coeffs[:, l], B[:, l], int_buf,
                         QdagV[l].conj().T @ basis @ QdagV[l],
                         optimize=R_path)

    return out


def calculate_control_matrix_periodic(phases: ndarray, R: ndarray,
                                      Q_liouville: ndarray,
                                      repeats: int) -> ndarray:
    r"""
    Calculate the control matrix of a periodic pulse given the phase factors,
    control matrix and transfer matrix of the total propagator, Q_liouville, of
    the atomic pulse.

    Parameters
    ----------
    phases: ndarray, shape (n_omega,)
        The phase factors :math:`e^{i\omega T}` of the atomic pulse.
    R: ndarray, shape (n_nops, d**2, n_omega)
        The control matrix :math:`\mathcal{R}^{(1)}(\omega)` of the atomic
        pulse.
    Q_liouville: ndarray, shape (d**2, d**2)
        The transfer matrix :math:`\mathcal{Q}^{(1)}` of the atomic pulse.
    repeats: int
        The number of repetitions.

    Returns
    -------
    R: ndarray, shape (n_nops, d**2, n_omega)
        The control matrix :math:`\mathcal{R}(\omega)` of the repeated pulse.

    Notes
    -----
    The control matrix is computed as

    .. math::

        \mathcal{R}(\omega) &= \mathcal{R}^{(1)}(\omega)\sum_{g=0}^{G-1}\left(
                               e^{i\omega T}\right)^g \\
                            &= \mathcal{R}^{(1)}(\omega)\bigl(
                               \mathbb{I} - e^{i\omega T}\mathcal{Q}^{(1)}
                               \bigr)^{-1}\bigl(\mathbb{I} - \bigl(
                               e^{i\omega T}\mathcal{Q}^{(1)}\bigr)^G\bigr).

    with :math:`G` the number of repetitions.
    """
    # Compute the finite geometric series \sum_{g=0}^{G-1} T^g. First check if
    # inv(I - T) is 'good', i.e. if inv(I - T) @ (I - T) == I, since NumPy will
    # compute the inverse in any case. For those frequencies where the inverse
    # is well-behaved, evaluate the sum as a Neumann series and for the rest
    # evaluate it explicitly.
    eye = np.eye(Q_liouville.shape[0])
    T = np.multiply.outer(phases, Q_liouville)

    # Mask the invertible frequencies. The chosen atol is somewhat empiric.
    M = eye - T
    M_inv = nla.inv(M)
    good_inverse = np.isclose(M_inv @ M, eye, atol=1e-10, rtol=0).all((1, 2))

    # Allocate memory
    S = np.empty((*phases.shape, *Q_liouville.shape), dtype=complex)
    # Evaluate the sum for invertible frequencies
    S[good_inverse] = (M_inv[good_inverse] @
                       (eye - nla.matrix_power(T[good_inverse], repeats)))

    # Evaluate the sum for non-invertible frequencies
    if (~good_inverse).any():
        S[~good_inverse] = eye +\
            sum(accumulate(repeat(T[~good_inverse], repeats-1), np.matmul))

    # Multiply with R_at to get the final control matrix
    R_tot = (R.transpose(2, 0, 1) @ S).transpose(1, 2, 0)
    return R_tot


@util.parse_optional_parameters({'which': ('total', 'correlations')})
def calculate_cumulant_function(
        pulse: 'PulseSequence',
        S: ndarray,
        omega: Coefficients,
        n_oper_identifiers: Optional[Sequence[str]] = None,
        which: Optional[str] = 'total',
        show_progressbar: Optional[bool] = False,
        memory_parsimonious: Optional[bool] = False) -> ndarray:
    r"""Calculate the cumulant function :math:`K(\tau)`.

    The error transfer matrix is obtained from the cumulant function by
    exponentiation, :math:`\langle\tilde{\mathcal{U}}\rangle = \exp K(\tau)`.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to compute the cumulant
        function.
    S: array_like, shape ([[n_nops,] n_nops,] n_omega)
        The two-sided noise power spectral density in units of inverse
        frequencies as an array of shape (n_omega,), (n_nops, n_omega), or
        (n_nops, n_nops, n_omega). In the first case, the same spectrum is
        taken for all noise operators, in the second, it is assumed that there
        are no correlations between different noise sources and thus there is
        one spectrum for each noise operator. In the third and most general
        case, there may be a spectrum for each pair of noise operators
        corresponding to the correlations between them. n_nops is the number of
        noise operators considered and should be equal to
        ``len(n_oper_identifiers)``.
    omega: array_like,
        The frequencies. Note that the frequencies are assumed to be symmetric
        about zero.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which to evaluate the
        cumulant function. The default is all.
    which: str, optional
        Which decay amplitudes should be calculated, may be either 'total'
        (default) or 'correlations'. See :func:`infidelity` and
        :ref:`Notes <notes>`.
    show_progressbar: bool, optional
        Show a progress bar for the calculation of the control matrix.
    memory_parsimonious: bool, optional
        Trade memory footprint for performance. See
        :func:`~numeric.calculate_decay_amplitudes`. The default is ``False``.

    Returns
    -------
    K: ndarray, shape ([[n_pls, n_pls,] n_nops,] n_nops, d**2, d**2)
        The cumulant function. The individual noise operator contributions
        chosen by ``n_oper_identifiers`` are on the third to last axis / axes,
        depending on whether the noise is cross-correlated or not. If
        ``which == 'correlations'``, the first two axes correspond to the
        contributions of the pulses in the sequence.

    .. _notes:

    Notes
    -----
    The cumulant function is given by

    .. math::

        K_{\alpha\beta,ij}(\tau) = -\frac{1}{2} \sum_{kl}\biggl(
            &\Delta_{\alpha\beta,kl}\left(
                T_{klji} - T_{lkji} - T_{klij} + T_{lkij}
            \right) \\ + &\Gamma_{\alpha\beta,kl}\left(
                T_{klji} - T_{kjli} - T_{kilj} + T_{kijl}
            \right)
        \biggr)

    Here, :math:`T_{ijkl} = \mathrm{tr}(C_i C_j C_k C_l)` is a trivial function
    of the basis elements :math:`C_i`, and :math:`\Gamma_{\alpha\beta,kl}` and
    :math:`\Delta_{\alpha\beta,kl}` are the decay amplitudes and frequency
    shifts which correspond to first and second order in the Magnus expansion,
    respectively. Since the latter induce coherent errors, we can approximately
    neglect them if we assume that the pulse has been experimentally
    calibrated.

    For a single qubit and represented in the Pauli basis, the above reduces to

    .. math::

        K_{\alpha\beta,ij}(\tau) = \begin{cases}
            - \sum_{k\neq i}\Gamma_{\alpha\beta,kk}
                &\quad\mathrm{if}\: i = j,   \\
            - \Delta_{\alpha\beta,ij} + \Delta_{\alpha\beta,ji}
            + \Gamma_{\alpha\beta,ij}
                &\quad\mathrm{if}\: i\neq j,
        \end{cases}

    for :math:`i\in\{1,2,3\}` and :math:`K_{0j} = K_{i0} = 0`.

    Lastly, the pulse correlation cumulant function resolves correlations in
    the cumulant function of a sequence of pulses :math:`g = 1,\dotsc,G` such
    that the following holds:

    .. math::

        K_{\alpha\beta,ij}(\tau) = \sum_{g,g'=1}^G
             K_{\alpha\beta,ij}^{(gg')}(\tau).

    See Also
    --------
    calculate_decay_amplitudes: Calculate the :math:`\Gamma_{\alpha\beta,kl}`
    error_transfer_matrix: Calculate the error transfer matrix :math:`\exp K`.
    infidelity: Calculate only infidelity of a pulse.
    pulse_sequence.concatenate: Concatenate ``PulseSequence`` objects.
    calculate_pulse_correlation_filter_function

    """
    N, d = pulse.basis.shape[:2]
    Gamma = calculate_decay_amplitudes(pulse, S, omega, n_oper_identifiers,
                                       which, show_progressbar,
                                       memory_parsimonious)

    if d == 2 and pulse.basis.btype in ('Pauli', 'GGM'):
        # Single qubit case. Can use simplified expression
        K = np.zeros_like(Gamma)
        diag_mask = np.eye(N, dtype=bool)

        # Offdiagonal terms
        K[..., ~diag_mask] = Gamma[..., ~diag_mask]

        # Diagonal terms K_ii given by sum over diagonal of Gamma excluding
        # Gamma_ii. Since the Pauli basis is traceless, K_00 is zero, therefore
        # start at K_11.
        diag_items = deque((True, False, True, True))
        for i in range(1, N):
            K[..., i, i] = - Gamma[..., diag_items, diag_items].sum(axis=-1)
            # shift the item not summed over by one
            diag_items.rotate()
    else:
        # Multi qubit case. Use general expression.
        T = pulse.basis.four_element_traces
        K = - (oe.contract('...kl,klji->...ij', Gamma, T, backend='sparse') -
               oe.contract('...kl,kjli->...ij', Gamma, T, backend='sparse') -
               oe.contract('...kl,kilj->...ij', Gamma, T, backend='sparse') +
               oe.contract('...kl,kijl->...ij', Gamma, T, backend='sparse'))/2

    return K.real


@util.parse_optional_parameters({'which': ('total', 'correlations')})
def calculate_decay_amplitudes(
        pulse: 'PulseSequence',
        S: ndarray,
        omega: Coefficients,
        n_oper_identifiers: Optional[Sequence[str]] = None,
        which: str = 'total',
        show_progressbar: Optional[bool] = False,
        memory_parsimonious: Optional[bool] = False) -> ndarray:
    r"""
    Get the decay amplitudes :math:`\Gamma_{\alpha\beta, kl}` for noise sources
    :math:`\alpha,\beta` and basis elements :math:`k,l`.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to compute the decay
        amplitudes.
    S: array_like, shape ([[n_nops,] n_nops,] n_omega)
        The two-sided noise power spectral density. If 1-d, the same spectrum
        is used for all noise operators. If 2-d, one (self-) spectrum for each
        noise operator is expected. If 3-d, should be a matrix of cross-spectra
        such that ``S[i, j] == S[j, i].conj()``.
    omega: array_like,
        The frequencies. Note that the frequencies are assumed to be symmetric
        about zero.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which to calculate the decay
        amplitudes. The default is all.
    which: str, optional
        Which decay amplitudes should be calculated, may be either 'total'
        (default) or 'correlations'. See :func:`infidelity` and
        :ref:`Notes <notes>`.
    show_progressbar: bool, optional
        Show a progress bar for the calculation.
    memory_parsimonious: bool, optional
        For large dimensions, the integrand

        .. math::

            \mathcal{R}^\ast_{\alpha k}(\omega)S_{\alpha\beta}(\omega)
            \mathcal{R}_{\beta l}(\omega)

        can consume quite a large amount of memory if set up for all
        :math:`\alpha,\beta,k,l` at once. If ``True``, it is only set up and
        integrated for a single :math:`k` at a time and looped over. This is
        slower but requires much less memory. The default is ``False``.

    Raises
    ------
    ValueError
        If S has incompatible shape.

    Returns
    -------
    Gamma: ndarray, shape ([[n_pls, n_pls,] n_nops,] n_nops, d**2, d**2)
        The decay amplitudes.

    .. _notes:

    Notes
    -----
    The total decay amplitudes are given by

    .. math::

        \Gamma_{\alpha\beta, kl} = \int
            \frac{\mathrm{d}\omega}{2\pi}\mathcal{R}^\ast_{\alpha k}(\omega)
            S_{\alpha\beta}(\omega)\mathcal{R}_{\beta l}(\omega).

    If pulse correlations are taken into account, they are given by

    .. math::

        \Gamma_{\alpha\beta, kl}^{(gg')} = \int\frac{\mathrm{d}\omega}{2\pi}
            S_{\alpha\beta}(\omega)F_{\alpha\beta, kl}^{(gg')}(\omega).

    See Also
    --------
    infidelity: Compute the infidelity directly.
    pulse_sequence.concatenate: Concatenate ``PulseSequence`` objects.
    calculate_pulse_correlation_filter_function
    """
    # TODO: Replace infidelity() by this?
    # Noise operator indices
    idx = util.get_indices_from_identifiers(pulse, n_oper_identifiers, 'noise')
    if which == 'total':
        # Faster to use filter function instead of control matrix
        if pulse.is_cached('F_kl'):
            R = None
            F = pulse.get_filter_function(omega, which='generalized')
        else:
            R = pulse.get_control_matrix(omega, show_progressbar)
            F = None
    else:
        # which == 'correlations'
        if pulse.is_cached('omega'):
            if not np.array_equal(pulse.omega, omega):
                raise ValueError('Pulse correlation decay amplitudes ' +
                                 'requested but omega not equal to ' +
                                 'cached frequencies.')

        if pulse.is_cached('F_pc_kl'):
            R = None
            F = pulse.get_pulse_correlation_filter_function(
                    which='generalized')
        else:
            R = pulse.get_pulse_correlation_control_matrix()
            F = None

    if not memory_parsimonious:
        integrand = _get_integrand(S, omega, idx, which, 'generalized', R=R,
                                   F=F)
        Gamma = integrate.trapz(integrand, omega, axis=-1)/(2*np.pi)
        return Gamma.real

    # Conserve memory by looping. Let _get_integrand determine the shape
    if R is not None:
        integrand = _get_integrand(S, omega, idx, which, 'generalized',
                                   R=[R[..., 0:1, :], R], F=F)
        n_kl = R.shape[-2]
    else:
        integrand = _get_integrand(S, omega, idx, which, 'generalized',
                                   R=R, F=F[..., 0:1, :, :])
        n_kl = F.shape[-2]

    Gamma = np.zeros(integrand.shape[:-3] + (n_kl,)*2,
                     dtype=integrand.dtype)
    Gamma[..., 0:1, :] = integrate.trapz(integrand, omega, axis=-1)/(2*np.pi)

    for k in util.progressbar_range(1, n_kl, show_progressbar=show_progressbar,
                                    desc='Integrating'):
        if R is not None:
            integrand = _get_integrand(S, omega, idx, which, 'generalized',
                                       R=[R[..., k:k+1, :], R], F=F)
        else:
            integrand = _get_integrand(S, omega, idx, which, 'generalized',
                                       R=R, F=F[..., k:k+1, :, :])

        Gamma[..., k:k+1, :] = integrate.trapz(integrand, omega,
                                               axis=-1)/(2*np.pi)

    return Gamma.real


@util.parse_which_FF_parameter
def calculate_filter_function(R: ndarray, which: str = 'fidelity') -> ndarray:
    r"""Compute the filter function from the control matrix.

    Parameters
    ----------
    R: array_like, shape (n_nops, d**2, n_omega)
        The control matrix.
    which : str, optional
        Which filter function to return. Either 'fidelity' (default) or
        'generalized' (see :ref:`Notes <notes>`).

    Returns
    -------
    F: ndarray, shape (n_nops, n_nops, [d**2, d**2,] n_omega)
        The filter functions for each noise operator correlation. The diagonal
        corresponds to the filter functions for uncorrelated noise sources.

    .. _notes:

    Notes
    -----
    The generalized filter function is given by

    .. math::

        F_{\alpha\beta,kl}(\omega) = \mathcal{R}_{\alpha k}^\ast(\omega)
                                     \mathcal{R}_{\beta l}(\omega),

    where :math:`\alpha,\beta` are indices counting the noise operators
    :math:`B_\alpha` and :math:`k,l` indices counting the basis elements
    :math:`C_k`.

    The fidelity filter function is obtained by tracing over the basis indices:

    .. math::

        F_{\alpha\beta}(\omega) = \sum_{k} F_{\alpha\beta,kk}(\omega).

    See Also
    --------
    calculate_control_matrix_from_scratch: Control matrix from scratch.
    calculate_control_matrix_from_atomic: Control matrix from concatenation.
    calculate_pulse_correlation_filter_function: Pulse correlations.
    """
    if which == 'fidelity':
        return np.einsum('ako,bko->abo', R.conj(), R)
    else:
        # which == 'generalized'
        return np.einsum('ako,blo->abklo', R.conj(), R)


def calculate_filter_function_second_order(
        HD: ndarray,
        HV: ndarray,
        Q: ndarray,
        omega: Coefficients,
        basis: Basis,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        dt: Coefficients,
        memory_parsimonious: Optional[bool] = False,
        show_progressbar: Optional[bool] = False) -> ndarray:

    d = HV.shape[-1]
    # We're lazy
    E = omega
    n_coeffs = np.asarray(n_coeffs)

    # Precompute noise opers transformed to eigenbasis of each pulse
    # segment and Q^\dagger @ HV
    QdagV = Q[:-1].transpose(0, 2, 1).conj() @ HV
    B = np.empty((len(n_opers), len(dt), d, d), dtype=complex)
    for j, n_oper in enumerate(n_opers):
        B[j] = HV.conj().transpose(0, 2, 1) @ n_oper @ HV

    # Allocate result and buffers for intermediate arrays
    dE_bufs = (np.empty((d, d, d, d), dtype=float),
               np.empty((len(E), d, d), dtype=float),
               np.empty((len(E), d, d), dtype=float))
    exp_buf = np.empty((len(E), d, d), dtype=complex)
    frc_bufs = (np.empty((len(E), d, d), dtype=complex),
                np.empty((d, d, d, d), dtype=complex))
    int_buf = np.empty((len(E), d, d, d, d), dtype=complex)
    msk_bufs = np.empty((2, len(E), d, d, d, d), dtype=bool)
    G = np.zeros((len(n_coeffs), d**2, len(E)), dtype=complex)
    G_cumulative = np.zeros((len(n_coeffs), d**2, len(E)), dtype=complex)
    GG = np.empty((len(n_coeffs), len(n_coeffs), d**2, d**2, len(E)),
                  dtype=complex)
    result = np.zeros((len(n_coeffs), len(n_coeffs), d**2, d**2, len(E)),
                      dtype=complex)

    for g in util.progressbar_range(len(dt), show_progressbar=show_progressbar,
                                    desc='Calculating second order Magnus'):

        int_buf = _second_order_integral(E, HD[g], dt[g], int_buf, frc_bufs,
                                         dE_bufs, exp_buf, msk_bufs)

        C = QdagV[g].conj().T @ basis @ QdagV[g]  # shared
        BC = np.einsum('akl,ilk->aikl', B[:, g], C)
        GG = np.einsum('oijmn,akij,blmn->abklo', int_buf, BC, BC,
                       optimize=['einsum_path', (0, 1), (0, 1)], out=GG)

        result += GG  # last interval
        if g > 0:
            # Add G^(g-1) to cumulative sum
            G_cumulative += G

            # Compute G^(g)
            G = calculate_control_matrix_from_scratch(
                HD[g:g+1], HV[g:g+1], Q[g:g+2], omega, basis, n_opers,
                n_coeffs[:, g:g+1], dt[g:g+1], t=None, show_progressbar=False,
                out=G
            )  # shared

            GG = np.einsum('ako,blo->abklo', G.conj(), G_cumulative, out=GG)
            result += GG  # all intervals up to last

    return result


@util.parse_which_FF_parameter
def calculate_pulse_correlation_filter_function(
        R: ndarray, which: str = 'fidelity') -> ndarray:
    r"""Compute pulse correlation filter function from the control matrix.

    Parameters
    ----------
    R: array_like, shape (n_pulses, n_nops, d**2, n_omega)
        The control matrix.
    which : str, optional
        Which filter function to return. Either 'fidelity' (default) or
        'generalized' (see :ref:`Notes <notes>`).

    Returns
    -------
    F_pc: ndarray, shape (n_pls, n_pls, n_nops, n_nops, [d**2, d**2,] n_omega)
        The pulse correlation filter functions for each pulse and noise
        operator correlations. The first two axes hold the pulse correlations,
        the second two the noise correlations.

    .. _notes:

    Notes
    -----
    The generalized pulse correlation filter function is given by

    .. math::

        F_{\alpha\beta,kl}^{(gg')}(\omega) = \bigl[
            \mathcal{Q}^{(g'-1)\dagger}\mathcal{R}^{(g')\dagger}(\omega)
        \bigr]_{k\alpha} \bigl[
            \mathcal{R}^{(g)}(\omega)\mathcal{Q}^{(g-1)}
        \bigr]_{\beta l} e^{i\omega(t_{g-1} - t_{g'-1})},

    with :math:`\mathcal{R}^{(g)}` the control matrix of the :math:`g`-th
    pulse. The fidelity pulse correlation function is obtained by tracing out
    the basis indices,

    .. math::

        F_{\alpha\beta}^{(gg')}(\omega) =
          \sum_{k} F_{\alpha\beta,kk}^{(gg')}(\omega)

    See Also
    --------
    calculate_control_matrix_from_scratch: Control matrix from scratch.
    calculate_control_matrix_from_atomic: Control matrix from concatenation.
    calculate_filter_function: Regular filter function.
    """
    if R.ndim != 4:
        raise ValueError('Expected R.ndim == 4.')

    if which == 'fidelity':
        return np.einsum('gako,hbko->ghabo', R.conj(), R)
    else:
        # which == 'generalized'
        return np.einsum('gako,hblo->ghabklo', R.conj(), R)


def diagonalize(H: ndarray, dt: Coefficients) -> Tuple[ndarray]:
    r"""Diagonalize a Hamiltonian.

    Diagonalize the Hamiltonian *H* which is piecewise constant during the
    times given by *dt* and return eigenvalues, eigenvectors, and the
    cumulative propagators :math:`Q_l`. Note that we calculate in units where
    :math:`\hbar\equiv 1` so that

    .. math::

        U(t, t_0) = \mathcal{T}\exp\left(
                        -i\int_{t_0}^t\mathrm{d}t'\mathcal{H}(t')
                    \right).

    Parameters
    ----------
    H: array_like, shape (n_dt, d, d)
        Hamiltonian of shape (n_dt, d, d) with d the dimensionality of the
        system
    dt: array_like
        The time differences

    Returns
    -------
    HD: ndarray
        Array of eigenvalues of shape (n_dt, d)
    HV: ndarray
        Array of eigenvectors of shape (n_dt, d, d)
    Q: ndarray
        Array of cumulative propagators of shape (n_dt+1, d, d)
    """
    d = H.shape[-1]
    # Calculate Eigenvalues and -vectors
    HD, HV = nla.eigh(H)
    # Propagator P = V exp(-j D dt) V^\dag. Middle term is of shape
    # (d, n_dt) due to transpose, so switch around indices in einsum
    # instead of transposing again. Same goes for the last term. This saves
    # a bit of time. The following is faster for larger dimensions but not for
    # many time steps:
    # P = np.empty((500, 4, 4), dtype=complex)
    # for l, (V, D) in enumerate(zip(HV, np.exp(-1j*dt*HD.T).T)):
    #     P[l] = (V * D) @ V.conj().T
    P = np.einsum('lij,jl,lkj->lik',
                  HV, util.cexp(-np.asarray(dt)*HD.T), HV.conj())
    # The cumulative propagator Q with the identity operator as first
    # element (Q_0 = P_0 = I), i.e.
    # Q = [Q_0, Q_1, ..., Q_n] = [P_0, P_1 @ P_0, ..., P_n @ ... @ P_0]
    Q = np.empty((len(dt)+1, d, d), dtype=complex)
    Q[0] = np.identity(d)
    for i in range(len(dt)):
        Q[i+1] = P[i] @ Q[i]

    return HD, HV, Q


def error_transfer_matrix(
        pulse: Optional['PulseSequence'] = None,
        S: Optional[ndarray] = None,
        omega: Optional[Coefficients] = None,
        K: Optional[ndarray] = None,
        n_oper_identifiers: Optional[Sequence[str]] = None,
        show_progressbar: Optional[bool] = False,
        memory_parsimonious: Optional[bool] = False) -> ndarray:
    r"""Compute the error transfer matrix up to unitary rotations.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to compute the error transfer
        matrix.
    S: array_like, shape ([[n_nops,] n_nops,] n_omega)
        The two-sided noise power spectral density in units of inverse
        frequencies as an array of shape (n_omega,), (n_nops, n_omega), or
        (n_nops, n_nops, n_omega). In the first case, the same spectrum is
        taken for all noise operators, in the second, it is assumed that there
        are no correlations between different noise sources and thus there is
        one spectrum for each noise operator. In the third and most general
        case, there may be a spectrum for each pair of noise operators
        corresponding to the correlations between them. n_nops is the number of
        noise operators considered and should be equal to
        ``len(n_oper_identifiers)``.
    omega: array_like,
        The frequencies. Note that the frequencies are assumed to be symmetric
        about zero.
    K: ndarray, shape ([[n_pls, n_pls,] n_nops,] n_nops, d**2, d**2)
        A precomputed cumulant function. If given, *pulse*, *S*, *omega*
        are not required.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which to evaluate the
        error transfer matrix. The default is all. Note that, since in general
        contributions from different noise operators won't commute, not
        selecting all noise operators results in neglecting terms of order
        :math:`\xi^4`.
    show_progressbar: bool, optional
        Show a progress bar for the calculation of the control matrix.
    memory_parsimonious: bool, optional
        Trade memory footprint for performance. See
        :func:`~numeric.calculate_decay_amplitudes`. The default is ``False``.

    Returns
    -------
    U: ndarray, shape (d**2, d**2)
        The error transfer matrix. The individual noise operator contributions
        are summed up before exponentiating as they might not commute.

    Notes
    -----
    The error transfer matrix is given by

    .. math::

        \tilde{\mathcal{U}} = \exp K(\tau)

    with :math:`K(\tau)` the cumulant function (see
    :func:`calculate_cumulant_function`). For Gaussian noise this expression is
    exact when taking into account the decay amplitudes :math:`\Gamma` and
    frequency shifts :math:`\Delta`. As the latter effects coherent errors it
    can be neglected if we assume that the experimenter has calibrated their
    pulse.

    For non-Gaussian noise the expression above is perturbative and includes
    noise up to order :math:`\xi^2` and hence
    :math:`\tilde{\mathcal{U}} = \mathbb{1} + K(\tau) + \mathcal{O}(\xi^2)`
    (although it is evaluated as a matrix exponential in any case).

    Given the above expression of the error transfer matrix, the entanglement
    fidelity is given by

    .. math::

        \mathcal{F}_\mathrm{e} = \frac{1}{d^2}\mathrm{tr}\,\tilde{\mathcal{U}}.

    See Also
    --------
    calculate_cumulant_function: Calculate the cumulant function :math:`K`
    calculate_decay_amplitudes: Calculate the :math:`\Gamma_{\alpha\beta,kl}`
    infidelity: Calculate only infidelity of a pulse.
    """
    if K is None:
        if pulse is None or S is None or omega is None:
            raise ValueError('Require either precomputed cumulant function K' +
                             ' or pulse, S, and omega as arguments.')

        K = calculate_cumulant_function(pulse, S, omega, n_oper_identifiers,
                                        'total', show_progressbar,
                                        memory_parsimonious)

    try:
        U = sla.expm(K.sum(axis=tuple(range(K.ndim - 2))))
    except AttributeError as aerr:
        raise TypeError('K invalid type: {}'.format(type(K))) from aerr
    except ValueError as verr:
        raise ValueError('K invalid shape: {}'.format(K.shape)) from verr

    return U


@util.parse_optional_parameters({'which': ('total', 'correlations')})
def infidelity(pulse: 'PulseSequence',
               S: Union[Coefficients, Callable],
               omega: Union[Coefficients, Dict[str, Union[int, str]]],
               n_oper_identifiers: Optional[Sequence[str]] = None,
               which: str = 'total',
               return_smallness: bool = False,
               test_convergence: bool = False) -> Union[ndarray, Any]:
    r"""Calculate the leading order entanglement infidelity.

    This function calculates the infidelity approximately from the leading
    peturbation (see :ref:`Notes <notes>`). To compute it exactly for Gaussian
    noise and vanishing coherent errors (second order Magnus terms), use
    :func:`error_transfer_matrix` to obtain it from the full process matrix.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to calculate the infidelity
        for.
    S: array_like, shape ([[n_nops,] n_nops,] omega) or callable
        The two-sided noise power spectral density in units of inverse
        frequencies as an array of shape (n_omega,), (n_nops, n_omega), or
        (n_nops, n_nops, n_omega). In the first case, the same spectrum is
        taken for all noise operators, in the second, it is assumed that there
        are no correlations between different noise sources and thus there is
        one spectrum for each noise operator. In the third and most general
        case, there may be a spectrum for each pair of noise operators
        corresponding to the correlations between them. n_nops is the number of
        noise operators considered and should be equal to
        ``len(n_oper_identifiers)``.

        If *test_convergence* is ``True``, a function handle to
        compute the power spectral density from a sequence of frequencies is
        expected.
    omega: array_like or dict
        The frequencies at which the integration is to be carried out. If
        *test_convergence* is ``True``, a dict with possible keys ('omega_IR',
        'omega_UV', 'spacing', 'n_min', 'n_max', 'n_points'), where all
        entries are integers except for ``spacing`` which should be a string,
        either 'linear' or 'log'. 'n_points' controls how many steps are taken.
        Note that the frequencies are assumed to be symmetric about zero.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which to calculate the
        infidelity  contribution. If given, the infidelities for each noise
        operator will be returned. Otherwise, all noise operators will be taken
        into account.
    which: str, optional
        Which infidelities should be calculated, may be either 'total'
        (default) or 'correlations'. In the former case, one value is returned
        for each noise operator, corresponding to the total infidelity of the
        pulse (or pulse sequence). In the latter, an array of infidelities is
        returned where element (i,j) corresponds to the infidelity contribution
        of the correlations between pulses i and j (see :ref:`Notes <notes>`).
        Note that this option is only available if the pulse correlation filter
        functions have been computed during concatenation (see
        :func:`calculate_pulse_correlation_filter_function` and
        :func:`~filter_functions.pulse_sequence.concatenate`).
    return_smallness: bool, optional
        Return the smallness parameter :math:`\xi` for the given spectrum.
    test_convergence: bool, optional
        Test the convergence of the integral with respect to the number of
        frequency samples. Returns the number of frequency samples and the
        corresponding fidelities. See *S* and *omega* for more information.

    Returns
    -------
    infid: ndarray, shape ([[n_pls, n_pls,], n_nops,] n_nops)
        Array with the infidelity contributions for each spectrum *S* on the
        last axis or axes, depending on the shape of *S* and *which*. If
        ``which`` is ``correlations``, the first two axes are the individual
        pulse contributions. If *S* is 2-d (3-d), the last axis (two axes) are
        the individual spectral contributions.
        Only if *test_convergence* is ``False``.
    n_samples: array_like
        Array with number of frequency samples used for convergence test.
        Only if *test_convergence* is ``True``.
    convergence_infids: array_like
        Array with infidelities calculated in convergence test.
        Only if *test_convergence* is ``True``.

    .. _notes:

    Notes
    -----
    The infidelity is given by

    .. math::

        \mathcal{I}_{\alpha\beta}
            &= 1 - \frac{1}{d^2}\mathrm{tr}\:\tilde{\mathcal{U}} \\
            &= \frac{1}{d}\int_{-\infty}^{\infty}\frac{\mathrm{d}\omega}{2\pi}
                S_{\alpha\beta}(\omega)F_{\alpha\beta}(\omega)
                +\mathcal{O}\big(\xi^4\big) \\
            &= \sum_{g,g'=1}^G \mathcal{I}_{\alpha\beta}^{(gg')}

    with :math:`S_{\alpha\beta}(\omega)` the two-sided noise spectral density
    and :math:`F_{\alpha\beta}(\omega)` the first-order filter function for
    noise sources :math:`\alpha,\beta`. The noise spectrum may include
    correlated noise sources, that is, its entry at :math:`(\alpha,\beta)`
    corresponds to the correlations between sources :math:`\alpha` and
    :math:`\beta`.
    :math:`\mathcal{I}_{\alpha\beta}^{(gg')}` are the correlation
    infidelities that can be computed by setting ``which='correlations'``.

    To convert to the average gate infidelity, use the
    following relation given by Horodecki et al. [Hor99]_ and
    Nielsen [Nie02]_:

    .. math::

        \mathcal{I}_\mathrm{avg} = \frac{d}{d+1}\mathcal{I}.

    The smallness parameter is given by

    .. math::

        \xi^2 = \sum_\alpha\left[
                    \lvert\lvert B_\alpha\rvert\rvert^2
                    \int_{-\infty}^\infty\frac{\mathrm{d}\omega}{2\pi}
                    S_\alpha(\omega)\left(\sum_ls_\alpha^{(l)}\Delta t_l
                    \right)^2
                \right].

    Note that in practice, the integral is only evaluated on the interval
    :math:`\omega\in[\omega_\mathrm{min},\omega_\mathrm{max}]`.

    See Also
    --------
    calculate_decay_amplitudes
    pulse_sequence.concatenate: Concatenate ``PulseSequence`` objects.
    calculate_pulse_correlation_filter_function

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

    See Also
    --------
    error_transfer_matrix: Calculate the full process matrix.
    plotting.plot_infidelity_convergence: Convenience function to plot results.
    """
    # Noise operator indices
    idx = util.get_indices_from_identifiers(pulse, n_oper_identifiers, 'noise')

    if test_convergence:
        if not callable(S):
            raise TypeError('Spectral density S should be callable when ' +
                            'test_convergence == True.')

        # Parse argument dict
        try:
            omega_IR = omega.get('omega_IR', 2*np.pi/pulse.tau*1e-2)
        except AttributeError:
            raise TypeError('omega should be dictionary with parameters ' +
                            'when test_convergence == True.')

        omega_UV = omega.get('omega_UV', 2*np.pi/pulse.tau*1e+2)
        spacing = omega.get('spacing', 'linear')
        n_min = omega.get('n_min', 100)
        n_max = omega.get('n_max', 500)
        n_points = omega.get('n_points', 10)

        # Alias numpy's linspace or logspace method depending on the spacing
        # omega has
        if spacing == 'linear':
            xspace = np.linspace
        elif spacing == 'log':
            xspace = np.geomspace
        else:
            raise ValueError("spacing should be either 'linear' or 'log'.")

        delta_n = (n_max - n_min)//n_points
        n_samples = np.arange(n_min, n_max + delta_n, delta_n)

        convergence_infids = np.empty((len(n_samples), len(idx)))
        for i, n in enumerate(n_samples):
            freqs = xspace(omega_IR, omega_UV, n//2)
            convergence_infids[i] = infidelity(
                pulse, *util.symmetrize_spectrum(S(freqs), freqs),
                n_oper_identifiers, which='total', return_smallness=False,
                test_convergence=False
            )

        return n_samples, convergence_infids

    if which == 'total':
        if not pulse.basis.istraceless:
            # Fidelity not simply sum of diagonal of decay amplitudes Gamma_kk
            # but trace tensor plays a role, cf eq. (39). For traceless bases,
            # the trace tensor term reduces to delta_ij.
            T = pulse.basis.four_element_traces
            Tp = (sparse.diagonal(T, axis1=2, axis2=3).sum(-1) -
                  sparse.diagonal(T, axis1=1, axis2=3).sum(-1)).todense()

            R = pulse.get_control_matrix(omega)
            F = np.einsum('ako,blo,kl->abo', R.conj(), R, Tp)/pulse.d
        else:
            F = pulse.get_filter_function(omega)
    else:
        # which == 'correlations'
        if not pulse.basis.istraceless:
            warn('Calculating pulse correlation fidelities with non-' +
                 'traceless basis. The results will be off.')

        if pulse.is_cached('omega'):
            if not np.array_equal(pulse.omega, omega):
                raise ValueError('Pulse correlation infidelities requested ' +
                                 'but omega not equal to cached frequencies.')

        F = pulse.get_pulse_correlation_filter_function()

    integrand = _get_integrand(S, omega, idx, which, 'fidelity', F=F)
    infid = integrate.trapz(integrand, omega).real/(2*np.pi*pulse.d)

    if return_smallness:
        if S.ndim > 2:
            raise NotImplementedError('Smallness parameter only implemented' +
                                      'for uncorrelated noise sources')

        T1 = integrate.trapz(S, omega)/(2*np.pi)
        T2 = (pulse.dt*pulse.n_coeffs[idx]).sum(axis=-1)**2
        T3 = util.abs2(pulse.n_opers[idx]).sum(axis=(1, 2))
        xi = np.sqrt((T1*T2*T3).sum())

        return infid, xi

    return infid


def _get_integrand(S: ndarray, omega: ndarray, idx: ndarray, which_pulse: str,
                   which_FF: str,
                   R: Optional[Union[ndarray, Sequence[ndarray]]] = None,
                   F: Optional[ndarray] = None) -> ndarray:
    """
    Private function to generate the integrand for either :func:`infidelity` or
    :func:`calculate_decay_amplitudes`.

    Parameters
    ----------
    S: array_like, shape ([[n_nops,] n_nops,] n_omega)
        The two-sided noise power spectral density.
    omega: array_like,
        The frequencies. Note that the frequencies are assumed to be symmetric
        about zero.
    idx: ndarray
        Noise operator indices to consider.
    which_pulse: str, optional {'total', 'correlations'}
        Use pulse correlations or total filter function.
    which_FF: str, optional {'fidelity', 'generalized'}
        Fidelity or generalized filter functions. Needed to determine output
        shape.
    R: ndarray, optional
        Control matrix. If given, returns the integrand for
        :func:`calculate_decay_amplitudes`. If given as a list or tuple, taken
        to be the left and right control matrices in the integrand (allows for
        slicing up the integrand).
    F: ndarray, optional
        Filter function. If given, returns the integrand for
        :func:`infidelity`.

    Raises
    ------
    ValueError
        If ``S`` and ``R`` or ``F``, depending on which was given, have
        incompatible shapes.

    Returns
    -------
    integrand: ndarray, shape (..., n_omega)
        The integrand.

    """
    if R is not None:
        # R_left is the complex conjugate
        funs = (np.conj, lambda x: x)
        if isinstance(R, (list, tuple)):
            R_left, R_right = [f(r) for f, r in zip(funs, R)]
        else:
            R_left, R_right = [f(r) for f, r in zip(funs, [R]*2)]
    else:
        # F is not None
        if which_FF == 'generalized':
            # Everything simpler if noise operators always on 2nd-to-last axes
            F = np.moveaxis(F, source=[-5, -4], destination=[-3, -2])

    S = np.asarray(S)
    S_err_str = 'S should be of shape {}, not {}.'
    if S.ndim == 1 or S.ndim == 2:
        if S.ndim == 1:
            # Only single spectrum
            shape = (len(omega),)
            if S.shape != shape:
                raise ValueError(S_err_str.format(shape, S.shape))

            S = np.expand_dims(S, 0)
        else:
            # S.ndim == 2, S is diagonal (no correlation between noise sources)
            shape = (len(idx), len(omega))
            if S.shape != shape:
                raise ValueError(S_err_str.format(shape, S.shape))

        # S is real, integrand therefore also
        if F is not None:
            integrand = (F[..., tuple(idx), tuple(idx), :]*S).real
            if which_FF == 'generalized':
                # move axes back to expected position, ie (pulses, noise opers,
                # basis elements, frequencies)
                integrand = np.moveaxis(integrand, source=-2, destination=-4)
        else:
            # R is not None
            if which_pulse == 'correlations':
                if which_FF == 'fidelity':
                    einsum_str = 'gako,ao,hako->ghao'
                else:
                    # which_FF == 'generalized'
                    einsum_str = 'gako,ao,halo->ghaklo'
            else:
                # which_pulse == 'total'
                if which_FF == 'fidelity':
                    einsum_str = 'ako,ao,ako->ao'
                else:
                    # which_FF == 'generalized'
                    einsum_str = 'ako,ao,alo->aklo'

            integrand = np.einsum(einsum_str,
                                  R_left[..., idx, :, :], S,
                                  R_right[..., idx, :, :]).real
    elif S.ndim == 3:
        # General case where S is a matrix with correlation spectra on off-diag
        shape = (len(idx), len(idx), len(omega))
        if S.shape != shape:
            raise ValueError(S_err_str.format(shape, S.shape))

        if F is not None:
            integrand = F[..., idx[:, None], idx, :]*S
            if which_FF == 'generalized':
                integrand = np.moveaxis(integrand, source=[-3, -2],
                                        destination=[-5, -4])
        else:
            # R is not None
            if which_pulse == 'correlations':
                if which_FF == 'fidelity':
                    einsum_str = 'gako,abo,hbko->ghabo'
                else:
                    # which_FF == 'generalized'
                    einsum_str = 'gako,abo,hblo->ghabklo'
            else:
                # which_pulse == 'total'
                if which_FF == 'fidelity':
                    einsum_str = 'ako,abo,bko->abo'
                else:
                    # which_FF == 'generalized'
                    einsum_str = 'ako,abo,blo->abklo'

            integrand = np.einsum(einsum_str,
                                  R_left[..., idx, :, :], S,
                                  R_right[..., idx, :, :])
    else:
        raise ValueError('Expected S to be array_like with < 4 dimensions')

    return integrand

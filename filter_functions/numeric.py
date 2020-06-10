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
:func:`liouville_representation`
    Calculate the Liouville representation of a unitary with respect to a basis
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
from scipy import linalg as sla
from scipy import integrate

from . import util
from .basis import Basis, ggm_expand
from .plotting import plot_infidelity_convergence
from .types import Coefficients, Operator

__all__ = ['calculate_control_matrix_from_atomic',
           'calculate_control_matrix_from_scratch',
           'calculate_cumulant_function',
           'calculate_decay_amplitudes',
           'calculate_filter_function',
           'calculate_pulse_correlation_filter_function', 'diagonalize',
           'error_transfer_matrix', 'infidelity', 'liouville_representation']


def calculate_control_matrix_from_atomic(
        phases: ndarray, R_g: ndarray, Q_liouville: ndarray,
        show_progressbar: Optional[bool] = None) -> ndarray:
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

    Returns
    -------
    R: ndarray, shape (n_nops, d**2, n_omega)
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
    # Allocate memory
    R = np.zeros(R_g.shape[1:], dtype=complex)

    # Set up a reusable contraction expression. In some cases it is faster to
    # also contract the time dimension in the same expression instead of
    # looping over it, but we don't distinguish here for readability.
    R_expr = oe.contract_expression('ijo,jk->iko',
                                    R_g.shape[1:], Q_liouville.shape[1:])

    for g in util.progressbar_range(n, show_progressbar=show_progressbar,
                                    desc='Calculating control matrix'):
        R += R_expr(phases[g]*R_g[g], Q_liouville[g])

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
        show_progressbar: Optional[bool] = False) -> ndarray:
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
    if d < 4:
        # Einsum contraction faster
        QdagV = np.einsum('lba,lbc->lac', Q[:-1].conj(), HV)
        B = np.einsum('lba,jbc,lcd->jlad', HV.conj(), n_opers, HV,
                      optimize=['einsum_path', (0, 1), (0, 1)])
    else:
        QdagV = Q[:-1].transpose(0, 2, 1).conj() @ HV
        B = np.empty((len(n_opers), len(dt), d, d), dtype=complex)
        for j, n_oper in enumerate(n_opers):
            B[j] = HV.conj().transpose(0, 2, 1) @ n_oper @ HV

    # Allocate result and buffers for intermediate arrays
    R = np.zeros((len(n_opers), len(basis), len(E)), dtype=complex)
    exp_buf = np.empty((len(E), d, d), dtype=complex)
    int_buf = np.empty((len(E), d, d), dtype=complex)
    R_path = ['einsum_path', (0, 3), (0, 1), (0, 2), (0, 1)]

    for l in util.progressbar_range(len(dt), show_progressbar=show_progressbar,
                                    desc='Calculating control matrix'):

        dE = np.subtract.outer(HD[l], HD[l])
        # iEdE_nm = 1j*(omega + omega_n - omega_m)
        int_buf.real = 0
        int_buf.imag = np.add.outer(E, dE, out=int_buf.imag)

        # Use expm1 for better convergence with small arguments
        exp_buf = np.expm1(int_buf*dt[l], out=exp_buf)
        # Catch zero-division warnings
        mask = (int_buf != 0)
        int_buf = np.divide(exp_buf, int_buf, out=int_buf, where=mask)
        int_buf[~mask] = dt[l]

        # Faster for d = 2 to also contract over the time dimension instead of
        # loop, but for readability we don't distinguish.
        R += np.einsum('o,j,jmn,omn,knm->jko',
                       util.cexp(E*t[l]), n_coeffs[:, l], B[:, l], int_buf,
                       QdagV[l].conj().T @ basis @ QdagV[l],
                       optimize=R_path)

    return R


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


def calculate_cumulant_function(
        pulse: 'PulseSequence',
        S: ndarray,
        omega: Coefficients,
        n_oper_identifiers: Optional[Sequence[str]] = None,
        show_progressbar: Optional[bool] = False,
        memory_parsimonious: Optional[bool] = False) -> ndarray:
    r"""Calculate the cumulant function :math:`K(\tau)`.

    The error transfer matrix is obtained from the cumulant function by
    exponentiation, :math:`\tilde{\mathcal{U}} = \exp K(\tau)`.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to compute the cumulant
        function.
    S: array_like, shape (..., n_omega)
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
    show_progressbar: bool, optional
        Show a progress bar for the calculation of the control matrix.
    memory_parsimonious: bool, optional
        Trade memory footprint for performance. See
        :func:`~numeric.calculate_decay_amplitudes`. The default is ``False``.

    Returns
    -------
    K: ndarray, shape ([n_nops,] n_nops, d**2, d**2)
        The cumulant function. The individual noise operator contributions
        chosen by ``n_oper_identifiers`` are on the first axis / axes,
        depending on whether the noise is cross-correlated or not.

    Notes
    -----
    The cumulant function is given by

    .. math::

        K_{ij}(\tau) = -\frac{1}{2}\sum_{\alpha\beta} & \sum_{kl}\biggl(
            \Delta_{\alpha\beta,kl}\left(
                T_{klji} - T_{lkji} - T_{klij} + T_{lkij}
            \right) + \Gamma_{\alpha\beta,kl}\left(
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

        K_{ij}(\tau) = \begin{cases}
            - \sum_{k\neq i}\Gamma_{kk}                     &\qif* i = j,   \\
                - \Delta_{ij} + \Delta_{ji} + \Gamma_{ij}   &\qif* i\neq j,
        \end{cases}

    for :math:`i\in\{1,2,3\}` and :math:`K_{0j} = K_{i0} = 0`.

    See Also
    --------
    calculate_decay_amplitudes: Calculate the :math:`\Gamma_{\alpha\beta,kl}`
    error_transfer_matrix: Calculate the error transfer matrix :math:`\exp K`.
    infidelity: Calculate only infidelity of a pulse.

    """
    N, d = pulse.basis.shape[:2]
    Gamma = calculate_decay_amplitudes(pulse, S, omega, n_oper_identifiers,
                                       show_progressbar, memory_parsimonious)

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
            K[..., i, i] = -Gamma[..., diag_items, diag_items].sum(axis=-1)
            # shift the item not summed over by one
            diag_items.rotate()
    else:
        # Multi qubit case. Use general expression.
        T = pulse.basis.four_element_traces
        K = (oe.contract('...kl,lkji->...ij', Gamma, T, backend='sparse') +
             oe.contract('...kl,klij->...ij', Gamma, T, backend='sparse') -
             oe.contract('...kl,klji->...ij', Gamma, T, backend='sparse') -
             oe.contract('...kl,lkij->...ij', Gamma, T, backend='sparse'))/2

    return K


def calculate_decay_amplitudes(
        pulse: 'PulseSequence',
        S: ndarray,
        omega: Coefficients,
        n_oper_identifiers: Optional[Sequence[str]] = None,
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
    S: array_like, shape (..., n_omega)
        The two-sided noise power spectral density.
    omega: array_like,
        The frequencies. Note that the frequencies are assumed to be symmetric
        about zero.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which to calculate the decay
        amplitudes. The default is all.
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
    Gamma: ndarray, shape (..., d**2, d**2)
        The decay amplitudes.

    Notes
    -----
    The decay amplitudes are given by

    .. math::

        \Gamma_{\alpha\beta, kl} = \int
            \frac{\mathrm{d}\omega}{2\pi}\mathcal{R}^\ast_{\alpha k}(\omega)
            S_{\alpha\beta}(\omega)\mathcal{R}_{\beta l}(\omega).

    """
    # TODO: Implement for correlation FFs? Replace infidelity() by this?
    # Noise operator indices
    idx = util.get_indices_from_identifiers(pulse, n_oper_identifiers, 'noise')
    R = pulse.get_control_matrix(omega, show_progressbar)[idx]

    if not memory_parsimonious:
        integrand = _get_integrand(S, omega, idx, R=R)
        Gamma = integrate.trapz(integrand, omega, axis=-1)/(2*np.pi)
        return Gamma

    # Conserve memory by looping. Let _get_integrand determine the shape
    integrand = _get_integrand(S, omega, idx, R=[R[:, 0:1], R])

    n_kl = R.shape[1]
    Gamma = np.zeros(integrand.shape[:-3] + (n_kl,)*2,
                     dtype=integrand.dtype)
    Gamma[..., 0:1, :] = integrate.trapz(integrand, omega, axis=-1)/(2*np.pi)

    for k in util.progressbar_range(1, n_kl, show_progressbar=show_progressbar,
                                    desc='Integrating'):
        integrand = _get_integrand(S, omega, idx, R=[R[:, k:k+1], R])
        Gamma[..., k:k+1, :] = integrate.trapz(integrand, omega,
                                               axis=-1)/(2*np.pi)

    return Gamma


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
    elif which == 'generalized':
        return np.einsum('ako,blo->abklo', R.conj(), R)


@util.parse_which_FF_parameter
def calculate_pulse_correlation_filter_function(
        R: ndarray, which: str = 'fidelity') -> ndarray:
    r"""Compute pulse correlation filter function from the control matrix.

    Parameters
    ----------
    R: array_like, shape (n_pulses, n_nops, d**2, n_omega)
        The control matrix.

    Returns
    -------
    F_pc: ndarray, shape (n_pls, n_pls, n_nops, n_nops, [d**2, d**2,] n_omega)
        The pulse correlation filter functions for each pulse and noise
        operator correlations. The first two axes hold the pulse correlations,
        the second two the noise correlations.
    which : str, optional
        Which filter function to return. Either 'fidelity' (default) or
        'generalized' (see :ref:`Notes <notes>`).

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
    elif which == 'generalized':
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
        pulse: 'PulseSequence',
        S: ndarray,
        omega: Coefficients,
        n_oper_identifiers: Optional[Sequence[str]] = None,
        show_progressbar: Optional[bool] = False,
        memory_parsimonious: Optional[bool] = False) -> ndarray:
    r"""Compute the error transfer matrix up to unitary rotations.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to compute the error transfer
        matrix.
    S: array_like, shape (..., n_omega)
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
        error transfer matrix. The default is all.
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
    K = calculate_cumulant_function(pulse, S, omega, n_oper_identifiers,
                                    show_progressbar, memory_parsimonious)

    U = sla.expm(K.sum(axis=list(range(K.ndim - 2))))
    return U


def infidelity(pulse: 'PulseSequence',
               S: Union[Coefficients, Callable],
               omega: Union[Coefficients, Dict[str, Union[int, str]]],
               n_oper_identifiers: Optional[Sequence[str]] = None,
               which: str = 'total',
               return_smallness: bool = False,
               test_convergence: bool = False) -> Union[ndarray, Any]:
    r"""
    Calculate the ensemble average of the entanglement infidelity of the
    ``PulseSequence`` *pulse*.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to calculate the infidelity
        for.
    S: array_like or callable
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
        :func:`~filter_functions.pulse_sequence.concatenate`) and that in this
        case no checks are performed if the frequencies are compliant.
    return_smallness: bool, optional
        Return the smallness parameter :math:`\xi` for the given spectrum.
    test_convergence: bool, optional
        Test the convergence of the integral with respect to the number of
        frequency samples. Plots the infidelities against the number of
        frequency samples. See *S* and *omega* for more information.

    Returns
    -------
    infid: ndarray
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
    (fig, ax): tuple
        The matplotlib figure and axis instances used for plotting.
        Only if *test_convergence* is ``True``.

    .. _notes:

    Notes
    -----
    The infidelity is given by

    .. math::

        \big\langle\mathcal{I}_\mathrm{e}\big\rangle_{\alpha\beta} &=
                \frac{1}{2\pi d}\int_{-\infty}^{\infty}\mathrm{d}\omega\,
                S_{\alpha\beta}(\omega)F_{\alpha\beta}(\omega)
                +\mathcal{O}\big(\xi^4\big) \\
            &= \sum_{g,g'=1}^G \big\langle
                \mathcal{I}_\mathrm{e}\big\rangle_{\alpha\beta}^{(gg')}

    with :math:`S_{\alpha\beta}(\omega)` the two-sided noise spectral density
    and :math:`F_{\alpha\beta}(\omega)` the first-order filter function for
    noise sources :math:`\alpha,\beta`. The noise spectrum may include
    correlated noise sources, that is, its entry at :math:`(\alpha,\beta)`
    corresponds to the correlations between sources :math:`\alpha` and
    :math:`\beta`.
    :math:`\big\langle\mathcal{I}_\mathrm{e}\big\rangle_{\alpha\beta}^{(gg')}`
    are the correlation infidelities that can be computed by setting
    ``which='correlations'``.

    To convert to the average gate infidelity, use the
    following relation given by Horodecki et al. [Hor99]_ and
    Nielsen [Nie02]_:

    .. math::

        \big\langle\mathcal{I}_\mathrm{avg}\big\rangle = \frac{d}{d+1}
                \big\langle\mathcal{I}_\mathrm{e}\big\rangle.

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
    # Noise operator indices
    idx = util.get_indices_from_identifiers(pulse, n_oper_identifiers, 'noise')

    if test_convergence:
        if not callable(S):
            raise TypeError('Spectral density S should be callable when ' +
                            'test_convergence == True.')

        # Parse argument dict
        try:
            omega_IR = omega.get('omega_IR', 2*np.pi/pulse.t[-1]*1e-2)
        except AttributeError:
            raise TypeError('omega should be dictionary with parameters ' +
                            'when test_convergence == True.')

        omega_UV = omega.get('omega_UV', 2*np.pi/pulse.t[-1]*1e+2)
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

        fig, ax = plot_infidelity_convergence(n_samples,
                                              convergence_infids.sum(axis=1))

        return n_samples, convergence_infids, (fig, ax)

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
    elif which == 'correlations':
        if not pulse.basis.istraceless:
            warn('Calculating pulse correlation fidelities with non-' +
                 'traceless basis. The results will be off.')

        F = pulse.get_pulse_correlation_filter_function()
    else:
        raise ValueError("Unrecognized option for 'which': {}.".format(which))

    S = np.asarray(S)
    slices = [slice(None)]*F.ndim
    if S.ndim == 3:
        slices[-3] = idx[:, None]
        slices[-2] = idx[None, :]
    else:
        slices[-3] = idx
        slices[-2] = idx

    integrand = _get_integrand(S, omega, idx, F=F[tuple(slices)])

    infid = integrate.trapz(integrand, omega)/(2*np.pi*pulse.d)

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


def liouville_representation(U: ndarray, basis: Basis) -> ndarray:
    r"""
    Get the Liouville representaion of the unitary U with respect to the basis
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
    :math:`\mathbb{C}^{d\times d}` with :math:`d` the dimension of the Hilbert
    space.
    """
    U = np.asanyarray(U)
    if basis.btype == 'GGM' and basis.d > 12:
        # Can do closed form expansion and overhead compensated
        path = ['einsum_path', (0, 1), (0, 1)]
        conjugated_basis = np.einsum('...ba,ibc,...cd->...iad', U.conj(),
                                     basis, U, optimize=path)
        # If the basis is hermitian, the result will be strictly real so we can
        # drop the imaginary part
        R = ggm_expand(conjugated_basis).real
    else:
        path = ['einsum_path', (0, 1), (0, 1), (0, 1)]
        R = np.einsum('...ba,ibc,...cd,jda', U.conj(), basis, U, basis,
                      optimize=path).real

    return R


def _get_integrand(S: ndarray, omega: ndarray, idx: ndarray,
                   R: Optional[Union[ndarray, Sequence[ndarray]]] = None,
                   F: Optional[ndarray] = None) -> ndarray:
    """
    Private function to generate the integrand for either :func:`infidelity` or
    :func:`calculate_decay_amplitudes`.

    Parameters
    ----------
    S: array_like, shape (..., n_omega)
        The two-sided noise power spectral density.
    omega: array_like,
        The frequencies. Note that the frequencies are assumed to be symmetric
        about zero.
    idx: ndarray
        Noise operator indices to consider.
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

    S = np.asarray(S)
    S_err_str = 'S should be of shape {}, not {}.'
    if S.ndim == 1:
        # Only single spectrum
        shape = (len(omega),)
        if S.shape != shape:
            raise ValueError(S_err_str.format(shape, S.shape))

        # S is real, integrand therefore also
        if F is not None:
            integrand = (F*S).real
        elif R is not None:
            integrand = np.einsum('jko,jlo->jklo', R_left, S*R_right).real
    elif S.ndim == 2:
        # S is diagonal (no correlation between noise sources)
        shape = (len(idx), len(omega))
        if S.shape != shape:
            raise ValueError(S_err_str.format(shape, S.shape))

        # S is real, integrand therefore also
        if F is not None:
            integrand = (F*S).real
        elif R is not None:
            integrand = np.einsum('jko,jo,jlo->jklo', R_left, S, R_right).real
    elif S.ndim == 3:
        # General case where S is a matrix with correlation spectra on off-diag
        shape = (len(idx), len(idx), len(omega))
        if S.shape != shape:
            raise ValueError(S_err_str.format(shape, S.shape))

        if F is not None:
            integrand = F*S
        elif R is not None:
            integrand = np.einsum('iko,ijo,jlo->ijklo', R_left, S, R_right)
    elif S.ndim > 3:
        raise ValueError('Expected S to be array_like with < 4 dimensions')

    return integrand

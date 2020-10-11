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
This module defines the functions to calculate everything related to
filter functions.

Functions
---------
:func:`calculate_control_matrix_from_atomic`
    Calculate the control matrix from those of atomic pulse sequences
:func:`calculate_control_matrix_from_scratch`
    Calculate the control matrix from scratch
:func:`calculate_control_matrix_periodic`
    Calculate the control matrix for a periodic Hamiltonian
:func:`calculate_noise_operators_from_atomic`
    Calculate the interaction picture noise operators from atomic segments.
    Same calculation as :func:`calculate_control_matrix_from_atomic`
    except in Hilbert space.
:func:`calculate_noise_operators_from_scratch`
    Calculate the interaction picture noise operators from scratch. Same
    calculation as :func:`calculate_control_matrix_from_scratch` except
    in Hilbert space.
:func:`calculate_cumulant_function`
    Calculate the cumulant function for a given ``PulseSequence`` object.
:func:`calculate_decay_amplitudes`
    Calculate the decay amplitudes, corresponding to first order terms
    of the Magnus expansion
:func:`calculate_filter_function`
    Calculate the filter function from the control matrix
:func:`calculate_pulse_correlation_filter_function`
    Calculate the pulse correlation filter function from the control
    matrix
:func:`diagonalize`
    Diagonalize a Hamiltonian
:func:`error_transfer_matrix`
    Calculate the error transfer matrix of a pulse up to a unitary
    rotation
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

__all__ = ['calculate_control_matrix_from_atomic', 'calculate_control_matrix_from_scratch',
           'calculate_control_matrix_periodic', 'calculate_noise_operators_from_atomic',
           'calculate_noise_operators_from_scratch', 'calculate_cumulant_function',
           'calculate_decay_amplitudes', 'calculate_filter_function',
           'calculate_pulse_correlation_filter_function', 'diagonalize', 'error_transfer_matrix',
           'infidelity']


def _propagate_eigenvectors(propagators, eigvecs):
    """Propagate the eigenvectors with the unitary propagators"""
    return propagators.transpose(0, 2, 1).conj() @ eigvecs


def _transform_noise_operators(n_coeffs, n_opers, eigvecs):
    r"""
    Transform noise operators into the eigenspaces spanned by eigvecs.
    I.e., the following transformation is performed:

    .. math::

        B_\alpha\rightarrow s_\alpha^{(g)}V^{(g)}B_\alpha V^{(g)\dagger}

    """
    assert len(n_opers) == len(n_coeffs)
    n_opers_transformed = np.empty((len(n_opers), *eigvecs.shape), dtype=complex)
    for j, (n_coeff, n_oper) in enumerate(zip(n_coeffs, n_opers)):
        n_opers_transformed[j] = n_oper @ eigvecs
        n_opers_transformed[j] = eigvecs.conj().transpose(0, 2, 1) @ n_opers_transformed[j]
        n_opers_transformed[j] *= n_coeff[:, None, None]

    return n_opers_transformed


def _transform_basis(basis, eigvecs_propagated, out=None):
    r"""
    Transform the basis into the eigenspace spanned by V propagated by Q

    I.e., the following transformation is performed:

    .. math::

        C_k\rightarrow Q_{g-1}V^{(g)\dagger}C_k V^{(g)}Q_{g-1}^\dagger.

    """
    out = np.matmul(basis, eigvecs_propagated, out=out)
    out = np.matmul(eigvecs_propagated.conj().T, out, out=out)
    return out


def _first_order_integral(E: ndarray, eigvals: ndarray, dt: float,
                          exp_buf: ndarray, int_buf: ndarray) -> ndarray:
    r"""Calculate the integral appearing in first order Magnus expansion.

    The integral is evaluated as

    .. math::
        I_{mn}^{(g)}(\omega) = \frac
            {e^{i(\omega + \Omega_{mn}^{(g)})\Delta t_g} - 1}
            {i(\omega + \Omega_{mn}^{(g)})}

    """
    dE = np.subtract.outer(eigvals, eigvals)
    # iEdE_nm = 1j*(omega + omega_n - omega_m)
    int_buf.real = 0
    int_buf.imag = np.add.outer(E, dE, out=int_buf.imag)

    # Catch zero-division
    mask = (int_buf.imag != 0)
    exp_buf = util.cexp(int_buf.imag*dt, out=exp_buf, where=mask)
    exp_buf = np.subtract(exp_buf, 1, out=exp_buf, where=mask)
    int_buf = np.divide(exp_buf, int_buf, out=int_buf, where=mask)
    int_buf[~mask] = dt

    return int_buf


def _get_integrand(
        spectrum: ndarray,
        omega: ndarray,
        idx: ndarray,
        which_pulse: str,
        which_FF: str,
        control_matrix: Optional[Union[ndarray, Sequence[ndarray]]] = None,
        filter_function: Optional[ndarray] = None
) -> ndarray:
    """
    Private function to generate the integrand for either
    :func:`infidelity` or :func:`calculate_decay_amplitudes`.

    Parameters
    ----------
    spectrum: array_like, shape ([[n_nops,] n_nops,] n_omega)
        The two-sided noise power spectral density.
    omega: array_like,
        The frequencies at which to calculate the filter functions.
    idx: ndarray
        Noise operator indices to consider.
    which_pulse: str, optional {'total', 'correlations'}
        Use pulse correlations or total filter function.
    which_FF: str, optional {'fidelity', 'generalized'}
        Fidelity or generalized filter functions. Needed to determine
        output shape.
    control_matrix: ndarray, optional
        Control matrix. If given, returns the integrand for
        :func:`calculate_error_vector_correlation_functions`. If given
        as a list or tuple, taken to be the left and right control
        matrices in the integrand (allows for slicing up the integrand).
    filter_function: ndarray, optional
        Filter function. If given, returns the integrand for
        :func:`infidelity`.

    Raises
    ------
    ValueError
        If ``spectrum`` and ``control_matrix`` or ``filter_function``,
        depending on which was given, have incompatible shapes.

    Returns
    -------
    integrand: ndarray, shape (..., n_omega)
        The integrand.

    """
    if control_matrix is not None:
        # ctrl_left is the complex conjugate
        funs = (np.conj, lambda x: x)
        if isinstance(control_matrix, (list, tuple)):
            ctrl_left, ctrl_right = [f(c) for f, c in zip(funs, control_matrix)]
        else:
            ctrl_left, ctrl_right = [f(r) for f, r in zip(funs, [control_matrix]*2)]
    else:
        # filter_function is not None
        if which_FF == 'generalized':
            # Everything simpler if noise operators always on 2nd-to-last axes
            filter_function = np.moveaxis(filter_function, source=[-5, -4], destination=[-3, -2])

    spectrum = np.asarray(spectrum)
    S_err_str = 'spectrum should be of shape {}, not {}.'
    if spectrum.ndim == 1 or spectrum.ndim == 2:
        if spectrum.ndim == 1:
            # Only single spectrum
            shape = (len(omega),)
            if spectrum.shape != shape:
                raise ValueError(S_err_str.format(shape, spectrum.shape))

            spectrum = np.expand_dims(spectrum, 0)
        else:
            # spectrum.ndim == 2, spectrum is diagonal (no correlation between noise sources)
            shape = (len(idx), len(omega))
            if spectrum.shape != shape:
                raise ValueError(S_err_str.format(shape, spectrum.shape))

        # spectrum is real, integrand therefore also
        if filter_function is not None:
            integrand = (filter_function[..., tuple(idx), tuple(idx), :]*spectrum).real
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
                                  ctrl_left[..., idx, :, :], spectrum,
                                  ctrl_right[..., idx, :, :]).real
    elif spectrum.ndim == 3:
        # General case where spectrum is a matrix with correlation spectra on off-diag
        shape = (len(idx), len(idx), len(omega))
        if spectrum.shape != shape:
            raise ValueError(S_err_str.format(shape, spectrum.shape))

        if filter_function is not None:
            integrand = filter_function[..., idx[:, None], idx, :]*spectrum
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
                                  ctrl_left[..., idx, :, :], spectrum,
                                  ctrl_right[..., idx, :, :])
    else:
        raise ValueError('Expected spectrum to be array_like with < 4 dimensions')

    return integrand


def calculate_noise_operators_from_atomic(
        phases: ndarray,
        noise_operators_atomic: ndarray,
        propagators: ndarray,
        show_progressbar: bool = False
) -> ndarray:
    r"""
    Calculate the interaction picutre noise operators from atomic segments.

    Parameters
    ----------
    phases: array_like, shape (n_dt, n_omega)
        The phase factors for :math:`g\in\{0, 1, \dots, G-1\}`.
    noise_operators_atomic: array_like, shape (n_dt, n_nops, d, d, n_omega)
        The noise operators in the interaction picture of the g-th
        pulse, i.e. for :math:`g\in\{1, 2, \dots, G\}`.
    propagators: array_like, shape (n_dt, d, d)
        The cumulative propagators of the pulses
        :math:`g\in\{0, 1, \dots, G-1\}`.
    show_progressbar: bool, optional
        Show a progress bar for the calculation.

    Returns
    -------
    noise_operators: ndarray, shape (n_omega, n_nops, d, d)
        The interaction picture noise operators
        :math:`\tilde{B}_\alpha(\omega)`.

    Notes
    -----
    The noise operators are calculated by evaluating the sum

    .. math::

        \tilde{B}_\alpha(\omega) = \sum_{g=1}^G e^{i\omega t_{g-1}}
             Q_{g-1}^\dagger\tilde{B}_\alpha^{(g)}(\omega) Q_{g-1}.

    The control matrix then corresponds to the coefficients of expansion
    in an operator basis :math:`\{C_k\}_k`:

    .. math::
        \tilde{\mathcal{B}}_{k\alpha}(\omega) =
            \mathrm{tr}(\tilde{B}_\alpha(\omega) C_k).

    Due to differences in implementation (for performance reasons), the
    axes of the result are transposed compared to the control matrix:

    >>> ctrlmat = calculate_control_matrix_from_atomic(...)
    >>> ctrlmat.shape
    (n_nops, d**2, n_omega)
    >>> noiseops = calculate_noise_operators_from_atomic(...)
    >>> noiseops.shape
    (n_omega, n_nops, d, d)
    >>> ctrlmat_from_noiseops = ff.basis.expand(noiseops, basis)
    >>> np.allclose(ctrlmat, ctrlmat_from_noiseops.transpose(1, 2, 0))
    True

    See Also
    --------
    :func:`calculate_noise_operators_from_scratch`: Compute the operators from scratch.
    :func:`calculate_control_matrix_from_atomic`: Same calculation in Liouville space.
    """
    n = len(noise_operators_atomic)
    # Allocate memory
    noise_operators = np.zeros(noise_operators_atomic.shape[1:], dtype=complex)

    expr = oe.contract_expression('ji,...jk,kl->...il',
                                  propagators.shape[1:], noise_operators_atomic.shape[1:],
                                  propagators.shape[1:], optimize=[(0, 1), (0, 1)])

    for g in util.progressbar_range(n, show_progressbar=show_progressbar,
                                    desc='Calculating noise operators'):
        noise_operators += expr(propagators[g].conj(),
                                noise_operators_atomic[g]*phases[g, :, None, None, None],
                                propagators[g])

    return noise_operators


def calculate_noise_operators_from_scratch(
        eigvals: ndarray,
        eigvecs: ndarray,
        propagators: ndarray,
        omega: Coefficients,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        dt: Coefficients,
        t: Optional[Coefficients] = None,
        show_progressbar: bool = False
) -> ndarray:
    r"""
    Calculate the noise operators in interaction picture from scratch.

    Parameters
    ----------
    eigvals: array_like, shape (n_dt, d)
        Eigenvalue vectors for each time pulse segment *g* with the first
        axis counting the pulse segment, i.e.
        ``eigvals == array([D_0, D_1, ...])``.
    eigvecs: array_like, shape (n_dt, d, d)
        Eigenvector matrices for each time pulse segment *g* with the first
        axis counting the pulse segment, i.e.
        ``eigvecs == array([V_0, V_1, ...])``.
    propagators: array_like, shape (n_dt+1, d, d)
        The propagators :math:`Q_g = P_g P_{g-1}\cdots P_0` as a (d, d) array
        with *d* the dimension of the Hilbert space.
    omega: array_like, shape (n_omega,)
        Frequencies at which the pulse control matrix is to be evaluated.
    n_opers: array_like, shape (n_nops, d, d)
        Noise operators :math:`B_\alpha`.
    n_coeffs: array_like, shape (n_nops, n_dt)
        The sensitivities of the system to the noise operators given by
        *n_opers* at the given time step.
    dt: array_like, shape (n_dt)
        Sequence duration, i.e. for the :math:`g`-th pulse
        :math:`t_g - t_{g-1}`.
    t: array_like, shape (n_dt+1), optional
        The absolute times of the different segments. Can also be
        computed from *dt*.
    show_progressbar: bool, optional
        Show a progress bar for the calculation.

    Returns
    -------
    noise_operators: ndarray, shape (n_omega, n_nops, d, d)
        The interaction picture noise operators
        :math:`\tilde{B}_\alpha(\omega)`.

    Notes
    -----
    The interaction picture noise operators are calculated according to

    .. math::

        \tilde{B}_\alpha(\omega) = \sum_{g=1}^G e^{i\omega t_{g-1}}
            s_\alpha^{(g)} P^{(g)\dagger}\left[
                \bar{B}^{(g)}_\alpha \circ I^{(g)}(\omega)
            \right] P^{(g)}

    where

    .. math::

        I^{(g)}_{nm}(\omega) &= \int_0^{t_g - t_{g-1}}\mathrm{d}t\,
                                e^{i(\omega+\omega_n-\omega_m)t} \\
                             &= \frac{e^{i(\omega+\omega_n-\omega_m)
                                (t_g - t_{g-1})} - 1}
                                {i(\omega+\omega_n-\omega_m)}, \\
        \bar{B}_\alpha^{(g)} &= V^{(g)\dagger} B_\alpha V^{(g)}, \\
        P^{(g)} &= V^{(g)\dagger} Q_{g-1},

    and :math:`V^{(g)}` is the matrix of eigenvectors that diagonalizes
    :math:`\tilde{\mathcal{H}}_n^{(g)}`, :math:`B_\alpha` the
    :math:`\alpha`-th noise operator, and  :math:`s_\alpha^{(g)}` the
    noise sensitivity during interval :math:`g`.

    The control matrix then corresponds to the coefficients of expansion
    in an operator basis :math:`\{C_k\}_k`:

    .. math::
        \tilde{\mathcal{B}}_{k\alpha}(\omega) =
            \mathrm{tr}(\tilde{B}_\alpha(\omega) C_k).

    Due to differences in implementation (for performance reasons), the
    axes of the result are transposed compared to the control matrix:

    >>> ctrlmat = calculate_control_matrix_from_scratch(...)
    >>> ctrlmat.shape
    (n_nops, d**2, n_omega)
    >>> noiseops = calculate_noise_operators_from_scratch(...)
    >>> noiseops.shape
    (n_omega, n_nops, d, d)
    >>> ctrlmat_from_noiseops = ff.basis.expand(noiseops, basis)
    >>> np.allclose(ctrlmat, ctrlmat_from_noiseops.transpose(1, 2, 0))
    True

    See Also
    --------
    :func:`calculate_noise_operators_from_atomic`: Compute the operators from atomic segments.
    :func:`calculate_control_matrix_from_scratch`: Same calculation in Liouville space.
    """
    if t is None:
        t = np.concatenate(([0], np.asarray(dt).cumsum()))

    d = eigvecs.shape[-1]
    n_coeffs = np.asarray(n_coeffs)

    # Precompute noise opers transformed to eigenbasis of each pulse
    # segment and Q^\dagger @ V
    eigvecs_propagated = _propagate_eigenvectors(eigvecs, propagators[:-1])
    n_opers_transformed = _transform_noise_operators(n_coeffs, n_opers, eigvecs)

    # Allocate memory
    exp_buf, int_buf = np.empty((2, len(omega), d, d), dtype=complex)
    intermediate = np.empty((len(omega), len(n_opers), d, d), dtype=complex)
    noise_operators = np.zeros((len(omega), len(n_opers), d, d), dtype=complex)

    # Set up reusable expressions
    expr_1 = oe.contract_expression('akl,okl->oakl',
                                    n_opers_transformed[:, 0].shape, int_buf.shape)
    expr_2 = oe.contract_expression('ji,...jk,kl',
                                    eigvecs_propagated[0].shape, intermediate.shape,
                                    eigvecs_propagated[0].shape, optimize=[(0, 1), (0, 1)])

    for g in util.progressbar_range(len(dt), show_progressbar=show_progressbar,
                                    desc='Calculating noise operators'):
        int_buf = _first_order_integral(omega, eigvals[g], dt[g], exp_buf, int_buf)
        intermediate = expr_1(n_opers_transformed[:, g],
                              util.cexp(omega[:, None, None]*t[g])*int_buf, out=intermediate)

        noise_operators += expr_2(eigvecs_propagated[g].conj(), intermediate,
                                  eigvecs_propagated[g])

    return noise_operators


@util.parse_optional_parameters({'which': ('total', 'correlations')})
def calculate_control_matrix_from_atomic(
        phases: ndarray,
        control_matrix_atomic: ndarray,
        propagators_liouville: ndarray,
        show_progressbar: bool = False,
        which: str = 'total'
) -> ndarray:
    r"""
    Calculate the control matrix from the control matrices of atomic
    segments.

    Parameters
    ----------
    phases: array_like, shape (n_dt, n_omega)
        The phase factors for :math:`g\in\{0, 1, \dots, G-1\}`.
    control_matrix_atomic: array_like, shape (n_dt, n_nops, d**2, n_omega)
        The pulse control matrices for :math:`g\in\{1, 2, \dots, G\}`.
    propagators_liouville: array_like, shape (n_dt, n_nops, d**2, d**2)
        The transfer matrices of the cumulative propagators for
        :math:`g\in\{0, 1, \dots, G-1\}`.
    show_progressbar: bool, optional
        Show a progress bar for the calculation.
    which: str, ('total', 'correlations')
        Compute the total control matrix (the sum of all time steps) or
        the correlation control matrix (first axis holds each pulses'
        contribution)

    Returns
    -------
    control_matrix: ndarray, shape ([n_pls,] n_nops, d**2, n_omega)
        The control matrix :math:`\tilde{\mathcal{B}}(\omega)`.

    Notes
    -----
    The control matrix is calculated by evaluating the sum

    .. math::

        \tilde{\mathcal{B}}(\omega) = \sum_{g=1}^G e^{i\omega t_{g-1}}
            \tilde{\mathcal{B}}^{(g)}(\omega)\mathcal{Q}^{(g-1)}.

    See Also
    --------
    calculate_control_matrix_from_scratch: Control matrix from scratch.
    liouville_representation: Liouville representation for a given basis.
    """
    n = len(control_matrix_atomic)
    # Set up a reusable contraction expression. In some cases it is faster to
    # also contract the time dimension in the same expression instead of
    # looping over it, but we don't distinguish here for readability.
    expr = oe.contract_expression('ijo,jk->iko',
                                  control_matrix_atomic.shape[1:],
                                  propagators_liouville.shape[1:])

    # Allocate memory
    if which == 'total':
        control_matrix = np.zeros(control_matrix_atomic.shape[1:], dtype=complex)
        for g in util.progressbar_range(n, show_progressbar=show_progressbar,
                                        desc='Calculating control matrix'):
            control_matrix += expr(phases[g]*control_matrix_atomic[g], propagators_liouville[g])
    else:
        # which == 'correlations'
        control_matrix = np.zeros(control_matrix_atomic.shape, dtype=complex)
        for g in util.progressbar_range(n, show_progressbar=show_progressbar,
                                        desc='Calculating control matrix'):
            control_matrix[g] = expr(phases[g]*control_matrix_atomic[g], propagators_liouville[g])

    return control_matrix


def calculate_control_matrix_from_scratch(
        eigvals: ndarray,
        eigvecs: ndarray,
        propagators: ndarray,
        omega: Coefficients,
        basis: Basis,
        n_opers: Sequence[Operator],
        n_coeffs: Sequence[Coefficients],
        dt: Coefficients,
        t: Optional[Coefficients] = None,
        show_progressbar: bool = False,
        cache_intermediates: bool = False
) -> ndarray:
    r"""
    Calculate the control matrix from scratch, i.e. without knowledge of
    the control matrices of more atomic pulse sequences.

    Parameters
    ----------
    eigvals: array_like, shape (n_dt, d)
        Eigenvalue vectors for each time pulse segment *g* with the
        first axis counting the pulse segment, i.e.
        ``eigvals == array([D_0, D_1, ...])``.
    eigvecs: array_like, shape (n_dt, d, d)
        Eigenvector matrices for each time pulse segment *g* with the
        first axis counting the pulse segment, i.e.
        ``eigvecs == array([V_0, V_1, ...])``.
    propagators: array_like, shape (n_dt+1, d, d)
        The propagators :math:`Q_g = P_g P_{g-1}\cdots P_0` as a (d, d)
        array with *d* the dimension of the Hilbert space.
    omega: array_like, shape (n_omega,)
        Frequencies at which the pulse control matrix is to be
        evaluated.
    basis: Basis, shape (d**2, d, d)
        The basis elements in which the pulse control matrix will be
        expanded.
    n_opers: array_like, shape (n_nops, d, d)
        Noise operators :math:`B_\alpha`.
    n_coeffs: array_like, shape (n_nops, n_dt)
        The sensitivities of the system to the noise operators given by
        *n_opers* at the given time step.
    dt: array_like, shape (n_dt)
        Sequence duration, i.e. for the :math:`g`-th pulse
        :math:`t_g - t_{g-1}`.
    t: array_like, shape (n_dt+1), optional
        The absolute times of the different segments. Can also be
        computed from *dt*.
    show_progressbar: bool, optional
        Show a progress bar for the calculation.
    cache_intermediates: bool, optional
        Keep and return intermediate terms
        :math:`\mathcal{G}^{(g)}(\omega)` of the sum so that
        :math:`\mathcal{R}(\omega)=\sum_g\mathcal{G}^{(g)}(\omega)`.
        Otherwise the sum is performed in-place.

    Returns
    -------
    control_matrix: ndarray, shape (n_nops, d**2, n_omega)
        The control matrix :math:`\tilde{\mathcal{B}}(\omega)`
    intermediates: dict[str, ndarray]
        Intermediate results of the calculation. Only if
        cache_intermediates is True.

    Notes
    -----
    The control matrix is calculated according to

    .. math::

        \tilde{\mathcal{B}}_{\alpha k}(\omega) = \sum_{g=1}^G
            e^{i\omega t_{g-1}} s_\alpha^{(g)}\mathrm{tr}\left(
                [\bar{B}_\alpha^{(g)}\circ I(\omega)] \bar{C}_k^{(g)}
            \right)

    where

    .. math::

        I^{(g)}_{nm}(\omega) &= \int_0^{t_l - t_{g-1}}\mathrm{d}t\,
                                e^{i(\omega+\omega_n-\omega_m)t} \\
                             &= \frac{e^{i(\omega+\omega_n-\omega_m)
                                (t_l - t_{g-1})} - 1}
                                {i(\omega+\omega_n-\omega_m)}, \\
        \bar{B}_\alpha^{(g)} &= V^{(g)\dagger} B_\alpha V^{(g)}, \\
        \bar{C}_k^{(g)} &= V^{(g)\dagger} Q_{g-1} C_k Q_{g-1}^\dagger V^{(g)},

    and :math:`V^{(g)}` is the matrix of eigenvectors that diagonalizes
    :math:`\tilde{\mathcal{H}}_n^{(g)}`, :math:`B_\alpha` the
    :math:`\alpha`-th noise operator :math:`s_\alpha^{(g)}` the noise
    sensitivity during interval :math:`g`, and :math:`C_k` the
    :math:`k`-th basis element.

    See Also
    --------
    calculate_control_matrix_from_atomic: Control matrix from concatenation.
    """
    d = eigvecs.shape[-1]

    if t is None:
        t = np.concatenate(([0], np.asarray(dt).cumsum()))

    # Precompute noise opers transformed to eigenbasis of each pulse segment
    # and Q^\dagger @ V
    eigvecs_propagated = _propagate_eigenvectors(propagators[:-1], eigvecs)
    n_opers_transformed = _transform_noise_operators(n_coeffs, n_opers, eigvecs)

    # Allocate result and buffers for intermediate arrays
    control_matrix = np.zeros((len(n_opers), len(basis), len(omega)), dtype=complex)
    exp_buf, int_buf = np.empty((2, len(omega), d, d), dtype=complex)

    if cache_intermediates:
        basis_transformed_cache = np.empty((len(dt), *basis.shape), dtype=complex)
        sum_cache = np.empty((len(dt), len(n_opers), len(basis), len(omega)), dtype=complex)
    else:
        basis_transformed = np.empty(basis.shape, dtype=complex)
        sum_buf = np.empty((len(n_opers), len(basis), len(omega)), dtype=complex)

    # Optimize the contraction path dynamically since it differs for different
    # values of d
    expr = oe.contract_expression('o,jmn,omn,knm->jko',
                                  omega.shape, n_opers_transformed[:, 0].shape,
                                  int_buf.shape, basis.shape, optimize=True)
    for g in util.progressbar_range(len(dt), show_progressbar=show_progressbar,
                                    desc='Calculating control matrix'):

        if cache_intermediates:
            # Assign references to the locations in the cache for the quantities
            # that should be stored
            basis_transformed = basis_transformed_cache[g]
            sum_buf = sum_cache[g]

        basis_transformed = _transform_basis(basis, eigvecs_propagated[g], out=basis_transformed)
        int_buf = _first_order_integral(omega, eigvals[g], dt[g], exp_buf, int_buf)
        sum_buf = expr(util.cexp(omega*t[g]), n_opers_transformed[:, g], int_buf,
                       basis_transformed, out=sum_buf)

        control_matrix += sum_buf

    if cache_intermediates:
        intermediates = dict(n_opers_transformed=n_opers_transformed,
                             basis_transformed=basis_transformed_cache,
                             control_matrix_step=sum_cache)
        return control_matrix, intermediates

    return control_matrix


def calculate_control_matrix_periodic(phases: ndarray, control_matrix: ndarray,
                                      total_propagator_liouville: ndarray,
                                      repeats: int) -> ndarray:
    r"""
    Calculate the control matrix of a periodic pulse given the phase
    factors, control matrix and transfer matrix of the total propagator,
    total_propagator_liouville, of the atomic pulse.

    Parameters
    ----------
    phases: ndarray, shape (n_omega,)
        The phase factors :math:`e^{i\omega T}` of the atomic pulse.
    control_matrix: ndarray, shape (n_nops, d**2, n_omega)
        The control matrix :math:`\tilde{\mathcal{B}}^{(1)}(\omega)` of
        the atomic pulse.
    total_propagator_liouville: ndarray, shape (d**2, d**2)
        The transfer matrix :math:`\mathcal{Q}^{(1)}` of the atomic
        pulse.
    repeats: int
        The number of repetitions.

    Returns
    -------
    control_matrix: ndarray, shape (n_nops, d**2, n_omega)
        The control matrix :math:`\tilde{\mathcal{B}}(\omega)` of the
        repeated pulse.

    Notes
    -----
    The control matrix is computed as

    .. math::

        \tilde{\mathcal{B}}(\omega)
                            &= \tilde{\mathcal{B}}^{(1)}(\omega)
                               \sum_{g=0}^{G-1}
                               \left(e^{i\omega T}\right)^g \\
                            &= \tilde{\mathcal{B}}^{(1)}(\omega)\bigl(
                               \mathbb{I} - e^{i\omega T}
                               \mathcal{Q}^{(1)}\bigr)^{-1}\bigl(
                               \mathbb{I} - \bigl(e^{i\omega T}
                               \mathcal{Q}^{(1)}\bigr)^G\bigr).

    with :math:`G` the number of repetitions.
    """
    # Compute the finite geometric series \sum_{g=0}^{G-1} T^g. First check if
    # inv(I - T) is 'good', i.e. if inv(I - T) @ (I - T) == I, since NumPy will
    # compute the inverse in any case. For those frequencies where the inverse
    # is well-behaved, evaluate the sum as a Neumann series and for the rest
    # evaluate it explicitly.
    eye = np.eye(total_propagator_liouville.shape[0])
    T = np.multiply.outer(phases, total_propagator_liouville)

    # Mask the invertible frequencies. The chosen atol is somewhat empiric.
    M = eye - T
    M_inv = nla.inv(M)
    good_inverse = np.isclose(M_inv @ M, eye, atol=1e-10, rtol=0).all((1, 2))

    # Allocate memory
    S = np.empty((*phases.shape, *total_propagator_liouville.shape), dtype=complex)
    # Evaluate the sum for invertible frequencies
    S[good_inverse] = M_inv[good_inverse] @ (eye - nla.matrix_power(T[good_inverse], repeats))

    # Evaluate the sum for non-invertible frequencies
    # HINT: Using numba, this could be a factor of ten or so faster. But since
    # usually very few omega-values have a bad inverse, the compilation
    # overhead is not compensated.
    if (~good_inverse).any():
        S[~good_inverse] = eye + sum(accumulate(repeat(T[~good_inverse], repeats-1), np.matmul))

    # Multiply with control_matrix_at to get the final control matrix
    control_matrix_tot = (control_matrix.transpose(2, 0, 1) @ S).transpose(1, 2, 0)
    return control_matrix_tot


@util.parse_optional_parameters({'which': ('total', 'correlations')})
def calculate_cumulant_function(
        pulse: 'PulseSequence',
        spectrum: Optional[ndarray] = None,
        omega: Optional[Coefficients] = None,
        n_oper_identifiers: Optional[Sequence[str]] = None,
        which: str = 'total',
        decay_amplitudes: Optional[ndarray] = None,
        show_progressbar: bool = False,
        memory_parsimonious: bool = False
) -> ndarray:
    r"""Calculate the cumulant function :math:`\mathcal{K}(\tau)`.

    The error transfer matrix is obtained from the cumulant function by
    exponentiation,
    :math:`\langle\tilde{\mathcal{U}}\rangle = \exp\mathcal{K}(\tau)`.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to compute the cumulant
        function.
    spectrum: array_like, shape ([[n_nops,] n_nops,] n_omega), optional
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
    omega: array_like, shape (n_omega,), optional
        The frequencies at which to evaluate the filter functions.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which to evaluate the
        cumulant function. The default is all.
    which: str, optional
        Which decay amplitudes should be calculated, may be either
        'total' (default) or 'correlations'. See :func:`infidelity` and
        :ref:`Notes <notes>`.
    decay_amplitudes, array_like, shape ([[n_pls, n_pls,] n_nops,] n_nops, d**2, d**2), optional
        A precomputed cumulant function. If given, *spectrum*, *omega*
        are not required.
    show_progressbar: bool, optional
        Show a progress bar for the calculation of the control matrix.
    memory_parsimonious: bool, optional
        Trade memory footprint for performance. See
        :func:`~numeric.calculate_decay_amplitudes`. The default is
        ``False``.

    Returns
    -------
    cumulant_function: ndarray, shape ([[n_pls, n_pls,] n_nops,] n_nops, d**2, d**2)
        The cumulant function. The individual noise operator
        contributions chosen by ``n_oper_identifiers`` are on the third
        to last axis / axes, depending on whether the noise is
        cross-correlated or not. If ``which == 'correlations'``, the
        first two axes correspond to the contributions of the pulses in
        the sequence.

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

    Here, :math:`T_{ijkl} = \mathrm{tr}(C_i C_j C_k C_l)` is a trivial
    function of the basis elements :math:`C_i`, and
    :math:`\Gamma_{\alpha\beta,kl}` and :math:`\Delta_{\alpha\beta,kl}`
    are the decay amplitudes and frequency shifts which correspond to
    first and second order in the Magnus expansion, respectively. Since
    the latter induce coherent errors, we can approximately neglect them
    if we assume that the pulse has been experimentally calibrated.

    For a single qubit and represented in the Pauli basis, the above
    reduces to

    .. math::

        K_{\alpha\beta,ij}(\tau) = \begin{cases}
            - \sum_{k\neq i}\Gamma_{\alpha\beta,kk}
                &\quad\mathrm{if}\: i = j,   \\
            - \Delta_{\alpha\beta,ij} + \Delta_{\alpha\beta,ji}
            + \Gamma_{\alpha\beta,ij}
                &\quad\mathrm{if}\: i\neq j,
        \end{cases}

    for :math:`i\in\{1,2,3\}` and :math:`K_{0j} = K_{i0} = 0`.

    Lastly, the pulse correlation cumulant function resolves
    correlations in the cumulant function of a sequence of pulses
    :math:`g = 1,\dotsc,G` such that the following holds:

    .. math::

        K_{\alpha\beta,ij}(\tau) = \sum_{g,g'=1}^G
             K_{\alpha\beta,ij}^{(gg')}(\tau).

    See Also
    --------
    calculate_decay_amplitudes: Calculate the :math:`\Gamma_{\alpha\beta,kl}`
    error_transfer_matrix: Calculate the error transfer matrix :math:`\exp\mathcal{K}`.
    infidelity: Calculate only infidelity of a pulse.
    pulse_sequence.concatenate: Concatenate ``PulseSequence`` objects.
    calculate_pulse_correlation_filter_function

    """
    N, d = pulse.basis.shape[:2]
    if decay_amplitudes is None:
        if spectrum is None and omega is None:
            raise ValueError('Require either spectrum and frequencies or precomputed ' +
                             'decay amplitudes')

        decay_amplitudes = calculate_decay_amplitudes(pulse, spectrum, omega, n_oper_identifiers,
                                                      which, show_progressbar, memory_parsimonious)

    if d == 2 and pulse.basis.btype in ('Pauli', 'GGM'):
        # Single qubit case. Can use simplified expression
        cumulant_function = np.zeros(decay_amplitudes.shape)
        diag_mask = np.eye(N, dtype=bool)

        # Offdiagonal terms
        cumulant_function[..., ~diag_mask] = decay_amplitudes[..., ~diag_mask]

        # Diagonal terms K_ii given by sum over diagonal of Gamma excluding
        # Gamma_ii. Since the Pauli basis is traceless, K_00 is zero, therefore
        # start at K_11.
        diag_idx = deque((True, False, True, True))
        for i in range(1, N):
            cumulant_function[..., i, i] = - decay_amplitudes[..., diag_idx, diag_idx].sum(axis=-1)
            # shift the item not summed over by one
            diag_idx.rotate()
    else:
        # Multi qubit case. Use general expression.
        traces = pulse.basis.four_element_traces
        cumulant_function = - (
            oe.contract('...kl,klji->...ij', decay_amplitudes, traces, backend='sparse') -
            oe.contract('...kl,kjli->...ij', decay_amplitudes, traces, backend='sparse') -
            oe.contract('...kl,kilj->...ij', decay_amplitudes, traces, backend='sparse') +
            oe.contract('...kl,kijl->...ij', decay_amplitudes, traces, backend='sparse')
        ) / 2

    return cumulant_function.real


@util.parse_optional_parameters({'which': ('total', 'correlations')})
def calculate_decay_amplitudes(
        pulse: 'PulseSequence',
        spectrum: ndarray,
        omega: Coefficients,
        n_oper_identifiers: Optional[Sequence[str]] = None,
        which: str = 'total',
        show_progressbar: bool = False,
        memory_parsimonious: bool = False
) -> ndarray:
    r"""
    Get the decay amplitudes :math:`\Gamma_{\alpha\beta, kl}` for noise
    sources :math:`\alpha,\beta` and basis elements :math:`k,l`.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to compute the decay
        amplitudes.
    spectrum: array_like, shape ([[n_nops,] n_nops,] n_omega)
        The two-sided noise power spectral density. If 1-d, the same
        spectrum is used for all noise operators. If 2-d, one (self-)
        spectrum for each noise operator is expected. If 3-d, should be
        a matrix of cross-spectra such that
        ``spectrum[i, j] == spectrum[j, i].conj()``.
    omega: array_like,
        The frequencies at which to calculate the filter functions.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which to calculate
        the decay amplitudes. The default is all.
    which: str, optional
        Which decay amplitudes should be calculated, may be either
        'total' (default) or 'correlations'. See :func:`infidelity` and
        :ref:`Notes <notes>`.
    show_progressbar: bool, optional
        Show a progress bar for the calculation.
    memory_parsimonious: bool, optional
        For large dimensions, the integrand

        .. math::

            \tilde{\mathcal{B}}^\ast_{\alpha k}(\omega)
            S_{\alpha\beta}(\omega)\tilde{\mathcal{B}}_{\beta l}(\omega)

        can consume quite a large amount of memory if set up for all
        :math:`\alpha,\beta,k,l` at once. If ``True``, it is only set up
        and integrated for a single :math:`k` at a time and looped over.
        This is slower but requires much less memory. The default is
        ``False``.

    Raises
    ------
    ValueError
        If spectrum has incompatible shape.

    Returns
    -------
    decay_amplitudes: ndarray, shape ([[n_pls, n_pls,] n_nops,] n_nops, d**2, d**2)
        The decay amplitudes.

    .. _notes:

    Notes
    -----
    The total decay amplitudes are given by

    .. math::

        \Gamma_{\alpha\beta, kl} = \int\frac{\mathrm{d}\omega}{2\pi}
            \tilde{\mathcal{B}}^\ast_{\alpha k}(\omega)
            S_{\alpha\beta}(\omega)\tilde{\mathcal{B}}_{\beta l}(\omega).

    If pulse correlations are taken into account, they are given by

    .. math::

        \Gamma_{\alpha\beta, kl}^{(gg')} = \int
            \frac{\mathrm{d}\omega}{2\pi} S_{\alpha\beta}(\omega)
            F_{\alpha\beta, kl}^{(gg')}(\omega).

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
        if pulse.is_cached('filter_function_gen'):
            control_matrix = None
            filter_function = pulse.get_filter_function(omega, which='generalized')
        else:
            control_matrix = pulse.get_control_matrix(omega, show_progressbar)
            filter_function = None
    else:
        # which == 'correlations'
        if pulse.is_cached('omega'):
            if not np.array_equal(pulse.omega, omega):
                raise ValueError('Pulse correlation decay amplitudes requested but omega not ' +
                                 'equal to cached frequencies.')

        if pulse.is_cached('filter_function_pc_gen'):
            control_matrix = None
            filter_function = pulse.get_pulse_correlation_filter_function(which='generalized')
        else:
            control_matrix = pulse.get_pulse_correlation_control_matrix()
            filter_function = None

    if not memory_parsimonious:
        integrand = _get_integrand(spectrum, omega, idx, which, 'generalized',
                                   control_matrix=control_matrix,
                                   filter_function=filter_function)
        decay_amplitudes = integrate.trapz(integrand, omega, axis=-1)/(2*np.pi)
        return decay_amplitudes.real

    # Conserve memory by looping. Let _get_integrand determine the shape
    if control_matrix is not None:
        n_kl = control_matrix.shape[-2]
        integrand = _get_integrand(spectrum, omega, idx, which, 'generalized',
                                   control_matrix=[control_matrix[..., 0:1, :], control_matrix],
                                   filter_function=filter_function)
    else:
        n_kl = filter_function.shape[-2]
        integrand = _get_integrand(spectrum, omega, idx, which, 'generalized',
                                   control_matrix=control_matrix,
                                   filter_function=filter_function[..., 0:1, :, :])

    decay_amplitudes = np.zeros(integrand.shape[:-3] + (n_kl,)*2, dtype=integrand.dtype)
    decay_amplitudes[..., 0:1, :] = integrate.trapz(integrand, omega, axis=-1)/(2*np.pi)

    for k in util.progressbar_range(1, n_kl, show_progressbar=show_progressbar,
                                    desc='Integrating'):
        if control_matrix is not None:
            integrand = _get_integrand(
                spectrum, omega, idx, which, 'generalized',
                control_matrix=[control_matrix[..., k:k+1, :], control_matrix],
                filter_function=filter_function
            )
        else:
            integrand = _get_integrand(spectrum, omega, idx, which, 'generalized',
                                       control_matrix=control_matrix,
                                       filter_function=filter_function[..., k:k+1, :, :])

        decay_amplitudes[..., k:k+1, :] = integrate.trapz(integrand, omega, axis=-1)/(2*np.pi)

    return decay_amplitudes.real


@util.parse_which_FF_parameter
def calculate_filter_function(control_matrix: ndarray, which: str = 'fidelity') -> ndarray:
    r"""Compute the filter function from the control matrix.

    Parameters
    ----------
    control_matrix: array_like, shape (n_nops, d**2, n_omega)
        The control matrix.
    which : str, optional
        Which filter function to return. Either 'fidelity' (default) or
        'generalized' (see :ref:`Notes <notes>`).

    Returns
    -------
    filter_function: ndarray, shape (n_nops, n_nops, [d**2, d**2,] n_omega)
        The filter functions for each noise operator correlation. The
        diagonal corresponds to the filter functions for uncorrelated
        noise sources.

    .. _notes:

    Notes
    -----
    The generalized filter function is given by

    .. math::

        F_{\alpha\beta,kl}(\omega) =
            \tilde{\mathcal{B}}_{\alpha k}^\ast(\omega)
            \tilde{\mathcal{B}}_{\beta l}(\omega),

    where :math:`\alpha,\beta` are indices counting the noise operators
    :math:`B_\alpha` and :math:`k,l` indices counting the basis elements
    :math:`C_k`.

    The fidelity filter function is obtained by tracing over the basis
    indices:

    .. math::

        F_{\alpha\beta}(\omega) = \sum_{k} F_{\alpha\beta,kk}(\omega).

    See Also
    --------
    calculate_control_matrix_from_scratch: Control matrix from scratch.
    calculate_control_matrix_from_atomic: Control matrix from concatenation.
    calculate_pulse_correlation_filter_function: Pulse correlations.
    """
    if which == 'fidelity':
        subscripts = 'ako,bko->abo'
    else:
        # which == 'generalized'
        subscripts = 'ako,blo->abklo'

    return np.einsum(subscripts, control_matrix.conj(), control_matrix)


@util.parse_which_FF_parameter
def calculate_pulse_correlation_filter_function(control_matrix: ndarray,
                                                which: str = 'fidelity') -> ndarray:
    r"""Compute pulse correlation filter function from control matrix.

    Parameters
    ----------
    control_matrix: array_like, shape (n_pulses, n_nops, d**2, n_omega)
        The control matrix.
    which : str, optional
        Which filter function to return. Either 'fidelity' (default) or
        'generalized' (see :ref:`Notes <notes>`).

    Returns
    -------
    filter_function_pc: ndarray, shape (n_pls, n_pls, n_nops, n_nops, [d**2, d**2,] n_omega)
        The pulse correlation filter functions for each pulse and noise
        operator correlations. The first two axes hold the pulse
        correlations, the second two the noise correlations.
    which : str, optional
        Which filter function to return. Either 'fidelity' (default) or
        'generalized' (see :ref:`Notes <notes>`).

    .. _notes:

    Notes
    -----
    The generalized pulse correlation filter function is given by

    .. math::

        F_{\alpha\beta,kl}^{(gg')}(\omega) = \bigl[
            \mathcal{Q}^{(g'-1)\dagger}
            \tilde{\mathcal{B}}^{(g')\dagger}(\omega)
        \bigr]_{k\alpha} \bigl[
            \tilde{\mathcal{B}}^{(g)}(\omega)\mathcal{Q}^{(g-1)}
        \bigr]_{\beta l} e^{i\omega(t_{g-1} - t_{g'-1})},

    with :math:`\tilde{\mathcal{B}}^{(g)}` the control matrix of the
    :math:`g`-th pulse. The fidelity pulse correlation function is
    obtained by tracing out the basis indices,

    .. math::

        F_{\alpha\beta}^{(gg')}(\omega) =
          \sum_{k} F_{\alpha\beta,kk}^{(gg')}(\omega)

    See Also
    --------
    calculate_control_matrix_from_scratch: Control matrix from scratch.
    calculate_control_matrix_from_atomic: Control matrix from concatenation.
    calculate_filter_function: Regular filter function.
    """
    if control_matrix.ndim != 4:
        raise ValueError('Expected control_matrix.ndim == 4.')

    if which == 'fidelity':
        subscripts = 'gako,hbko->ghabo'
    else:
        # which == 'generalized'
        subscripts = 'gako,hblo->ghabklo'

    return np.einsum(subscripts, control_matrix.conj(), control_matrix)


def diagonalize(H: ndarray, dt: Coefficients) -> Tuple[ndarray]:
    r"""Diagonalize a Hamiltonian.

    Diagonalize the Hamiltonian *H* which is piecewise constant during
    the times given by *dt* and return eigenvalues, eigenvectors, and
    the cumulative propagators :math:`Q_l`. Note that we calculate in
    units where :math:`\hbar\equiv 1` so that

    .. math::

        U(t, t_0) = \mathcal{T}\exp\left(
                        -i\int_{t_0}^t\mathrm{d}t'\mathcal{H}(t')
                    \right).

    Parameters
    ----------
    H: array_like, shape (n_dt, d, d)
        Hamiltonian of shape (n_dt, d, d) with d the dimensionality of
        the system
    dt: array_like
        The time differences

    Returns
    -------
    eigvals: ndarray
        Array of eigenvalues of shape (n_dt, d)
    eigvecs: ndarray
        Array of eigenvectors of shape (n_dt, d, d)
    propagators: ndarray
        Array of cumulative propagators of shape (n_dt+1, d, d)
    """
    d = H.shape[-1]
    # Calculate Eigenvalues and -vectors
    eigvals, eigvecs = nla.eigh(H)
    # Propagator P = V exp(-j D dt) V^\dag. Middle term is of shape
    # (d, n_dt) due to transpose, so switch around indices in einsum
    # instead of transposing again. Same goes for the last term. This saves
    # a bit of time. The following is faster for larger dimensions but not for
    # many time steps:
    # P = np.empty((500, 4, 4), dtype=complex)
    # for l, (V, D) in enumerate(zip(eigvecs, np.exp(-1j*dt*eigvals.T).T)):
    #     P[l] = (V * D) @ V.conj().T
    P = np.einsum('lij,jl,lkj->lik', eigvecs, util.cexp(-np.asarray(dt)*eigvals.T), eigvecs.conj())
    # The cumulative propagator Q with the identity operator as first
    # element (Q_0 = P_0 = I), i.e.
    # Q = [Q_0, Q_1, ..., Q_n] = [P_0, P_1 @ P_0, ..., P_n @ ... @ P_0]
    Q = np.empty((len(dt)+1, d, d), dtype=complex)
    Q[0] = np.identity(d)
    for i in range(len(dt)):
        Q[i+1] = P[i] @ Q[i]

    return eigvals, eigvecs, Q


def error_transfer_matrix(
        pulse: Optional['PulseSequence'] = None,
        spectrum: Optional[ndarray] = None,
        omega: Optional[Coefficients] = None,
        n_oper_identifiers: Optional[Sequence[str]] = None,
        cumulant_function: Optional[ndarray] = None,
        show_progressbar: bool = False,
        memory_parsimonious: bool = False
) -> ndarray:
    r"""Compute the error transfer matrix up to unitary rotations.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to compute the error
        transfer matrix.
    spectrum: array_like, shape ([[n_nops,] n_nops,] n_omega)
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
        The frequencies at which to calculate the filter functions.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which to evaluate the
        error transfer matrix. The default is all. Note that, since in
        general contributions from different noise operators won't
        commute, not selecting all noise operators results in neglecting
        terms of order :math:`\xi^4`.
    cumulant_function: ndarray, shape ([[n_pls, n_pls,] n_nops,] n_nops, d**2, d**2), optional
        A precomputed cumulant function. If given, *pulse*, *spectrum*,
        *omega* are not required.
    show_progressbar: bool, optional
        Show a progress bar for the calculation of the control matrix.
    memory_parsimonious: bool, optional
        Trade memory footprint for performance. See
        :func:`~numeric.calculate_decay_amplitudes`. The default is
        ``False``.

    Returns
    -------
    error_transfer_matrix: ndarray, shape (d**2, d**2)
        The error transfer matrix. The individual noise operator
        contributions are summed up before exponentiating as they might
        not commute.

    Notes
    -----
    The error transfer matrix is given by

    .. math::

        \tilde{\mathcal{U}} = \exp\mathcal{K}(\tau)

    with :math:`\mathcal{K}(\tau)` the cumulant function (see
    :func:`calculate_cumulant_function`). For Gaussian noise this
    expression is exact when taking into account the decay amplitudes
    :math:`\Gamma` and frequency shifts :math:`\Delta`. As the latter
    effects coherent errors it can be neglected if we assume that the
    experimenter has calibrated their pulse.

    For non-Gaussian noise the expression above is perturbative and
    includes noise up to order :math:`\xi^2` and hence
    :math:`\tilde{\mathcal{U}} = \mathbb{1} + \mathcal{K}(\tau) +
    \mathcal{O}(\xi^2)`
    (although it is evaluated as a matrix exponential in any case).

    Given the above expression of the error transfer matrix, the
    entanglement fidelity is given by


    .. math::

        \mathcal{F}_\mathrm{e} =
            \frac{1}{d^2}\mathrm{tr}\,\tilde{\mathcal{U}}.

    See Also
    --------
    calculate_cumulant_function: Calculate the cumulant function :math:`\mathcal{K}`
    calculate_decay_amplitudes: Calculate the :math:`\Gamma_{\alpha\beta,kl}`
    infidelity: Calculate only infidelity of a pulse.
    """
    if cumulant_function is None:
        if pulse is None or spectrum is None or omega is None:
            raise ValueError('Require either precomputed cumulant function ' +
                             'or pulse, spectrum, and omega as arguments.')

        cumulant_function = calculate_cumulant_function(pulse, spectrum, omega,
                                                        n_oper_identifiers=n_oper_identifiers,
                                                        which='total',
                                                        show_progressbar=show_progressbar,
                                                        memory_parsimonious=memory_parsimonious)

    try:
        error_transfer_matrix = sla.expm(
            cumulant_function.sum(axis=tuple(range(cumulant_function.ndim - 2)))
        )
    except AttributeError as aerr:
        raise TypeError(f'cumulant_function invalid type: {type(cumulant_function)}') from aerr
    except ValueError as verr:
        raise ValueError(f'cumulant_function invalid shape: {cumulant_function.shape}') from verr

    return error_transfer_matrix


@util.parse_optional_parameters({'which': ('total', 'correlations')})
def infidelity(pulse: 'PulseSequence', spectrum: Union[Coefficients, Callable],
               omega: Union[Coefficients, Dict[str, Union[int, str]]],
               n_oper_identifiers: Optional[Sequence[str]] = None,
               which: str = 'total', return_smallness: bool = False,
               test_convergence: bool = False) -> Union[ndarray, Any]:
    r"""Calculate the leading order entanglement infidelity.

    This function calculates the infidelity approximately from the
    leading peturbation (see :ref:`Notes <notes>`). To compute it
    exactly for Gaussian noise and vanishing coherent errors (second
    order Magnus terms), use :func:`error_transfer_matrix` to obtain it
    from the full process matrix.

    Parameters
    ----------
    pulse: PulseSequence
        The ``PulseSequence`` instance for which to calculate the
        infidelity for.
    spectrum: array_like, shape ([[n_nops,] n_nops,] omega) or callable
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

        If *test_convergence* is ``True``, a function handle to
        compute the power spectral density from a sequence of
        frequencies is expected.
    omega: array_like or dict
        The frequencies at which the integration is to be carried out.
        If *test_convergence* is ``True``, a dict with possible keys
        ('omega_IR', 'omega_UV', 'spacing', 'n_min', 'n_max',
        'n_points'), where all entries are integers except for
        ``spacing`` which should be a string, either 'linear' or 'log'.
        'n_points' controls how many steps are taken.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which to calculate
        the infidelity  contribution. If given, the infidelities for
        each noise operator will be returned. Otherwise, all noise
        operators will be taken into account.
    which: str, optional
        Which infidelities should be calculated, may be either 'total'
        (default) or 'correlations'. In the former case, one value is
        returned for each noise operator, corresponding to the total
        infidelity of the pulse (or pulse sequence). In the latter, an
        array of infidelities is returned where element (i,j)
        corresponds to the infidelity contribution of the correlations
        between pulses i and j (see :ref:`Notes <notes>`). Note that
        this option is only available if the pulse correlation filter
        functions have been computed during concatenation (see
        :func:`calculate_pulse_correlation_filter_function` and
        :func:`~filter_functions.pulse_sequence.concatenate`).
    return_smallness: bool, optional
        Return the smallness parameter :math:`\xi` for the given
        spectrum.
    test_convergence: bool, optional
        Test the convergence of the integral with respect to the number
        of frequency samples. Returns the number of frequency samples
        and the corresponding fidelities. See *spectrum* and *omega* for
        more information.

    Returns
    -------
    infid: ndarray, shape ([[n_pls, n_pls,], n_nops,] n_nops)
        Array with the infidelity contributions for each spectrum
        *spectrum* on the last axis or axes, depending on the shape of
        *spectrum* and *which*. If ``which`` is ``correlations``, the
        first two axes are the individual pulse contributions. If
        *spectrum* is 2-d (3-d), the last axis (two axes) are the
        individual spectral contributions. Only if *test_convergence* is
        ``False``.
    n_samples: array_like
        Array with number of frequency samples used for convergence
        test. Only if *test_convergence* is ``True``.
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
            &= \frac{1}{d}\int_{-\infty}^{\infty}
                \frac{\mathrm{d}\omega}{2\pi}S_{\alpha\beta}(\omega)
                F_{\alpha\beta}(\omega) + \mathcal{O}\big(\xi^4\big) \\
            &= \sum_{g,g'=1}^G \mathcal{I}_{\alpha\beta}^{(gg')}

    with :math:`S_{\alpha\beta}(\omega)` the two-sided noise spectral
    density and :math:`F_{\alpha\beta}(\omega)` the first-order filter
    function for noise sources :math:`\alpha,\beta`. The noise spectrum
    may include correlated noise sources, that is, its entry at
    :math:`(\alpha,\beta)` corresponds to the correlations between
    sources :math:`\alpha` and :math:`\beta`.
    :math:`\mathcal{I}_{\alpha\beta}^{(gg')}` are the correlation
    infidelities that can be computed by setting
    ``which='correlations'``.


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
                    S_\alpha(\omega)\left(\sum_gs_\alpha^{(g)}\Delta t_g
                    \right)^2
                \right].

    Note that in practice, the integral is only evaluated on the
    interval :math:`\omega\in[\omega_\mathrm{min},\omega_\mathrm{max}]`.

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
        Physical Review A - Atomic, Molecular, and Optical Physics,
        60(3), 1888–1898. https://doi.org/10.1103/PhysRevA.60.1888

    .. [Nie02]
        Nielsen, M. A. (2002). A simple formula for the average gate
        fidelity of a quantum dynamical operation. Physics Letters,
        Section A: General, Atomic and Solid State Physics, 303(4),
        249–252. https://doi.org/10.1016/S0375-9601(02)01272-0

    See Also
    --------
    error_transfer_matrix: Calculate the full process matrix.
    plotting.plot_infidelity_convergence: Convenience function to plot results.
    """
    # Noise operator indices
    idx = util.get_indices_from_identifiers(pulse, n_oper_identifiers, 'noise')

    if test_convergence:
        if not callable(spectrum):
            raise TypeError('Spectrum should be callable when test_convergence == True.')

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
            convergence_infids[i] = infidelity(pulse, spectrum(freqs), freqs,
                                               n_oper_identifiers=n_oper_identifiers,
                                               which='total', return_smallness=False,
                                               test_convergence=False)

        return n_samples, convergence_infids

    if which == 'total':
        if not pulse.basis.istraceless:
            # Fidelity not simply sum of diagonal of decay amplitudes Gamma_kk
            # but trace tensor plays a role, cf eq. (39). For traceless bases,
            # the trace tensor term reduces to delta_ij.
            T = pulse.basis.four_element_traces
            Tp = (sparse.diagonal(T, axis1=2, axis2=3).sum(-1) -
                  sparse.diagonal(T, axis1=1, axis2=3).sum(-1)).todense()

            control_matrix = pulse.get_control_matrix(omega)
            filter_function = np.einsum('ako,blo,kl->abo',
                                        control_matrix.conj(), control_matrix, Tp)/pulse.d
        else:
            filter_function = pulse.get_filter_function(omega)
    else:
        # which == 'correlations'
        if not pulse.basis.istraceless:
            warn('Calculating pulse correlation fidelities with non-' +
                 'traceless basis. The results will be off.')

        if pulse.is_cached('omega'):
            if not np.array_equal(pulse.omega, omega):
                raise ValueError('Pulse correlation infidelities requested ' +
                                 'but omega not equal to cached frequencies.')

        filter_function = pulse.get_pulse_correlation_filter_function()

    spectrum = np.asarray(spectrum)
    slices = [slice(None)]*filter_function.ndim
    if spectrum.ndim == 3:
        slices[-3] = idx[:, None]
        slices[-2] = idx[None, :]
    else:
        slices[-3] = idx
        slices[-2] = idx

    integrand = _get_integrand(spectrum, omega, idx, which, 'fidelity',
                               filter_function=filter_function)
    infid = integrate.trapz(integrand, omega).real/(2*np.pi*pulse.d)

    if return_smallness:
        if spectrum.ndim > 2:
            raise NotImplementedError('Smallness parameter only implemented ' +
                                      'for uncorrelated noise sources')

        T1 = integrate.trapz(spectrum, omega)/(2*np.pi)
        T2 = (pulse.dt*pulse.n_coeffs[idx]).sum(axis=-1)**2
        T3 = util.abs2(pulse.n_opers[idx]).sum(axis=(1, 2))
        xi = np.sqrt((T1*T2*T3).sum())

        return infid, xi

    return infid

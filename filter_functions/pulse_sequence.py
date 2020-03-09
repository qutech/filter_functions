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
This module defines the PulseSequence class, the central object of the
formalism.

Classes
-------
:class:`PulseSequence`
    The pulse sequence defined by a Hamiltonian

Functions
---------
:func:`concatenate`
    Function to concatenate different ``PulseSequence`` instances and
    efficiently compute their joint filter function
:func:`concatenate_periodic`
    Function to more efficiently concatenate many versions of the same
    ``PulseSequence`` instances and compute their joint filter function
:func:`extend`
    Function to map several ``PulseSequence`` instances to different qubits,
    efficiently scaling up cached attributes.
"""

import bisect
from copy import copy
from itertools import accumulate, compress, zip_longest
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, Union)
from warnings import warn

import numpy as np
from numpy import linalg, ndarray
from qutip import Qobj

from .basis import (Basis, equivalent_pauli_basis_elements,
                    remap_pauli_basis_elements)
from .numeric import (
    calculate_control_matrix_from_atomic,
    calculate_control_matrix_from_scratch, calculate_control_matrix_periodic,
    calculate_filter_function, calculate_pulse_correlation_filter_function,
    diagonalize, liouville_representation)
from .types import Coefficients, Hamiltonian, Operator, PulseMapping
from .util import (CalculationError, all_array_equal, cexp,
                   get_indices_from_identifiers, hash_array_along_axis, mdot,
                   tensor, tensor_insert, tensor_merge, tensor_transpose)

__all__ = ['PulseSequence', 'concatenate', 'concatenate_periodic', 'extend',
           'remap']


class PulseSequence:
    r"""
    A class for pulse sequences and their filter functions.

    The Hamiltonian is separated into a control and a noise part with

    .. math::

        \mathcal{H}_c &= \sum_i a_i(t) A_i \\
        \mathcal{H}_n &= \sum_j s_j(t) b_j(t) B_j

    where :math:`A_i` and :math:`B_j` are hermitian operators and
    :math:`b_j(t)` are classically fluctuating noise variables captured in a
    power spectral density and not needed at instantiation of a
    ``PulseSequence``.

    Parameters
    ----------
    H_c : list of lists
        A nested list of *n_cops* nested lists as taken by QuTiP functions
        (see for example :func:`~qutip.propagator`) describing the control
        part of the Hamiltonian. The *i*-th entry of the list should be a
        list consisting of the *i*-th operator :math:`A_i` making up the
        control Hamiltonian and a list or array :math:`a_i(t)` describing the
        magnitude of that operator during the time intervals *dt*. Optionally,
        the list may also include operator identifiers. That is, *H_c* should
        look something like this::

            H = [[c_oper1, c_coeff1, c_oper_identifier1],
                 [c_oper2, c_coeff2, c_oper_identifier2], ...]

        The operators may be given either as NumPy arrays or QuTiP Qobjs
        and each coefficient array should have the same number of elements
        as *dt*, and should be given in units of :math:`\hbar`. If not every
        sublist (read: operator) was given a identifier, they are automatically
        filled up with 'A_i' where i is the position of the operator.

    H_n : list of lists
        A nested list of *n_nops* nested lists as taken by QuTiP functions
        (see for example :func:`~qutip.propagator`) describing the noise
        part of the Hamiltonian. The *j*-th entry of the list should be a
        list consisting of the *j*-th operator :math:`B_j` making up the noise
        Hamiltonian and a list or array describing the sensitivity
        :math:`s_j(t)` of the system to the noise operator during the time
        intervals *dt*. Optionally, the list may also include operator
        identifiers. That is, *H_n* should look something like this::

            H = [[n_oper1, n_coeff1, n_oper_identifier1],
                 [n_oper2, n_coeff2, n_oper_identifier2], ...]

        The operators may be given either as NumPy arrays or QuTiP Qobjs
        and each coefficient array should have the same number of elements
        as *dt*, and should be given in units of :math:`\hbar`. If not every
        sublist (read: operator) was given a identifier, they are automatically
        filled up with 'A_i' where i is the position of the operator.
    dt : array_like, shape (n_dt,)
        The segment durations of the Hamiltonian (i.e. durations of
        constant control). Internally, the control operation is taken to
        start at :math:`t_0\equiv 0`, i.e. the edges of the constant
        control segments are at times ``t = [0, *np.cumsum(dt)]``.
    basis : Basis, shape (d**2, d, d), optional
        The operator basis in which to calculate. If a Generalized Gell-Mann
        basis (see :meth:`~basis.Basis.ggm`) is chosen, some calculations will
        be faster for large dimensions due to a simpler basis expansion.
        However, when extending the pulse sequence to larger qubit registers,
        cached filter functions cannot be retained since the GGM basis does not
        factor into tensor products. In this case a Pauli basis is preferable.

    Examples
    --------
    A rotation by :math:`\pi` around the axis between x and y preceeded and
    followed by a period of free evolution with the system subject to dephasing
    noise.

    >>> import qutip as qt; import numpy as np
    >>> H_c = [[qt.sigmax(), [0, np.pi, 0]],
               [qt.sigmay(), [0, np.pi, 0]]]
    >>> # Equivalent pulse:
    >>> # H_c = [[qt.sigmax() + qt.sigmay(), [0, np.pi, 0]]]
    >>> # The noise sensitivity is constant
    >>> H_n = [[qt.sigmaz()/np.sqrt(2), [1, 1, 1], 'Z']]
    >>> dt = [1, 1, 1]
    >>> # Free evolution between t=0 and t=1, rotation between t=1 and t=2,
    >>> # and free evolution again from t=2 to t=3.
    >>> pulse = PulseSequence(H_c, H_n, dt)
    >>> pulse.c_oper_identifiers
    ['A_0', 'A_1']
    >>> pulse.n_oper_identifiers
    ['Z']
    >>> omega = np.logspace(-1, 2, 500)
    >>> F = pulse.get_filter_function(omega)    # shape (1, 500)
    >>> # Plot the resulting filter function:
    >>> from filter_functions import plot_filter_function
    >>> fig, ax, leg = plot_filter_function(pulse)

    Attributes
    ----------
    c_opers : ndarray, shape (n_cops, d, d)
        Control operators
    n_opers : ndarray, shape (n_nops, d, d)
        Noise operators
    c_oper_identifers : sequence of str
        Identifiers for the control operators of the system
    n_oper_identifers : sequence of str
        Identifiers for the noise operators of the system
    c_coeffs : ndarray, shape (n_cops, n_dt)
        Control parameters in units of :math:`\hbar`
    n_coeffs : ndarray, shape (n_nops, n_dt)
        Noise sensitivities in units of :math:`\hbar`
    dt : ndarray, shape (n_dt,)
        Time steps
    t : ndarray, shape (n_dt + 1,)
        Absolute times taken to start at :math:`t_0\equiv 0`
    d : int
        Dimension of the Hamiltonian
    basis : Basis, shape (d**2, d, d)
        The operator basis used for calculation
    nbytes : int
        An estimate of the memory consumed by the PulseSequence instance and
        its attributes

    If the Hamiltonian was diagonalized, the eigenvalues and -vectors as well
    as the cumulative propagators are cached:

    HD : ndarray, shape (n_dt, d)
        Eigenvalues
    HV : ndarray, shape (n_dt, d, d)
        Eigenvectors
    Q : ndarray, shape (n_dt+1, d, d)
        Cumulative propagators
    total_Q : ndarray, shape (d, d)
        The total propagator :math:`Q` of the pulse alone. That is,
        :math:`|\psi(\tau)\rangle = Q|\psi(0)\rangle`.
    total_Q_liouville : array_like, shape (d**2, d**2)
        The transfer matrix for the total propagator of the pulse. Given by
        ``liouville_representation(pulse.total_Q, pulse.basis)``.

    Furthermore, when the filter function is calculated, the frequencies are
    cached as well as other relevant quantities.

    Methods
    -------
    :meth:`cleanup`
        Delete cached attributes
    :meth:`is_cached`
        Checks if a given attribute of the ``PulseSequence`` is cached
    :meth:`diagonalize`
        Diagonalize the Hamiltonian of the pulse sequence, computing
        eigenvalues and -vectors as well as cumulative propagators
    :meth:`get_control_matrix`
        Calculate the control matrix for frequencies omega
    :meth:`get_filter_function`
        Calculate the filter function for frequencies omega
    :meth:`get_pulse_correlation_filter_function`
        Get the pulse correlation filter function (only possible if computed
        during concatenation)
    :meth:`propagator_at_arb_t`
        Calculate the propagator at arbitrary times

    Notes
    -----
    Due to the heavy use of NumPy's :func:`~numpy.einsum` function, results
    have a floating point error of ~1e-13.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize a PulseSequence instance."""
        # Initialize attributes set by _parse_args to None to satisfy static
        # code checker
        self.c_opers = None
        self.n_opers = None
        self.c_oper_identifiers = None
        self.n_oper_identifiers = None
        self.c_coeffs = None
        self.n_coeffs = None
        self.dt = None
        self.t = None
        self.d = None
        self.basis = None

        # Parse the input arguments and set attributes
        attributes = ('c_opers', 'c_oper_identifiers', 'c_coeffs', 'n_opers',
                      'n_oper_identifiers', 'n_coeffs', 'dt', 't', 'd',
                      'basis')
        if not args:
            # Bypass args parsing and directly set necessary attributes
            values = (kwargs[attr] for attr in attributes)
        else:
            if len(args) == 4:
                # basis given as arg, not kwarg
                kwargs['basis'] = args[-1]
            elif len(args) < 3:
                posargs = ['H_c', 'H_n', 'dt']
                raise TypeError('Missing {} required '.format(3 - len(args)) +
                                'positional argument(s): ' +
                                '{}'.format(posargs[len(args):]))

            values = _parse_args(*args[:3], **kwargs)

        for attr, value in zip(attributes, values):
            setattr(self, attr, value)

        # Initialize attributes that can be set by bound methods to None
        self._omega = None
        self._HD = None
        self._HV = None
        self._Q = None
        # Attributes that are cached during calculation of the filter function
        self._total_phases = None
        self._total_Q = None
        self._total_Q_liouville = None
        self._R = None
        self._F = None
        self._F_pc = None

    def __str__(self):
        """String method."""
        s = 'PulseSequence object with the following attributes:\n'
        for attr in ('c_opers', 'c_coeffs', 'n_opers', 'n_coeffs', 'dt'):
            s += '{}:\n'.format(attr)
            s += str(getattr(self, attr)) + '\n'

        return s

    def __eq__(self, other: object) -> bool:
        """
        Equality operator. Returns True if the following attributes of the
        operands are equivalent:

            - dt
            - c_opers
            - n_opers
            - c_oper_identifiers
            - n_oper_identifiers
            - c_coeffs
            - n_coeffs
            - basis

        """
        if not isinstance(other, PulseSequence):
            return NotImplemented

        A = self
        B = other
        atol = np.finfo(complex).eps*A.basis.shape[0]
        rtol = 1e-10

        # Two consecutive segments might be equal for all H_opers. In that case
        # they should evaluate equal whether they are written as two or one.
        # Thus, join equal segments to compare.
        c_coeffs_A, n_coeffs_A, dt_A = _join_equal_segments(A)
        c_coeffs_B, n_coeffs_B, dt_B = _join_equal_segments(B)

        if len(dt_A) != len(dt_B):
            # Sequences not the same length
            return False

        if not np.allclose(dt_A, dt_B, rtol, atol):
            return False

        # We require a certain reproducible order for the opers and coeffs so
        # that also after concatenation of different pulses they will be in a
        # deterministic order for comparison. Sort the hashes of the operators'
        # bytes array
        c_idx_A = np.argsort(A.c_oper_identifiers)
        c_idx_B = np.argsort(B.c_oper_identifiers)
        n_idx_A = np.argsort(A.n_oper_identifiers)
        n_idx_B = np.argsort(B.n_oper_identifiers)

        # opers
        if not all(np.array_equal(AH, BH) for AH, BH in
                   zip(A.c_opers[c_idx_A], B.c_opers[c_idx_B])):
            return False

        if not all(np.array_equal(AH, BH) for AH, BH in
                   zip(A.n_opers[n_idx_A], B.n_opers[n_idx_B])):
            return False

        # oper identifiers
        if not all(np.array_equal(AH, BH) for AH, BH in
                   zip(A.c_oper_identifiers[c_idx_A],
                       B.c_oper_identifiers[c_idx_B])):
            return False

        if not all(np.array_equal(AH, BH) for AH, BH in
                   zip(A.n_oper_identifiers[n_idx_A],
                       B.n_oper_identifiers[n_idx_B])):
            return False

        # coefficients
        if not all(np.array_equal(AH, BH) for AH, BH in
                   zip(c_coeffs_A[c_idx_A], c_coeffs_B[c_idx_B])):
            return False

        if not all(np.array_equal(AH, BH) for AH, BH in
                   zip(n_coeffs_A[n_idx_A], n_coeffs_B[n_idx_B])):
            return False

        if not A.basis == B.basis:
            return False

        return True

    def __copy__(self) -> 'PulseSequence':
        """Return shallow copy of self"""
        cls = self.__class__
        copy = cls.__new__(cls)
        copy.__dict__.update(self.__dict__)
        return copy

    def __matmul__(self, other: 'PulseSequence') -> 'PulseSequence':
        """Concatenation of PulseSequences."""
        # Make sure other is a PulseSequence instance (awkward check for type)
        if not hasattr(other, 'c_opers'):
            raise TypeError('Incompatible type for concatenation: ' +
                            '{}'.format(type(other)))

        return concatenate((self, other))

    def __imatmul__(self, other: 'PulseSequence') -> 'PulseSequence':
        raise NotImplementedError

    def is_cached(self, attr: str) -> bool:
        """Returns True if the attribute is cached"""
        if not attr.startswith('_'):
            attr = '_' + attr

        return getattr(self, attr) is not None

    def diagonalize(self) -> None:
        r"""
        Diagonalize the Hamiltonian defining the pulse sequence.
        """
        # Only calculate if not done so before
        if not all(self.is_cached(attr) for attr in ('HD', 'HV', 'Q')):
            # Control Hamiltonian as a (n_dt, d, d) array
            H = np.einsum('ijk,il->ljk', self.c_opers, self.c_coeffs)
            self.HD, self.HV, self.Q = diagonalize(H, self.dt)

        # Set the total propagator
        self.total_Q = self.Q[-1]

    def get_control_matrix(self, omega: Coefficients,
                           show_progressbar: bool = False) -> ndarray:
        r"""
        Get the control matrix for the frequencies *omega*. If it has been
        cached for the same frequencies, the cached version is returned,
        otherwise it is calculated from scratch.

        Parameters
        ----------
        omega : array_like, shape (n_omega,)
            The frequencies at which to evaluate the control matrix.
        show_progressbar : bool
            Show a progress bar for the calculation of the control matrix.

        Returns
        -------
        R : ndarray, shape (n_nops, d**2, n_omega)
            The control matrix for the noise operators.
        """
        # Only calculate if not calculated before for the same frequencies
        if np.array_equal(self.omega, omega):
            if self.is_cached('R'):
                return self._R
        else:
            # Getting with different frequencies. Remove all cached attributes
            # that are frequency-dependent
            self._R = None
            self._F = None
            self._F_pc = None
            self._total_phases = None

        # Make sure the Hamiltonian has been diagonalized
        self.diagonalize()

        R = calculate_control_matrix_from_scratch(
            self.HD, self.HV, self.Q, omega, self.basis, self.n_opers,
            self.n_coeffs, self.dt, self.t, show_progressbar=show_progressbar
        )

        # Cache the result
        self.cache_control_matrix(omega, R)

        return self._R

    def cache_control_matrix(self, omega: Coefficients,
                             R: Optional[ndarray] = None,
                             show_progressbar: bool = False) -> None:
        r"""
        Cache the control matrix and the frequencies it was calculated for.

        Parameters
        ----------
        omega : array_like, shape (n_omega,)
            The frequencies for which to cache the filter function.
        R : array_like, shape (n_nops [, n_nops], d**2, n_omega), optional
            The control matrix for the frequencies *omega*. If ``None``, it is
            computed.
        show_progressbar : bool
            Show a progress bar for the calculation of the control matrix.
        """
        if R is None:
            R = self.get_control_matrix(omega, show_progressbar)

        self.omega = omega
        self._R = R

        # Cache total phase and total transfer matrices as well
        self.cache_total_phases(omega)
        if not self.is_cached('total_Q_liouville'):
            self.total_Q_liouville = liouville_representation(self.total_Q,
                                                              self.basis)

    def get_filter_function(self, omega: Coefficients,
                            show_progressbar: bool = False) -> ndarray:
        r"""
        Calculate the first-order filter function

        .. math::

            F_{\alpha\beta}(\omega) = \left[\mathcal{R}\mathcal{R}^\dagger
                                      \right]_{\alpha\beta}(\omega),

        where :math:`\alpha,\beta` are indices counting the noise operators
        :math:`B_\alpha`. Thus, the filter function :math:`\alpha,\beta`
        corresponds to the noise correlations between :math:`B_\alpha` and
        :math:`B_{\beta}`.

        The filter function is cached so it doesn't need to be calculated
        twice for the same frequencies.

        Parameters
        ----------
        omega : array_like, shape (n_omega,)
            The frequencies at which to evaluate the filter function.
        show_progressbar : bool
            Show a progress bar for the calculation of the control matrix.

        Returns
        -------
        F : ndarray, shape (n_nops, n_nops, n_omega)
            The filter function for each combination of noise operators as a
            function of omega.
        """
        # Only calculate if not calculated before for the same frequencies
        if np.array_equal(self.omega, omega):
            if self.is_cached('F'):
                return self._F
        else:
            # Getting with different frequencies. Remove all cached attributes
            # that are frequency-dependent
            self._R = None
            self._F = None
            self._F_pc = None
            self._total_phases = None

        # Cache filter function
        self.cache_filter_function(
            omega, R=self.get_control_matrix(omega, show_progressbar)
        )

        return self._F

    def cache_filter_function(self, omega: Coefficients,
                              R: Optional[ndarray] = None,
                              F: Optional[ndarray] = None,
                              show_progressbar: bool = False) -> None:
        r"""
        Cache the filter function. If R.ndim == 4, it is taken to be the 'pulse
        correlation control matrix' and summed along the first axis. In that
        case, also the pulse correlation filter function is calculated and
        cached. Total phase factors and transfer matrices of the the cumulative
        propagator are also cached so they can be reused during concatenation.

        Parameters
        ----------
        omega : array_like, shape (n_omega,)
            The frequencies for which to cache the filter function.
        R : array_like, shape (n_nops [, n_nops], d**2, n_omega), optional
            The control matrix for the frequencies *omega*. If ``None``, it is
            computed and the filter function derived from it.
        F : array_like, shape (n_nops, n_nops, n_omega), optional
            The filter function for the frequencies *omega*. If ``None``, it is
            computed from R.
        show_progressbar : bool
            Show a progress bar for the calculation of the control matrix.
        """
        if F is None:
            if R is None:
                R = self.get_control_matrix(omega, show_progressbar)

            if R.ndim == 4:
                # Cache regular control matrix
                self.cache_control_matrix(omega, R.sum(axis=0),
                                          show_progressbar=show_progressbar)
                # Calculate pulse correlation FF and derive canonical FF from
                # it
                self._F_pc = calculate_pulse_correlation_filter_function(R)
                F = np.einsum('ghjlo->jlo', self._F_pc)

            else:
                # Regular case
                self.cache_control_matrix(omega, R,
                                          show_progressbar=show_progressbar)
                F = calculate_filter_function(R)

        self.omega = omega
        self._F = F

    def get_pulse_correlation_filter_function(self) -> ndarray:
        r"""
        Get the pulse correlation filter function given by

        .. math::

            F_{\alpha\beta}^{(gg')}(\omega) = e^{i\omega(t_{g-1} - t_{g'-1})}
                \mathcal{R}^{(g)}(\omega)\mathcal{Q}^{(g-1)}
                \mathcal{Q}^{(g'-1)\dagger}\mathcal{R}^{(g')\dagger}(\omega),

        where :math:`g,g'` index the pulse in the sequence and
        :math:`\alpha,\beta` index the noise operators, if it was computed
        during concatenation. Since the calculation requires the individual
        pulse's control matrices and phase factors, which are not retained
        after concatenation, the pulse correlation filter function cannot be
        computed afterwards.

        Note that the frequencies for which the filter function was calculated
        are not stored.

        Returns
        -------
        F_pc : ndarray, shape (n_pulses, n_pulses, n_nops, n_nops, n_omega)
            The pulse correlation filter function for each noise operator as a
            function of omega. The first two axes correspond to the pulses in
            the sequence, i.e. if the concatenated pulse sequence is
            :math:`C\circ B\circ A`, the first two axes are arranged like

            .. math::

                F_{\alpha\beta}^{(gg')} &= \begin{pmatrix}
                    F_{\alpha\beta}^{(AA)} & F_{\alpha\beta}^{(AB)} &
                        F_{\alpha\beta}^{(AC)} \\
                    F_{\alpha\beta}^{(BA)} & F_{\alpha\beta}^{(BB)} &
                        F_{\alpha\beta}^{(BC)} \\
                    F_{\alpha\beta}^{(CA)} & F_{\alpha\beta}^{(CB)} &
                        F_{\alpha\beta}^{(CC)}
                \end{pmatrix}

            for :math:`g,g'\in\{A, B, C\}`.
        """
        if self.is_cached('F_pc'):
            return self._F_pc

        raise CalculationError("Could not get the pulse correlation " +
                               "function since it was not computed during " +
                               "concatenation. Please run the concatenation " +
                               "again with 'calc_pulse_correlation_ff' set " +
                               "to True.")

    def get_total_phases(self, omega: Coefficients) -> ndarray:
        """Get the (cached) total phase factors for this pulse and omega."""
        # Only calculate if not calculated before for the same frequencies
        if np.array_equal(self.omega, omega):
            if self.is_cached('total_phases'):
                return self._total_phases
        else:
            # Getting with different frequencies. Remove all cached attributes
            # that are frequency-dependent
            self._R = None
            self._F = None
            self._F_pc = None
            self._total_phases = None

        self.cache_total_phases(omega)
        return self._total_phases

    def cache_total_phases(self, omega: Coefficients,
                           total_phases: Optional[ndarray] = None) -> None:
        """
        Cache the total phase factors for this pulse and omega.

        Parameters
        ----------
        omega : array_like, shape (n_omega,)
            The frequencies for which to cache the phase factors.
        total_phases : array_like, shape (n_omega,), optional
            The total phase factors for the frequencies *omega*. If ``None``,
            they are computed.
        """
        if total_phases is None:
            total_phases = cexp(np.asarray(omega)*self.t[-1])

        self.omega = omega
        self._total_phases = total_phases

    @property
    def HD(self) -> ndarray:
        """Get the eigenvalues of the pulse's Hamiltonian."""
        if not self.is_cached('HD'):
            self.diagonalize()

        return self._HD

    @HD.setter
    def HD(self, value) -> None:
        """Set the eigenvalues of the pulse's Hamiltonian."""
        self._HD = value

    @property
    def HV(self) -> ndarray:
        """Get the eigenvectors of the pulse's Hamiltonian."""
        if not self.is_cached('HV'):
            self.diagonalize()

        return self._HV

    @HV.setter
    def HV(self, value) -> None:
        """Set the eigenvalues of the pulse's Hamiltonian."""
        self._HV = value

    @property
    def Q(self) -> ndarray:
        """Get the eigenvectors of the pulse's Hamiltonian."""
        if not self.is_cached('Q'):
            self.diagonalize()

        return self._Q

    @Q.setter
    def Q(self, value) -> None:
        """Set the eigenvalues of the pulse's Hamiltonian."""
        self._Q = value

    @property
    def total_Q(self) -> ndarray:
        """Get total propagator of the pulse."""
        if not self.is_cached('total_Q'):
            self.diagonalize()

        return self._total_Q

    @total_Q.setter
    def total_Q(self, value: ndarray) -> None:
        """Set total propagator of the pulse."""
        self._total_Q = value

    @property
    def total_Q_liouville(self) -> ndarray:
        """Get the transfer matrix for the total propagator of the pulse."""
        if not self.is_cached('total_Q_liouville'):
            self._total_Q_liouville = liouville_representation(self.total_Q,
                                                               self.basis)

        return self._total_Q_liouville

    @total_Q_liouville.setter
    def total_Q_liouville(self, value: ndarray) -> None:
        """Set the transfer matrix of the total cumulative propagator."""
        self._total_Q_liouville = value

    @property
    def omega(self) -> ndarray:
        """Cached frequencies"""
        return self._omega

    @omega.setter
    def omega(self, value: Coefficients) -> None:
        """Cache frequencies"""
        self._omega = np.asarray(value) if value is not None else value

    @property
    def nbytes(self) -> int:
        """
        Return an estimate of the amount of memory consumed by this object (or,
        more precisely, the array attributes of this object).
        """
        _nbytes = []
        for val in self.__dict__.values():
            try:
                _nbytes.append(val.nbytes)
            except AttributeError:
                pass

        return sum(_nbytes)

    def cleanup(self, method: str = 'conservative') -> None:
        """
        Delete cached byproducts of the calculation of the filter function that
        are not necessarily needed anymore in order to free up memory.

        Parameters
        ----------
        method : {'conservative', 'greedy', 'all'}, optional
            If set to 'conservative' (the default), only the following
            attributes are deleted:

                - _HD
                - _HV
                - _Q

            If set to 'greedy', all of the above as well as the following
            attributes are deleted:

                - _total_phases
                - _total_Q
                - _total_Q_liouville
                - _R

            If set to 'all', all of the above as well as the following
            attributes are deleted:

                - omega
                - _F
                - _F_pc

            Note that if this ``PulseSequence`` is concatenated with another
            one, some of the attributes might need to be calculated again,
            resulting in slower execution of the concatenation.
        """
        attrs = ['_HD', '_HV', '_Q']
        if method != 'conservative':
            attrs.extend(['_R', '_total_phases', '_total_Q',
                          '_total_Q_liouville'])
            if method != 'greedy':
                attrs.extend(['omega', '_F', '_F_pc'])

        for attr in attrs:
            setattr(self, attr, None)

    def propagator_at_arb_t(self, t: Coefficients) -> ndarray:
        """
        Calculate the cumulative propagator Q(t) at times *t* by making use of
        the fact that we assume piecewise-constant control.
        """
        # Index of the the propagator Q(t_{l-1}) that evolves the state up to
        # the l-1-st step. Since control is piecewise constant, all we have to
        # do to get the state at an arbitrary time t_{l-1} <= t < t_l is
        # propagate with a time-delta t - t_{l-1} and H_{l}
        self.diagonalize()
        idx = np.searchsorted(self.t, t) - 1
        # Manually set possible negative idx's to zero (happens for t = 0)
        idx[idx < 0] = 0
        Q_prev = self.Q[idx]
        U_curr = np.einsum('lij,jl,lkj->lik', self.HV[idx],
                           cexp((self.t[idx] - t)*self.HD[idx].T),
                           self.HV[idx].conj())

        return U_curr @ Q_prev


def _join_equal_segments(pulse: PulseSequence) -> Sequence[Coefficients]:
    """Join potentially equal consecutive segments of *pulse*'s Hamiltonian."""
    equal_ind = (np.diff(pulse.c_coeffs) == 0).all(axis=0).nonzero()[0]

    if equal_ind.size > 0:
        c_coeffs = np.delete(pulse.c_coeffs, equal_ind, axis=1)
        n_coeffs = np.delete(pulse.n_coeffs, equal_ind, axis=1)
        dt = np.delete(pulse.dt, equal_ind)
        for old, new in zip(equal_ind, equal_ind - np.arange(len(equal_ind))):
            dt[new] += pulse.dt[old]
    else:
        c_coeffs = pulse.c_coeffs
        n_coeffs = pulse.n_coeffs
        dt = pulse.dt

    return c_coeffs, n_coeffs, dt


def _parse_args(H_c: Hamiltonian, H_n: Hamiltonian, dt: Coefficients,
                **kwargs) -> Any:
    """
    Function to parse the arguments given at instantiation of the PulseSequence
    object.
    """

    if not hasattr(dt, '__getitem__'):
        raise TypeError('Expected a sequence of time steps, not {}'.format(
            type(dt)))

    dt = np.asarray(dt)
    # Check the time argument for data type and monotonicity (should be
    # increasing)
    if not np.isreal(dt).all():
        raise ValueError('Times dt are not (all) real!')
    if (dt < 0).any():
        raise ValueError('Time steps are not (all) positive!')

    control_args = _parse_Hamiltonian(H_c, len(dt), 'H_c')
    noise_args = _parse_Hamiltonian(H_n, len(dt), 'H_n')

    if control_args[0].shape[-2:] != noise_args[0].shape[-2:]:
        # Check operator shapes
        raise ValueError('Control and noise Hamiltonian not same dimension!')

    t = np.concatenate(([0], dt.cumsum()))
    # Dimension of the system
    d = control_args[0].shape[-1]

    basis = kwargs.pop('basis', None)
    if basis is None:
        # Use generalized Gell-Mann basis by default since we have a nice
        # expression for a basis expansion
        basis = Basis.ggm(d)
    else:
        if not all(hasattr(basis, attr) for attr in ('_atol', '_rtol')):
            raise ValueError("Expected basis to be an instance of the " +
                             "'filter_functions.basis.Basis' class, not " +
                             "{}!".format(type(basis)))
        if basis.shape[1:] != (d, d):
            # Make sure the basis has the correct dimension (we allow an
            # incomplete set)
            raise ValueError("Expected basis elements to be of shape " +
                             "({0}, {0}), not {1}!".format(d, basis.shape[1:]))

    return (*control_args, *noise_args, dt, t, d, basis)


def _parse_Hamiltonian(H: Hamiltonian, n_dt: int,
                       H_str: str) -> Tuple[Sequence[Operator],
                                            Sequence[str],
                                            Sequence[Coefficients]]:
    """Helper function to parse the Hamiltonian in QuTiP format."""
    # Check correct types of the various levels of nestedness
    if not isinstance(H, (list, tuple)):
        raise TypeError('Expected {} to be a list of '.format(H_str) +
                        'lists, not of type {}!'.format(type(H)))

    if not all(isinstance(item, (list, tuple)) for item in H):
        raise TypeError('Expected {} to be a list of '.format(H_str) +
                        'lists but found at least one item of H not of ' +
                        'type list or tuple!')

    # Unzip the nested lists into operators and coefficient arrays. Since
    # identifiers are optional, we need to perform a check if they were given.
    opers, *args = zip_longest(*H, fillvalue=None)
    if len(args) == 1:
        coeffs = args[0]
        identifiers = None
    elif len(args) == 2:
        coeffs = args[0]
        identifiers = list(args[1])
    else:
        coeffs = args[0]
        identifiers = list(args[1])

    if not all(isinstance(oper, (ndarray, Qobj)) for oper in opers):
        raise TypeError('Expected operators in '.format(H_str) +
                        'to be NumPy arrays or QuTiP Qobjs!')

    if not all(hasattr(coeff, '__getitem__') for coeff in coeffs):
        raise TypeError('Expected coefficients in '.format(H_str) +
                        'to be a sequence')

    # Convert qt.Qobjs to full arrays
    try:
        opers = np.array([oper.full() if isinstance(oper, Qobj) else oper
                          for oper in opers])
    except ValueError:
        raise TypeError("Couldn't parse operators in {}. ".format(H_str) +
                        "Are you sure they are all 2d arrays or Qobjs?")

    # Check correct dimensions for the operators
    if set(oper.ndim for oper in opers) != {2}:
        raise ValueError('Expected all operators in {} '.format(H_str) +
                         'to be two-dimensional!')

    if len(set(oper.shape for oper in opers)) != 1:
        raise ValueError('Expected all operators in {} '.format(H_str) +
                         'to have the same dimensions!')

    if len(set(opers[0].shape)) != 1:
        raise ValueError('Expected operators in {} '.format(H_str) +
                         'to be square!')

    # parse the identifiers
    if identifiers is None:
        if H_str == 'H_c':
            identifiers = np.fromiter(
                ('A_{}'.format(i) for i in range(len(opers))),
                dtype='<U4'
            )
        elif H_str == 'H_n':
            identifiers = np.fromiter(
                ('B_{}'.format(i) for i in range(len(opers))),
                dtype='<U4'
            )
    else:
        for i, identifier in enumerate(identifiers):
            if identifier is None:
                if H_str == 'H_c':
                    identifiers[i] = 'A_{}'.format(i)
                elif H_str == 'H_n':
                    identifiers[i] = 'B_{}'.format(i)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('{} identifiers should be unique'.format(H_str))

        identifiers = np.asarray(identifiers)

    # Check coeffs are all the same length as dt
    if not all(len(coeff) == n_dt for coeff in coeffs):
        raise ValueError('Expected all coefficients in {} '.format(H_str) +
                         'to be of len(dt) = {}!'.format(n_dt))

    coeffs = np.asarray(coeffs)
    idx = np.argsort(identifiers)
    return opers[idx], identifiers[idx], coeffs[idx]


def _concatenate_Hamiltonian(
        opers: Sequence[Sequence[Operator]],
        identifiers: Sequence[Sequence[str]],
        coeffs: Sequence[Sequence[Coefficients]],
        kind: str) -> Tuple[Sequence[Operator],
                            Sequence[str],
                            Sequence[Coefficients],
                            Dict[int, Dict[str, str]]]:
    """
    Concatenate Hamiltonians.

    Returns lists of opers, identifiers, and coeffs so that
    ``list(zip(opers, coeffs, identifiers))`` is in the format required
    by ``PulseSequence``.

    If two operators have the same identifier but are actually different, the
    clash is removed by adding the position of the pulse in the sequence as a
    subscript to each identifier.

    Parameters
    ----------
    opers : array_like
        The operators, should be of structure::

        ((A_oper_1, A_oper_2, ...), (B_oper_1, ...), (...), ...)

        for Hamiltonians *A*, *B*, ...
    identifiers : array_like
        The identifiers, should be of same structure as opers.
    coeffs : array_like
        The coefficients, should be of same structure as opers.
    kind : str
        The type of Hamiltonian, either 'control' or 'noise'.

    Returns
    -------
    concat_opers : ndarray, shape (n_opers, d, d)
        The operators of the concatenated Hamiltonian.
    concat_identifiers : ndarray, shape (n_opers,)
        The identifiers of the concatenated Hamiltonian.
    concat_coeffs : ndarray, shape (n_opers, n_dt)
        The coefficients of the concatenated Hamiltonian.
    pulse_identifier_mapping : Dict[int, Dict[str, str]]
        Dictionary that maps the operator identifiers of the original
        pulses to those of the new pulse.
    """
    # Number of time steps in each pulse
    n_dt = [0] + [coeff.shape[1] for coeff in coeffs]
    # Number of operators in each subset of opers, i.e. number of operators for
    # each pulse being concatenated
    n_ops = [len(op) for op in opers]
    # Indices where operators of another pulse follow in flatten list of opers.
    # I.e, for opers = ((O1, O2), (O3, O4, O5)), pulse_idx == (2, 5).
    pulse_idx = list(accumulate(n_ops))
    # Indices similar to pulse_idx, except for the coefficients. I.e., for
    # coeffs = (([1,2,3], [1,2,3]), ([1,2,3,4,5])), seg_idx == (3,8) so that
    # we know where in the new coefficient array to place coeffs belonging to
    # a pulse
    seg_idx = list(accumulate(n_dt))

    # Check if we have any clashes between operators and identifiers and
    # if so, handle them
    all_opers = np.concatenate(opers, axis=0)
    all_identifiers = np.concatenate(identifiers)
    hashed_identifiers = [hash(i) for i in all_identifiers]
    hashed_opers = hash_array_along_axis(all_opers, axis=0)
    concat_hashed_opers, concat_idx, inverse_idx = np.unique(
        hashed_opers, return_index=True, return_inverse=True
    )
    # Convert to list so we can use .index()
    concat_hashed_opers = concat_hashed_opers.tolist()
    # Convert to list so we can modify the string
    concat_identifiers = all_identifiers[concat_idx].tolist()

    # Hash tables in both directions
    oper_to_identifier_mapping = {}
    identifier_to_oper_mapping = {}
    for oper, identifier in zip(hashed_opers, hashed_identifiers):
        op_to_id = oper_to_identifier_mapping.setdefault(oper, set())
        op_to_id.add(identifier)
        id_to_op = identifier_to_oper_mapping.setdefault(identifier, set())
        id_to_op.add(oper)

    if any(len(value) > 1 for value in oper_to_identifier_mapping.values()):
        # Clash: two different identifiers are assigned to the same operator
        raise ValueError('Trying to concatenate pulses with equal ' +
                         'operators with different identifiers. Please ' +
                         'choose unique identifiers!')

    # A dict that maps the identifiers of each Hamiltonian to the identifiers
    # in the new Hamiltonian
    pulse_identifier_mapping = {
        p: {identifier: identifier for identifier in identifiers[p]}
        for p in range(len(pulse_idx))
    }
    for identifier, oper in identifier_to_oper_mapping.items():
        identifier_str = all_identifiers[hashed_identifiers.index(identifier)]
        if len(oper) > 1:
            # Clash: two different operators are assigned to the same
            # identifier. Add pulse position suffix to identifiers to make them
            # unique
            pulse_pos = [bisect.bisect(pulse_idx, hashed_opers.index(op))
                         for op in oper]
            identifier_pos = [concat_hashed_opers.index(op) for op in oper]
            for i, p in zip(identifier_pos, pulse_pos):
                concat_identifiers[i] = concat_identifiers[i] + '_{}'.format(p)
                pulse_identifier_mapping[p].update(
                    {identifier_str: concat_identifiers[i]}
                )

    # Sort everything by the identifiers
    sort_idx = np.argsort(concat_identifiers)
    concat_opers = all_opers[concat_idx[sort_idx]]
    concat_identifiers = np.array([concat_identifiers[i] for i in sort_idx])

    # Concatenate the coefficients. Place them in the right time segments of
    # the concatenated Hamiltonian.
    concat_coeffs = np.zeros((len(concat_identifiers), sum(n_dt)), dtype=float)
    flat_coeffs = [co for coeff in coeffs for co in coeff]
    for i in range(len(concat_identifiers)):
        # Get the indices in opers (and coeffs) for the i-th unique operator
        indices = (inverse_idx == i).nonzero()[0]
        for ind in indices:
            # For each equal operator, place the corresponding coefficients at
            # the right place in the new coefficient array. This way they are
            # already sorted by identifiers
            seg = bisect.bisect(pulse_idx, ind)
            concat_coeffs[i, seg_idx[seg]:seg_idx[seg+1]] = flat_coeffs[ind]

    if kind == 'noise':
        # Noise Hamiltonian. If not all operators are present on all pulses,
        # we will try to infer the noise sensitivities (== coefficients) from
        # the remaining segments as usually the sensitivity is constant. If we
        # cannot do this, we have to raise an exception since we cannot know
        # the sensitivities at other moments in time if they are non-trivial.
        for i, c_coeffs in enumerate(concat_coeffs):
            zero_mask = (c_coeffs == 0)
            if zero_mask.any() and not zero_mask.all():
                nonzero_coeffs = c_coeffs[~zero_mask]
                constant = (nonzero_coeffs == nonzero_coeffs[0]).all()
                if constant:
                    # Fill with constant value
                    concat_coeffs[i, zero_mask] = nonzero_coeffs[0]
                else:
                    raise ValueError('Not all pulses have the same noise ' +
                                     'operators and non-trivial noise ' +
                                     'sensitivities so I cannot infer them.')

    return (concat_opers, concat_identifiers, concat_coeffs[sort_idx],
            pulse_identifier_mapping)


def _merge_attrs(old_attrs: List[ndarray], new_attrs: List[ndarray],
                 d_per_qubit: int, registers: List[int],
                 qubits: List[int]) -> Tuple[ndarray, List[int]]:
    """
    For each array in new_attrs, merge into the tensor product array defined
    on the qubit registers in old_attrs at qubits.
    """

    if registers is None:
        return new_attrs, qubits.copy()

    # Get the correct position where the array should be inserted
    pos = [bisect.bisect(registers, q) for q in qubits]
    attrs = []
    for old_attr, new_attr in zip(old_attrs, new_attrs):
        attrs.append(tensor_merge(old_attr, new_attr, pos,
                                  arr_dims=[[d_per_qubit]*len(registers)]*2,
                                  ins_dims=[[d_per_qubit]*len(pos)]*2))

    # Update the registers
    for q in qubits:
        bisect.insort(registers, q)

    return attrs, registers


def _insert_attrs(old_attrs: List[ndarray], new_attrs: List[ndarray],
                  d_per_qubit: int, registers: List[int],
                  qubit: int) -> Tuple[ndarray, List[int]]:
    """
    For each array in new_attrs, insert into the tensor product array defined
    on the qubit registers in old_attrs at qubit.
    """

    if registers is None:
        return new_attrs, [qubit]

    # Get the correct position where the array should be inserted
    pos = bisect.bisect(registers, qubit)
    attrs = []
    for old_attr, new_attr in zip(old_attrs, new_attrs):
        attrs.append(tensor_insert(old_attr, new_attr, pos=pos,
                                   arr_dims=[[d_per_qubit]*len(registers)]*2))
    # Update the registers
    bisect.insort(registers, qubit)

    return attrs, registers


def _map_identifiers(identifiers: Sequence[str],
                     mapping: Union[None, Mapping[str, str]]):
    """
    Return identifiers remapped according to mapping. If mapping is None, the
    identifiers are mapped to themselves.

    Parameters
    ----------
    identifiers : sequence of str
        The identifiers to remap.
    mapping : dict_like
        The mapping according to which to remap.

    Returns
    -------
    identifiers : ndarray
        The identifiers.
    sort_idx : ndarray
        The indices which sort the identifiers.
    """
    # Remap identifiers
    if mapping is None:
        remapped_identifiers = identifiers
        sort_idx = np.arange(len(identifiers))
    else:
        remapped_identifiers = np.array([mapping[identifier]
                                         for identifier in identifiers])
        sort_idx = np.argsort(remapped_identifiers)

    return remapped_identifiers, sort_idx


def _default_extend_mapping(identifiers: Sequence[str],
                            mapping: Union[None, Mapping[str, str]],
                            qubits: Union[Sequence[int], int]):
    """
    Get a default identifier mapping for a pulse that was extended to *qubits*
    if *mapping* is None, else return mapping.

    Parameters
    ----------
    identifiers : Sequence[str]
        The identifiers to remap.
    qubits : Union[Sequence[int], int]
        The qubits the pulse was mapped to.

    Returns
    -------
    identifiers : ndarray
        The identifiers.
    mapping : ndarray
        The default mapping.

    """
    if mapping is not None:
        return identifiers, mapping

    try:
        mapping = {l: l + '_' + ('{}'*len(qubits)).format(*qubits)
                   for l in identifiers}
    except TypeError:
        mapping = {l: l + '_{}'.format(qubits) for l in identifiers}

    return identifiers, mapping


def concatenate_without_filter_function(
        pulses: Iterable[PulseSequence],
        return_identifier_mappings: bool = False) -> Any:
    """
    Concatenate PulseSequences, disregarding the filter function.

    Parameters
    ----------
    pulses : iterable of PulseSequences
        The PulseSequence instances to be concatenated.
    return_identifier_mappings : bool, optional
        Return dictionaries which map the identifiers of control and noise
        operators of the input pulses to those of the new pulse. This mapping
        is only non-trivial if any of the pulses have two different operators
        assigned to the same identifier.

    Returns
    -------
    newpulse : PulseSequence
        The concatenated PulseSequence
    c_oper_identifier_mapping : Dict[int: Dict[str, str]]
        A dictionary that maps the control operator identifiers of the original
        pulses to those of the new pulse.
    n_oper_identifier_mapping : Dict[int: Dict[str, str]]
        A dictionary that maps the noise operator identifiers of the original
        pulses to those of the new pulse.

    See Also
    --------
    :func:`concatenate`

    :func:`concatenate_periodic`
    """
    pulses = tuple(pulses)
    try:
        # Do awkward checking for type
        if not all(hasattr(pls, 'c_opers') for pls in pulses):
            raise TypeError('Can only concatenate PulseSequences!')
    except TypeError:
        raise TypeError('Expected pulses to be iterable, not ' +
                        '{}'.format(type(pulses)))

    # Check if the Hamiltonians' shapes are compatible, ie the set of all
    # shapes has length 1
    if len(set(pulse.c_opers.shape[1:] for pulse in pulses)) != 1:
        raise ValueError('Trying to concatenate two PulseSequence ' +
                         'instances with incompatible Hamiltonian shapes')

    # Check if the bases are the same by hashing them and creating a set
    if not all_array_equal((pulse.basis for pulse in pulses)):
        raise ValueError('Trying to concatenate two PulseSequence ' +
                         'instances with different bases!')

    basis = pulses[0].basis
    control_keys = ('c_opers', 'c_oper_identifiers', 'c_coeffs')
    noise_keys = ('n_opers', 'n_oper_identifiers', 'n_coeffs')

    # Compose new control Hamiltonian
    control_values = _concatenate_Hamiltonian(
        *list(zip(*[tuple(getattr(pulse, key) for key in control_keys)
                    for pulse in pulses])),
        kind='control'
    )
    # Compose new control Hamiltonian
    noise_values = _concatenate_Hamiltonian(
        *list(zip(*[tuple(getattr(pulse, key) for key in noise_keys)
                    for pulse in pulses])),
        kind='noise'
    )

    dt = np.concatenate(tuple(pulse.dt for pulse in pulses))
    t = np.concatenate(([0], dt.cumsum()))

    attributes = {'dt': dt, 't': t, 'd': pulses[0].d, 'basis': basis}
    attributes.update(**{key: value for key, value
                         in zip(control_keys, control_values)})
    attributes.update(**{key: value for key, value
                         in zip(noise_keys, noise_values)})

    newpulse = PulseSequence(**attributes)
    if return_identifier_mappings:
        return newpulse, control_values[-1], noise_values[-1]

    return newpulse


def concatenate(pulses: Iterable[PulseSequence],
                calc_pulse_correlation_ff: bool = False,
                calc_filter_function: Optional[bool] = None,
                omega: Optional[Coefficients] = None,
                show_progressbar: bool = False) -> PulseSequence:
    r"""
    Concatenate an arbitrary number of pulses. Note that pulses are
    concatenated left-to-right, that is,

    .. math::

        \mathtt{concatenate((A, B))} \equiv B \circ A

    so that :math:`A` is executed before :math:`B` when applying the
    concatenated pulse.

    Parameters
    ----------
    pulses : sequence of PulseSequences
        The PulseSequence instances to be concatenated. If any of the instances
        have a cached filter function, the filter function for the composite
        pulse will also be calculated in order to make use of the speedup
        gained from concatenating the filter functions. If *omega* is given,
        calculation of the composite filter function is forced.
    calc_pulse_correlation_ff : bool, optional
        Switch to control whether the pulse correlation filter function (see
        :ref:`notes`) is calculated. If *omega* is not given, the cached
        frequencies of all *pulses* need to be equal.
    calc_filter_function : bool, optional
        Switch to force the calculation of the filter function to be carried
        out or not. Overrides the automatic behavior of calculating it if at
        least one pulse has a cached control matrix. If ``True`` and no pulse
        has a cached control matrix, a list of frequencies must be supplied
        as *omega*.
    omega : array_like, optional
        Frequencies at which to evaluate the (pulse correlation) filter
        functions. If ``None``, an attempt is made to use cached frequencies.
    show_progressbar : bool
        Show a progress bar for the calculation of the control matrix.

    Returns
    -------
    pulse : PulseSequence
        The concatenated pulse.

    .. _notes:

    Notes
    -----
    The pulse correlation filter function is given by

    .. math::

        F_{\alpha\beta}^{(gg')}(\omega) = e^{i\omega(t_{g-1} - t_{g'-1})}
            \mathcal{R}^{(g)}(\omega)\mathcal{Q}^{(g-1)}
            \mathcal{Q}^{(g'-1)\dagger}\mathcal{R}^{(g')\dagger}(\omega),

    where :math:`g,g'` index the pulse in the sequence and :math:`\alpha,\beta`
    index the noise operators.
    """
    pulses = tuple(pulses)
    if len(pulses) == 1:
        return copy(pulses[0])

    newpulse, _, n_oper_mapping = concatenate_without_filter_function(
        pulses, return_identifier_mappings=True
    )

    if all(pls.is_cached('total_Q') for pls in pulses):
        newpulse.total_Q = mdot([pls.total_Q for pls in pulses][::-1])

    if calc_filter_function is False and not calc_pulse_correlation_ff:
        return newpulse

    # If the pulses have different noise operators, we cannot reuse cached
    # filter functions. Since some noise operator identifiers might have been
    # remapped, we use the mapping dictionary returned by
    # concatenate_without_filter_function
    pulse_identifiers = []
    for _, mapping in sorted(n_oper_mapping.items()):
        pulse_identifiers.append([i for i in sorted(mapping.values())])

    unique_identifiers = sorted(set(h for i in pulse_identifiers for h in i))
    # matrix with pulses in rows and all noise operator identifier hashes in
    # columns. True if the noise operator is present in the pulse, False if
    # not. This will give us a boolean mask for indexing the pulses attributes
    # when retriving the filter functions.
    n_opers_present = np.zeros((len(pulses), len(unique_identifiers)),
                               dtype=bool)
    for i, pulse_identifier in enumerate(pulse_identifiers):
        for j, identifier in enumerate(unique_identifiers):
            if identifier in pulse_identifier:
                n_opers_present[i, j] = True

    # If at least two pulses have the same noise operators, we gain an
    # advantage when concatenating the filter functions over calculating them
    # from scratch at a later point
    equal_n_opers = (n_opers_present.sum(axis=0) > 1).any()
    if omega is None:
        cached_ctrl_mat = [pls.is_cached('R') for pls in pulses]
        equal_omega = all_array_equal(
            (pls.omega for pls in compress(pulses, cached_ctrl_mat))
        )
        if not equal_omega:
            if calc_filter_function:
                raise ValueError("Calculation of filter function forced " +
                                 "but not all pulses have the same " +
                                 "frequencies cached and none were supplied!")
            if calc_pulse_correlation_ff:
                raise ValueError("Cannot compute the pulse correlation " +
                                 "filter functions; do not have the " +
                                 "frequencies at which to evaluate.")

            return newpulse

        if calc_filter_function is None:
            # compute filter function only if at least one pulse has a control
            # matrix cached
            if not equal_n_opers or not any(cached_ctrl_mat):
                return newpulse

        # Can reuse cached filter functions or calculation explicitly asked
        # for; run calculation. Get the index of the first pulse with cached FF
        # to steal some attributes from.
        ind = np.nonzero(cached_ctrl_mat)[0][0]
        omega = pulses[ind].omega

    if not equal_n_opers:
        # Cannot reuse atomic filter functions
        newpulse.cache_filter_function(omega)
        return newpulse

    # Get the phase factors at the correct times (the individual gate
    # durations) which are just the total phase factors of the pulses cumprod'd
    phases = np.array(
        [np.ones_like(omega)] +
        [pls.get_total_phases(omega) for pls in pulses[:-1]]
    ).cumprod(axis=0)

    # Get the transfer matrices for the individual gates
    N = len(newpulse.basis)
    L = np.empty((len(pulses), N, N))
    L[0] = np.identity(N)
    for i in range(1, len(pulses)):
        L[i] = pulses[i-1].total_Q_liouville @ L[i-1]

    # Get the control matrices for each pulse (agnostic of if it was cached or
    # not). Those are the 'new' pulse control matrices. Sort them along the
    # axis belonging to the noise operators
    R_g = np.empty((len(pulses), len(newpulse.n_opers), N, len(omega)),
                   dtype=complex)
    n_dt_segs = [len(pulse.dt) for pulse in pulses]
    seg_idx = [0] + list(accumulate(n_dt_segs))
    for i, (pulse, idx) in enumerate(zip(pulses, n_opers_present)):
        R_g[i, idx] = pulse.get_control_matrix(omega, show_progressbar)
        if not idx.all():
            # calculate the control matrix for the noise operators that are
            # not present in pulse
            R_g[i, ~idx] = calculate_control_matrix_from_scratch(
                pulse.HD, pulse.HV, pulse.Q, omega, pulse.basis,
                newpulse.n_opers[~idx],
                newpulse.n_coeffs[~idx, seg_idx[i]:seg_idx[i+1]],
                pulse.dt, pulse.t, show_progressbar
            )

    # Set the total propagator for possible future concatenations (if not done
    # so above)
    if not newpulse.is_cached('total_Q'):
        newpulse.total_Q = mdot([pls.total_Q for pls in pulses][::-1])

    newpulse.cache_total_phases(omega)
    newpulse.total_Q_liouville = liouville_representation(newpulse.total_Q,
                                                          newpulse.basis)

    if calc_pulse_correlation_ff:
        path = ['einsum_path', (1, 2), (0, 1)]
        R = np.einsum('go,gjlo,glk->gjko', phases, R_g, L, optimize=path)
    else:
        R = calculate_control_matrix_from_atomic(phases, R_g, L,
                                                 show_progressbar)

    # Set the attribute and calculate filter function (if the pulse correlation
    # FF has been calculated, this is a little overhead but negligible)
    newpulse.cache_filter_function(omega, R)

    return newpulse


def concatenate_periodic(pulse: PulseSequence, repeats: int) -> PulseSequence:
    r"""
    Concatenate a pulse sequence *pulse* whose Hamiltonian is periodic
    *repeats* times. Although performing the same task, this function is much
    faster for concatenating many identical pulses with filter functions than
    :func:`concatenate`.

    Note that for large dimensions, the calculation of the control matrix using
    this function might be very memory intensive.

    Parameters
    ----------
    pulse : PulseSequence
        The ``PulseSequence`` instance to be repeated. If it has a cached
        filter function, the filter function for the new pulse will also be
        computed.
    repeats : int
        The number of repetitions

    Returns
    -------
    newpulse : PulseSequence
        The concatenated ``PulseSequence``

    Notes
    -----
    The total control matrix is given by

    .. math::

        \mathcal{R}(\omega) &= \mathcal{R}^{(1)}(\omega)\sum_{g=0}^{G-1}\left(
                               e^{i\omega T}\right)^g \\
                            &= \mathcal{R}^{(1)}(\omega)\bigl(
                               \mathbb{I} - e^{i\omega T}\mathcal{Q}^{(1)}
                               \bigr)^{-1}\bigl(\mathbb{I} - \bigl(
                               e^{i\omega T}\mathcal{Q}^{(1)}\bigr)^G\bigr).

    with :math:`T` the period of the control Hamiltonian and :math:`G` the
    number of periods. The last equality is valid only if
    :math:`\mathbb{I} - e^{i\omega T}\mathcal{Q}^{(1)}` is invertible.

    See also
    --------
    :func:`concatenate`
    """

    try:
        # Do awkward checking for type
        if not hasattr(pulse, 'c_opers'):
            raise TypeError('Can only concatenate PulseSequences!')
    except TypeError:
        raise TypeError('Expected pulses to be iterable, not ' +
                        '{}'.format(type(pulse)))

    cached_ctrl_mat = pulse.is_cached('R')

    # Initialize a new PulseSequence instance with the Hamiltonians sequenced
    # (this is much easier than in the general case, thus do it on the fly)
    dt = np.tile(pulse.dt, repeats)
    newpulse = PulseSequence(
        c_opers=pulse.c_opers,
        n_opers=pulse.n_opers,
        c_oper_identifiers=pulse.c_oper_identifiers,
        n_oper_identifiers=pulse.n_oper_identifiers,
        c_coeffs=np.tile(pulse.c_coeffs, (1, repeats)),
        n_coeffs=np.tile(pulse.n_coeffs, (1, repeats)),
        dt=dt,
        t=np.concatenate(([0], dt.cumsum())),
        d=pulse.d,
        basis=pulse.basis
    )

    if not cached_ctrl_mat:
        # No cached filter functions to reuse and pulse correlation FFs not
        # requested. If they were, continue even if there are no cached FF
        # they cannot be computed anymore afterwards.
        return newpulse

    phases_at = pulse.get_total_phases(pulse.omega)
    R_at = pulse.get_control_matrix(pulse.omega)
    L_at = pulse.total_Q_liouville

    newpulse.total_Q = linalg.matrix_power(pulse.total_Q, repeats)
    newpulse.cache_total_phases(pulse.omega)
    # Might be cheaper for small repeats to use matrix_power, but this function
    # is aimed at a large number so we calculate it explicitly
    newpulse.total_Q_liouville = newpulse.total_Q_liouville

    R_tot = calculate_control_matrix_periodic(phases_at, R_at, L_at,
                                              repeats)

    newpulse.cache_filter_function(pulse.omega, R_tot)

    return newpulse


def remap(pulse: PulseSequence, order: Sequence[int], d_per_qubit: int = 2,
          oper_identifier_mapping: Mapping[str, str] = None) -> PulseSequence:
    """
    Remap a PulseSequence by changing the order of qubits in the register.
    Cached attributes are automatically attempted to be retained.

    .. caution::

        This function simply permutes the order of the tensor product
        elements of control and noise operators. Thus, the resultant pulse
        will have its filter functions defined for different noise operators
        than the original one.

    Parameters
    ----------
    pulse : PulseSequence
        The pulse whose qubit order should be permuted.
    order : sequence of ints
        A list of permutation indices. E.g., if *pulse* is defined for two
        qubits, ``order == [1, 0]`` will reverse the order of qubits.
    d_per_qubit : int (default: 2)
        The size of the Hilbert space a single qubit inhabitates.
    oper_identifier_mapping : dict_like
        A mapping that maps operator identifiers from the old pulse to
        the remapped pulse. The default is the identity mapping.

    Returns
    -------
    remapped_pulse : PulseSequence
        A new ``PulseSequence`` instance with the order of the qubits permuted
        according to *order*.

    Examples
    --------
    >>> X, Y = util.P_np[1:3]
    >>> XY, YX = util.tensor(X, Y), util.tensor(Y, X)
    >>> pulse = PulseSequence([[XY, [np.pi/2], 'XY']], [[YX, [1], 'YX']], [1],
    ...                       Basis.pauli(2))
    >>> mapping = {'XY': 'YX', 'YX': 'XY'}
    >>> remapped_pulse = remap(pulse, (1, 0), oper_identifier_mapping=mapping)
    >>> target_pulse = PulseSequence([[YX, [np.pi/2], 'YX']],
    ...                              [[XY, [1], 'XY']], [1], Basis.pauli(2))
    >>> remapped_pulse == target_pulse
    True

    Caching of attributes is automatically handled
    >>> remapped_pulse.is_cached('F')
    False
    >>> pulse.cache_filter_function(util.get_sample_frequencies(pulse))
    >>> remapped_pulse = remap(pulse, (1, 0))
    >>> remapped_pulse.is_cached('F')
    True

    See Also
    --------
    :func:`extend`

    :func:`util.tensor_transpose`
    """
    # Number of qubits
    N = int(np.log(pulse.d)/np.log(d_per_qubit))

    # Transpose control and noise operators
    c_opers = tensor_transpose(pulse.c_opers, order, [[d_per_qubit]*N]*2)
    n_opers = tensor_transpose(pulse.n_opers, order, [[d_per_qubit]*N]*2)

    # Remap identifiers
    c_oper_identifiers, c_sort_idx = _map_identifiers(pulse.c_oper_identifiers,
                                                      oper_identifier_mapping)
    n_oper_identifiers, n_sort_idx = _map_identifiers(pulse.n_oper_identifiers,
                                                      oper_identifier_mapping)

    remapped_pulse = PulseSequence(
        c_opers=c_opers[c_sort_idx],
        n_opers=n_opers[n_sort_idx],
        c_oper_identifiers=c_oper_identifiers[c_sort_idx],
        n_oper_identifiers=n_oper_identifiers[n_sort_idx],
        c_coeffs=pulse.c_coeffs[c_sort_idx],
        n_coeffs=pulse.n_coeffs[n_sort_idx],
        dt=pulse.dt,
        t=pulse.t,
        d=pulse.d,
        basis=pulse.basis
    )

    if pulse.is_cached('HD'):
        remapped_pulse.HD = tensor_transpose(pulse.HD, order,
                                             [[d_per_qubit]*N], rank=1)

    for attr in ('HV', 'Q', 'total_Q'):
        if pulse.is_cached(attr):
            setattr(remapped_pulse, attr, tensor_transpose(
                getattr(pulse, attr), order, [[d_per_qubit]*N]*2
            ))

    if not pulse.is_cached('omega'):
        # If no frequencies are cached, stop here
        return remapped_pulse

    omega = pulse.omega

    if pulse.is_cached('total_phases'):
        remapped_pulse.cache_total_phases(omega, pulse.get_total_phases(omega))

    if pulse.is_cached('F'):
        remapped_pulse.cache_filter_function(
            omega,
            F=pulse.get_filter_function(omega)[n_sort_idx][:, n_sort_idx]
        )

    if pulse.is_cached('total_Q_liouville') or pulse.is_cached('R'):
        if pulse.basis.btype != 'Pauli':
            warn('pulse does not have a separable basis which is needed to ' +
                 'retain cached control matrices.')

            return remapped_pulse

        permutation = remap_pauli_basis_elements(order, N)[None, :]
        if pulse.is_cached('total_Q_liouville'):
            remapped_pulse.total_Q_liouville = np.empty_like(
                pulse.total_Q_liouville
            )
            remapped_pulse.total_Q_liouville[permutation.T, permutation] = \
                pulse.total_Q_liouville

        if pulse.is_cached('R'):
            pulse_R = pulse.get_control_matrix(omega)
            remapped_R = np.empty_like(pulse_R)
            remapped_R[:, permutation[0]] = pulse_R
            remapped_pulse.cache_control_matrix(omega,
                                                R=remapped_R[n_sort_idx])

    return remapped_pulse


def extend(pulse_to_qubit_mapping: PulseMapping,
           N: Optional[int] = None,
           d_per_qubit: int = 2,
           additional_noise_Hamiltonian: Optional[Hamiltonian] = None,
           cache_diagonalization: Optional[bool] = None,
           cache_filter_function: Optional[bool] = None,
           omega: Optional[Coefficients] = None,
           show_progressbar: bool = False) -> PulseSequence:
    r"""
    Map one or more pulse sequences to different qubits.

    Parameters
    ----------
    pulse_to_qubit_mapping : sequence of mapping tuples
        A sequence of tuples with the first entry a ``PulseSequence``
        instance and the second an ``int`` or tuple of ``int``\s indicating the
        qubits that the ``PulseSequence`` should be mapped to. A mapping of
        operator identifiers may optionally be given as a third element of each
        tuple. By default, the index of the qubit the operator is mapped to is
        appended to its identifier.

        Pulse sequences defined for multiple qubits may also be extended
        to non-neighboring qubits. Note that for multi-qubit pulses the order
        of the qubits is respected, i.e. mapping a pulse to (1, 0) is different
        from mapping it to (0, 1).
    N : int
        The total number of qubits the new ``PulseSequence`` should be defined
        for. By default, this is inferred from ``pulse_to_qubit_mapping``.
    d_per_qubit : int
        The size of the Hilbert space a single qubit requires.
    additional_noise_Hamiltonian : list of lists
        Additional noise operators and corresponding sensitivities for the new
        pulse sequence.
    cache_diagonalization : bool
        Force diagonalizing the new pulse sequence. By default, diagonalization
        is cached if all pulses in ``pulse_to_qubit_mapping`` have been
        diagonalized since it is much cheaper to get the relevant quantities
        as tensor products from the mapped pulses instead of diagonalizing the
        new pulse.
    cache_filter_function : bool
        Force computing the filter functions for the new pulse sequence. Noise
        operators of individual pulses will be extended to the new Hilbert
        space. By default, this is done if all pulses in
        ``pulse_to_qubit_mapping`` have their filter functions cached.

        Note that extending the filter functions is only possible if they the
        mapped pulses are using a separable basis like the Pauli basis.
    omega : array_like
        Frequencies for which to compute the filter functions if
        ``cache_filter_function == True``. Defaults to ``None``, in which case
        the cached frequencies of the individual pulses need to be the same.
    show_progressbar : bool
        Show a progress bar for the calculation of the control matrix.

    Returns
    -------
    newpulse : PulseSequence
        The new pulse sequence on the larger qubit register. The noise
        operators (and possibly filter functions) are stored in the following
        order: first those of the multi-qubit pulses in the order they appeared
        in ``pulse_to_qubit_mapping``, then those of the single-qubit pulses,
        and lastly any additional ones that may be given by
        ``additional_noise_Hamiltonian``.

    Examples
    --------
    >>> import filter_functions as ff
    >>> I, X, Y, Z = ff.util.P_np
    >>> X_pulse = ff.PulseSequence([[X, [np.pi/2], 'X']],
    ...                            [[X, [1], 'X'], [Z, [1], 'Z']],
    ...                            [1], basis=ff.Basis.pauli(1))
    >>> XX_pulse = ff.extend([(X_pulse, 0), (X_pulse, 1)])
    >>> XX_pulse.d
    4
    >>> XIX_pulse_1 = ff.extend([(X_pulse, 0), (X_pulse, 2)])
    >>> XIX_pulse_1.d
    8
    >>> XXI_pulse = ff.extend([(X_pulse, 0), (X_pulse, 1)], N=3)
    >>> XXI_pulse.d
    8

    Filter functions are automatically cached if they are for mapped pulses:

    >>> omega = ff.util.get_sample_frequencies(X_pulse)
    >>> X_pulse.cache_filter_function(omega)
    >>> XX_pulse = ff.extend([(X_pulse, 0), (X_pulse, 1)])
    >>> XX_pulse.is_cached('F')
    True

    This behavior can also be overriden manually:

    >>> XX_pulse = ff.extend([(X_pulse, 0), (X_pulse, 1)],
    ...                      cache_filter_function=False)
    >>> XX_pulse.is_cached('F')
    False

    Mapping pulses to non-neighboring qubits is also possible:

    >>> Y_pulse = ff.PulseSequence([[Y, [np.pi/2], 'Y']],
    ...                            [[Y, [1], 'Y'], [Z, [1], 'Z']],
    ...                            [1], basis=ff.Basis.pauli(1))
    >>> XXY_pulse = ff.extend([(XX_pulse, (0, 1)), (Y_pulse, 2)])
    >>> XYX_pulse = ff.extend([(XX_pulse, (0, 2)), (Y_pulse, 1)])

    Additionally, pulses can have the order of the qubits they are defined for
    permuted (see :func:`remap`):

    >>> Z_pulse = ff.PulseSequence([[Z, [np.pi/2], 'Z']], [[Z, [1], 'Z']],
    ...                            [1], basis=ff.Basis.pauli(1))
    >>> XY_pulse = ff.extend([(X_pulse, 0), (Y_pulse, 1)])
    >>> YZX_pulse = ff.extend([(XY_pulse, (2, 0)), (Z_pulse, 1)])

    Control and noise operator identifiers can be mapped according to a
    specified mapping:

    >>> YX_pulse = ff.extend([(X_pulse, 1, {'X': 'IX', 'Z': 'IZ'}),
    ...                       (Y_pulse, 0, {'Y': 'YI', 'Z': 'ZI'})])
    >>> YX_pulse.c_oper_identifiers
    array(['IX', 'YI'], dtype='<U2')
    >>> YX_pulse.n_oper_identifiers
    array(['IX', 'IZ', 'YI', 'ZI'], dtype='<U2')

    We can also add an additional noise Hamiltonian:

    >>> H_n = [[ff.util.tensor(Z, Z, Z), [1], 'ZZZ']]
    >>> XYX_pulse = ff.extend([(XX_pulse, (0, 2)), (Y_pulse, 1)],
    ...                       additional_noise_Hamiltonian=H_n)
    >>> 'ZZZ' in XYX_pulse.n_oper_identifiers
    True

    See Also
    --------
    :func:`remap`

    :func:`concatenate`

    :func:`concatenate_periodic`
    """
    # Parse pulse_to_qubit_mapping
    active_qubits_list = []
    single_qubit_pulses = []
    multi_qubit_pulses = []
    single_qubit_identifier_mappings = []
    multi_qubit_identifier_mappings = []
    single_qubit_idx = []
    multi_qubit_idx = []

    # Unpack and pack again
    pulses, *args = zip_longest(*pulse_to_qubit_mapping, fillvalue=None)
    if len(args) == 1:
        qubits = args[0]
        identifier_mappings = [None]*len(qubits)
    elif len(args) == 2:
        qubits = args[0]
        identifier_mappings = list(args[1])
    else:
        qubits = args[0]
        identifier_mappings = list(args[1])

    for pulse, qubit, id_mapping in zip(pulses, qubits, identifier_mappings):
        try:
            active_qubits_list.extend(qubit)
            if len(qubit) == 1:
                single_qubit_idx.extend(qubit)
                single_qubit_pulses.append(pulse)
                single_qubit_identifier_mappings.append(id_mapping)
            else:
                # sort the qubit tuple and get the sorting indices
                sorted_qubit, order = zip(*sorted(
                    zip(qubit, range(len(qubit)))
                ))
                if qubit == sorted_qubit:
                    # No need to remap
                    sorted_pulse = pulse
                else:
                    # remap the pulse in the given order
                    try:
                        sorted_pulse = remap(pulse, order, d_per_qubit)
                    except ValueError as err:
                        raise ValueError(
                            'Could not remap {} mapped '.format(repr(pulse)) +
                            'to qubits {}. Do the dimensions '.format(qubit) +
                            'match?'
                        ) from err

                multi_qubit_idx.append(list(sorted_qubit))
                multi_qubit_pulses.append(sorted_pulse)
                multi_qubit_identifier_mappings.append(id_mapping)
        except TypeError:
            # qubit is not iterable, ie single qubit
            active_qubits_list.append(int(qubit))
            single_qubit_idx.append(int(qubit))
            single_qubit_pulses.append(pulse)
            single_qubit_identifier_mappings.append(id_mapping)

    if not all(pulse.d == d_per_qubit for pulse in single_qubit_pulses):
        raise ValueError('Not all single-qubit pulses have dimension ' +
                         'd_per_qubit = {}.'.format(d_per_qubit))

    if not all(pulse.d == d_per_qubit**len(qubits)
               for pulse, qubits in zip(multi_qubit_pulses, multi_qubit_idx)):
        raise ValueError('Not all multi-qubit pulses have correct dimension!')

    # Get lists of pulses in deterministic order
    pulses = multi_qubit_pulses + single_qubit_pulses
    idx = multi_qubit_idx + single_qubit_idx

    if not all_array_equal((pulse.dt for pulse in pulses)):
        raise ValueError('All pulses should be defined on the same time steps')

    active_qubits = set(active_qubits_list)
    if len(active_qubits) != len(active_qubits_list):
        raise ValueError('Qubit clash: multiple pulses mapped to same qubit!')

    last_qubit = max(active_qubits)
    if N is None:
        N = last_qubit + 1
    else:
        if last_qubit + 1 > N:
            raise ValueError('Number of qubits N smaller than highest qubit ' +
                             'index + 1 = {}'.format(last_qubit + 1))

    if len(pulse_to_qubit_mapping) == 1:
        # return input pulse if not mapped to another qubit
        if multi_qubit_idx:
            if N == len(multi_qubit_idx[0]):
                warn('Single multi-qubit pulse given and mapped to its ' +
                     'original qubits. Returning the same.')
                return multi_qubit_pulses[0]
        if single_qubit_idx:
            if N == 1:
                warn('Single single-qubit pulse given and mapped to its ' +
                     'original qubit. Returning the same.')
                return single_qubit_pulses[0]

    if cache_filter_function is not False:
        # Cache filter function of extended pulse if cached before for the same
        # frequencies
        is_cached = all(pulse.is_cached('R') for pulse in pulses)

        try:
            equal_omega = all_array_equal((pulse.omega for pulse in pulses))
        except AttributeError:
            equal_omega = False

        if cache_filter_function is None:
            # automatically decide whether to cache filter function
            cache_filter_function = is_cached and equal_omega
            if cache_filter_function:
                omega = pulses[0].omega
        else:
            # cache_filter_function == True
            if omega is None:
                if not equal_omega:
                    raise ValueError('Filter function should be cached but ' +
                                     'omega was not provided and could not ' +
                                     'be inferred.')

                omega = pulses[0].omega

    if cache_diagonalization is None:
        if cache_filter_function and additional_noise_Hamiltonian is not None:
            # Need the diagonalization
            cache_diagonalization = True
        else:
            # Extend propagators, eigenvalue and -vector arrays if calculated
            cache_diagonalization = all(pulse.is_cached(attr)
                                        for attr in ('HD', 'HV', 'Q')
                                        for pulse in pulses)
    elif (cache_diagonalization is False and
          additional_noise_Hamiltonian is not None):
        raise ValueError('Additional noise Hamiltonian given and ' +
                         'cache_diagonalization set to False but required.')

    # Multi-qubit opers and coeffs
    all_qubits = {q for q in range(N)}
    d = d_per_qubit**N
    n_dt = len(pulses[0].dt)
    ID = np.identity(d_per_qubit)

    c_opers, c_oper_identifiers, c_coeffs = [], [], []
    n_opers, n_oper_identifiers, n_coeffs = [], [], []
    for pulse, qubits, id_mapping in zip(multi_qubit_pulses, multi_qubit_idx,
                                         multi_qubit_identifier_mappings):
        pos = [bisect.bisect(qubits, q) for q in all_qubits.difference(qubits)]

        # map the identifiers
        c_oper_identifier, _ = _map_identifiers(*_default_extend_mapping(
            pulse.c_oper_identifiers, id_mapping, qubits
        ))
        n_oper_identifier, _ = _map_identifiers(*_default_extend_mapping(
            pulse.n_oper_identifiers, id_mapping, qubits
        ))

        c_oper_identifiers.extend(c_oper_identifier)
        n_oper_identifiers.extend(n_oper_identifier)
        c_opers.extend(tensor_insert(pulse.c_opers, *[ID]*len(pos), pos=pos,
                                     arr_dims=[[d_per_qubit]*len(qubits)]*2))
        n_opers.extend(tensor_insert(pulse.n_opers, *[ID]*len(pos), pos=pos,
                                     arr_dims=[[d_per_qubit]*len(qubits)]*2))

        c_coeffs.extend(pulse.c_coeffs)
        n_coeffs.extend(pulse.n_coeffs)

    # Single-qubit opers and coeffs
    for pulse, qubit, id_mapping in zip(single_qubit_pulses, single_qubit_idx,
                                        single_qubit_identifier_mappings):
        ID_pre = [np.identity(d_per_qubit**qubit)] if qubit > 0 else []
        ID_post = [np.identity(d_per_qubit**(N - qubit - 1))] \
            if qubit < N - 1 else []

        # map the identifiers
        c_oper_identifier, _ = _map_identifiers(*_default_extend_mapping(
            pulse.c_oper_identifiers, id_mapping, qubit
        ))
        n_oper_identifier, _ = _map_identifiers(*_default_extend_mapping(
            pulse.n_oper_identifiers, id_mapping, qubit
        ))

        c_oper_identifiers.extend(c_oper_identifier)
        n_oper_identifiers.extend(n_oper_identifier)
        # extend control and noise operators
        c_opers.extend(tensor(*(ID_pre + [pulse.c_opers] + ID_post)))
        n_opers.extend(tensor(*(ID_pre + [pulse.n_opers] + ID_post)))
        c_coeffs.extend(pulse.c_coeffs)
        n_coeffs.extend(pulse.n_coeffs)

    # Add optional additional noise Hamiltonian
    if additional_noise_Hamiltonian is not None:
        noise_args = _parse_Hamiltonian(
            additional_noise_Hamiltonian, len(pulses[0].dt), 'H_n'
        )
        add_n_opers, add_n_oper_id, add_n_coeffs = noise_args

        if add_n_opers.shape[1:] != (d, d):
            raise ValueError('Expected additional noise operators to have ' +
                             'dimensions {}, '.format((d, d)) +
                             'not {}.'.format(add_n_opers.shape[1:]))
        if len(set(add_n_oper_id)) != len(add_n_oper_id):
            raise ValueError('Found duplicate noise operator identifiers')

        n_opers.extend(add_n_opers)
        n_coeffs.extend(add_n_coeffs)
        n_oper_identifiers.extend(add_n_oper_id)

    pulse_btypes = list(set(pulse.basis.btype for pulse in pulses))
    if not len(pulse_btypes) == 1:
        warn('Not all pulses had the same basis type. Cannot retain cached ' +
             'control matrices.')
        basis = Basis.ggm(d_per_qubit**N)
    else:
        btype = pulse_btypes[0]
        if btype == 'GGM':
            warn('Original pulses had GGM basis which is not separable into ' +
                 'a tensor product. Cannot retain cached control matrices.')
            basis = Basis.ggm(d_per_qubit**N)
        elif btype == 'Pauli':
            basis = Basis.pauli(N)
        elif btype == 'Custom':
            warn('Original pulses had custom basis which I cannot extend.')
            basis = Basis.ggm(d_per_qubit**N)

    # Sort the identifiers
    c_sort_idx = np.argsort(c_oper_identifiers)
    n_sort_idx = np.argsort(n_oper_identifiers)

    newpulse = PulseSequence(
        c_opers=np.asarray(c_opers)[c_sort_idx],
        n_opers=np.asarray(n_opers)[n_sort_idx],
        c_oper_identifiers=np.asarray(c_oper_identifiers)[c_sort_idx],
        n_oper_identifiers=np.asarray(n_oper_identifiers)[n_sort_idx],
        c_coeffs=np.asarray(c_coeffs)[c_sort_idx],
        n_coeffs=np.asarray(n_coeffs)[n_sort_idx],
        dt=pulses[0].dt,
        t=pulses[0].t,
        d=d,
        basis=basis
    )

    if newpulse.basis.btype != 'Pauli':
        # Cannot do any extensions
        if cache_diagonalization:
            newpulse.diagonalize()
        if cache_filter_function:
            newpulse.cache_filter_function(omega)

        return newpulse

    if cache_diagonalization:
        HD = np.zeros((n_dt, d_per_qubit**N))
        HV, Q = None, None
        # registers keeps track of the qubits already tensored
        registers = None

        for pulse, qubits in zip(multi_qubit_pulses, multi_qubit_idx):
            # Insert ones into HD at these positions
            HD_pos = [bisect.bisect(qubits, q)
                      for q in all_qubits.difference(qubits)]

            HD += tensor_insert(pulse.HD, *np.ones((len(HD_pos), d_per_qubit)),
                                pos=HD_pos, rank=1,
                                arr_dims=[[d_per_qubit]*len(qubits)])

            (HV, Q), registers = _merge_attrs([HV, Q], [pulse.HV, pulse.Q],
                                              d_per_qubit, registers, qubits)

        for pulse, qubit in zip(single_qubit_pulses, single_qubit_idx):
            # For single qubit pulses we can just use normal tensor for HD
            ones_pre = [np.ones(d_per_qubit**qubit)] if qubit > 0 else []
            ones_post = [np.ones(d_per_qubit**(N - qubit - 1))] \
                if qubit < N - 1 else []
            HD += tensor(*(ones_pre + [pulse.HD] + ones_post), rank=1)

            (HV, Q), registers = _insert_attrs([HV, Q], [pulse.HV, pulse.Q],
                                               d_per_qubit, registers, qubit)

        # Fill up registers no qubits have been mapped to with identities
        ID_idx = list(all_qubits.difference(active_qubits))
        if ID_idx:
            (HV, Q), registers = _merge_attrs(
                [HV, Q], [np.eye(d_per_qubit**len(ID_idx))]*2, d_per_qubit,
                registers, ID_idx
            )

        # Set the new pulses's attributes
        newpulse.HD = HD
        newpulse.HV = HV
        newpulse.Q = Q
        # Set total propagator (easier to just grab after Q has been calculated
        # than tensor separately)
        newpulse.total_Q = Q[-1]
    elif all(pulse.is_cached('total_Q') for pulse in pulses):
        total_Q = None
        # registers keeps track of the qubits already tensored
        registers = None

        for pulse, qubits in zip(multi_qubit_pulses, multi_qubit_idx):
            (total_Q,), registers = _merge_attrs([total_Q], [pulse.total_Q],
                                                 d_per_qubit, registers,
                                                 qubits)

        for pulse, qubit in zip(single_qubit_pulses, single_qubit_idx):
            (total_Q,), registers = _insert_attrs([total_Q], [pulse.total_Q],
                                                  d_per_qubit, registers,
                                                  qubit)

        # Fill up registers no qubits have been mapped to with identities
        ID_idx = list(all_qubits.difference(active_qubits))
        if ID_idx:
            (total_Q,), registers = _merge_attrs(
                [total_Q], [np.eye(d_per_qubit**len(ID_idx))]*2,
                d_per_qubit, registers, ID_idx
            )

        newpulse.total_Q = total_Q

    if cache_filter_function:
        newpulse.omega = omega

        n_nops_new = len(newpulse.n_opers)
        R = np.zeros((n_nops_new, (d_per_qubit**N)**2, len(omega)),
                     dtype=complex)
        F = np.zeros((n_nops_new, n_nops_new, len(omega)), dtype=complex)
        n_ops_counter = 0
        for ind, pulse in zip(idx, pulses):
            n_nops = len(pulse.n_opers)
            ind = [ind] if isinstance(ind, int) else ind
            # Indices in newpulse.basis of the pulse.basis elements
            basis_idx = equivalent_pauli_basis_elements(ind, N)
            # Indices in newpulse.n_opers of the pulse.n_opers elements
            n_oper_idx = slice(n_ops_counter, n_ops_counter + n_nops)
            n_ops_counter += n_nops

            # Need to scale the control matrix and filter function
            scaling_factor = d_per_qubit**(N - len(ind))
            R[n_oper_idx, basis_idx] = pulse.get_control_matrix(
                omega, show_progressbar=show_progressbar
            )*np.sqrt(scaling_factor)
            F[n_oper_idx, n_oper_idx] = pulse.get_filter_function(
                omega, show_progressbar=show_progressbar
            )*scaling_factor

        if additional_noise_Hamiltonian is not None:
            newpulse_n_oper_inds = get_indices_from_identifiers(
                newpulse, n_oper_identifiers[n_ops_counter:], 'noise'
            )
            R[n_ops_counter:] = calculate_control_matrix_from_scratch(
                newpulse.HD, newpulse.HV, newpulse.Q, omega, newpulse.basis,
                newpulse.n_opers[newpulse_n_oper_inds],
                newpulse.n_coeffs[newpulse_n_oper_inds],
                newpulse.dt, newpulse.t, show_progressbar=show_progressbar
            )

            F[n_ops_counter:, n_ops_counter:] = calculate_filter_function(
                R[n_ops_counter:]
            )

        newpulse.cache_total_phases(omega)
        newpulse.total_Q_liouville = liouville_representation(newpulse.total_Q,
                                                              newpulse.basis)
        newpulse.cache_control_matrix(omega, R=R[n_sort_idx])
        newpulse.cache_filter_function(omega, F=F[n_sort_idx][:, n_sort_idx])

    return newpulse

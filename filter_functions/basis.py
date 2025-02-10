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
This module defines the Basis class, a subclass of NumPy's ndarray, to
represent operator bases.

Classes
-------
:class:`Basis`
    The operator basis as an array of  shape (d**2, d, d) with d the
    dimension of the Hilbert space

Functions
---------
:func:`normalize`
    Function to normalize a ``Basis`` instance
:func:`expand`
    Function to expand an array of operators in a given basis
:func:`ggm_expand`
    Fast function to expand an array of operators in a Generalized
    Gell-Mann basis

"""
from functools import cached_property
from itertools import product
from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np
import opt_einsum as oe
from numpy import linalg as nla
from scipy import linalg as sla
from sparse import COO

from . import util

__all__ = ['Basis', 'expand', 'ggm_expand', 'normalize']


class Basis(np.ndarray):
    r"""
    Class for operator bases. There are several ways to instantiate a
    Basis object:

        - by just calling this constructor with a (possibly incomplete)
          array of basis matrices. No checks regarding orthonormality or
          hermiticity are performed.

        - by calling one of the classes alternative constructors
          (classmethods):

              - :meth:`pauli`: Pauli operator basis
              - :meth:`ggm`: Generalized Gell-Mann basis
              - :meth:`from_partial` Generate an complete basis from
                partial elements

          These bases guarantee the following properties:

              - hermitian
              - orthonormal
              - [traceless] (can be controlled by a flag)

    Since Basis is a subclass of NumPy's ``ndarray``, it inherits all of
    its attributes, e.g. ``shape``. The following attributes behave
    slightly differently to a ndarray, however

        - ``A == B`` is ``True`` if all elements evaluate almost equal,
          i.e. equivalent to ``np.allclose(A, B)``.
        - ``basis.T`` transposes the last two axes of ``basis``. For a
          full basis, this corresponds to transposing each element
          individually. For a basis element, it corresponds to normal
          transposition.

    Parameters
    ----------
    basis_array: array_like, shape (n, d, d)
        An array or list of square matrices that are elements of an
        operator basis spanning :math:`\mathbb{C}^{d\times d}`. *n*
        should be smaller than or equal to *d**2*.
    traceless: bool, optional (default: auto)
        Controls whether a traceless basis is forced. Here, traceless
        means that the first element of the basis is the identity and
        the remaining elements are matrices of trace zero. If an element
        of ``basis_array`` is neither traceless nor the identity and
        ``traceless == True``, an exception will be raised. Defaults to
        ``True`` if basis_array is traceless and ``False`` if not.
    btype: str, optional (default: ``'custom'``)
        A string describing the basis type. For example, a basis created
        by the factory method :meth:`pauli` has *btype* 'pauli'.
    labels: sequence of str, optional
        A list of labels for the individual basis elements. Defaults to
        'C_0', 'C_1', ...

    Attributes
    ----------
    Other than the attributes inherited from ``ndarray``, a ``Basis``
    instance has the following attributes:

    btype: str
        Basis type.
    labels: sequence of str
        The labels for the basis elements.
    d: int
        Dimension of the space spanned by the basis.
    H: Basis
        Hermitian conjugate.
    isherm: bool
        If the basis is hermitian.
    isorthonorm: bool
        If the basis is orthonormal.
    istraceless: bool
        If the basis is traceless except for an identity element
    iscomplete: bool
        If the basis is complete, ie spans the full space.
    sparse: COO, shape (n, d, d)
        Representation in the COO format supplied by the ``sparse``
        package.
    four_element_traces: COO, shape (n, n, n, n)
        Traces over all possible combinations of four elements of self.
        This is required for the calculation of the error transfer
        matrix and thus cached in the Basis instance.

    Most of the attributes above are properties which are lazily
    evaluated and cached.

    Methods
    -------
    Other than the methods inherited from ``ndarray``, a ``Basis``
    instance has the following methods:

    normalize(b)
        Normalizes the basis (used internally when creating a basis from
        elements)
    tidyup(eps_scale=None)
        Cleans up floating point errors in-place to make zeros actual
        zeros. ``eps_scale`` is an optional argument multiplied to the
        data type's ``eps`` to get the absolute tolerance.

    """

    def __new__(cls, basis_array: Sequence, traceless: Optional[bool] = None,
                btype: Optional[str] = None, labels: Optional[Sequence[str]] = None) -> 'Basis':
        """Constructor."""
        if not hasattr(basis_array, '__getitem__'):
            raise TypeError('Invalid data type. Must be array_like')

        if isinstance(basis_array, cls):
            basis = basis_array
        else:
            try:
                # Allow single 2d element
                if len(basis_array.shape) == 2:
                    basis_array = [basis_array]
            except AttributeError:
                pass

            basis = util.parse_operators(basis_array, 'basis_array')
            if basis.shape[0] > np.prod(basis.shape[1:]):
                raise ValueError('Given overcomplete set of basis matrices. '
                                 'Not linearly independent.')

        basis = basis.view(cls)
        basis.btype = btype or 'Custom'
        basis.d = basis.shape[-1]
        if labels is not None:
            if len(labels) != len(basis):
                raise ValueError(f'Got {len(labels)} basis labels but expected {len(basis)}')
            basis.labels = labels
        else:
            basis.labels = [f'$C_{{{i}}}$' for i in range(len(basis))]

        return basis

    def __array_finalize__(self, basis: 'Basis') -> None:
        """Required for subclassing ndarray."""
        if basis is None:
            return

        self.btype = getattr(basis, 'btype', 'Custom')
        self.labels = getattr(basis, 'labels', [f'$C_{{{i}}}$' for i in range(len(basis))])
        self.d = getattr(basis, 'd', basis.shape[-1])
        self._eps = np.finfo(complex).eps
        self._atol = self._eps*self.d**3
        self._rtol = 0

    def __eq__(self, other: object) -> bool:
        """Compare for equality."""
        try:
            if self.shape != other.shape:
                return False
        except AttributeError:
            # Not ndarray
            return np.equal(self, other)

        return np.allclose(self.view(np.ndarray), other.view(np.ndarray),
                           atol=self._atol, rtol=self._rtol)

    def __contains__(self, item: np.ndarray) -> bool:
        """Implement 'in' operator."""
        return any(np.isclose(item.view(np.ndarray), self.view(np.ndarray),
                              rtol=self._rtol, atol=self._atol).all(axis=(1, 2)))

    def __array_wrap__(self, arr, context=None, return_scalar=False):
        """
        Fixes problem that ufuncs return 0-d arrays instead of scalars.

        https://github.com/numpy/numpy/issues/5819#issue-72454838
        """
        try:
            return super().__array_wrap__(arr, context, return_scalar=True)
        except TypeError:
            if arr.ndim:
                # Numpy < 2
                return np.ndarray.__array_wrap__(self, arr, context)

    def _print_checks(self) -> None:
        """Print checks for debug purposes."""
        checks = ('isherm', 'istraceless', 'iscomplete', 'isorthonorm')
        for check in checks:
            print(check, ':\t', getattr(self, check))

    def _invalidate_cached_properties(self):
        for attr in {'isherm', 'isnorm', 'isorthogonal', 'istraceless', 'iscomplete'}:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    @cached_property
    def isherm(self) -> bool:
        """Returns True if all basis elements are hermitian."""
        return self.H == self

    @cached_property
    def isnorm(self) -> bool:
        """Returns True if all basis elements are normalized."""
        return self.normalize(copy=True) == self

    @cached_property
    def isorthogonal(self) -> bool:
        """Returns True if all basis elements are mutually orthogonal."""
        if self.ndim == 2 or len(self) == 1:
            return True

        # The basis is orthogonal iff the matrix consisting of all d**2
        # elements written as d**2-dimensional column vectors is
        # orthogonal.
        dim = self.shape[0]
        U = self.reshape((dim, -1))
        actual = U.conj() @ U.T
        atol = self._eps*(self.d**2)**3
        mask = np.identity(dim, dtype=bool)
        return np.allclose(actual[..., ~mask].view(np.ndarray), 0, atol=atol, rtol=self._rtol)

    @property
    def isorthonorm(self) -> bool:
        """Returns True if basis is orthonormal."""
        return self.isorthogonal and self.isnorm

    @cached_property
    def istraceless(self) -> bool:
        """
        Returns True if basis is traceless except for possibly the identity.
        """
        trace = np.einsum('...jj', self)
        trace = util.remove_float_errors(trace, self.d**2)
        nonzero = np.atleast_1d(trace).nonzero()
        if nonzero[0].size == 0:
            return True
        elif nonzero[0].size == 1:
            # Single element has nonzero trace, check if (proportional to)
            # identity
            elem = self[nonzero][0].view(np.ndarray) if self.ndim == 3 else self.view(np.ndarray)
            offdiag_nonzero = elem[~np.eye(self.d, dtype=bool)].nonzero()
            diag_equal = np.diag(elem) == elem[0, 0]
            if diag_equal.all() and not offdiag_nonzero[0].any():
                # Element is (proportional to) the identity, this we define
                # as 'traceless' since a complete basis cannot have only
                # traceless elems.
                return True
            else:
                # Element not the identity, therefore not traceless
                return False
        else:
            return False

    @cached_property
    def iscomplete(self) -> bool:
        """Returns True if basis is complete."""
        A = self.reshape(self.shape[0], -1)
        rank = np.linalg.matrix_rank(A)
        return rank == self.d**2

    @property
    def H(self) -> 'Basis':
        """Return the basis hermitian conjugated element-wise."""
        return self.T.conj()

    @property
    def T(self) -> 'Basis':
        """Return the basis transposed element-wise."""
        if self.ndim >= 2:
            return self.swapaxes(-1, -2)

        return self

    @cached_property
    def sparse(self) -> COO:
        """Return the basis as a sparse COO array"""
        return COO.from_numpy(self)

    @cached_property
    def four_element_traces(self) -> COO:
        r"""
        Return all traces of the form
        :math:`\mathrm{tr}(C_i C_j C_k C_l)` as a sparse COO array for
        :math:`i,j,k,l > 0` (i.e. excluding the identity).
        """
        # Most of the traces are zero, therefore store the result in a
        # sparse array. For GGM bases, which are inherently sparse, it
        # makes sense for any dimension to also calculate with sparse
        # arrays. For Pauli bases, which are very dense, this is not so
        # efficient but unavoidable for d > 12.
        path = [(0, 1), (0, 1), (0, 1)]
        if self.btype == 'Pauli' and self.d <= 12:
            # For d == 12, the result is ~270 MB.
            return COO.from_numpy(oe.contract('iab,jbc,kcd,lda->ijkl', *(self,)*4, optimize=path))
        else:
            return oe.contract('iab,jbc,kcd,lda->ijkl', *(self.sparse,)*4, backend='sparse',
                               optimize=path)

    def expand(self, M: np.ndarray, hermitian: bool = False, traceless: bool = False,
               tidyup: bool = False) -> np.ndarray:
        """Expand matrices M in this basis.

        Parameters
        ----------
        M: array_like
            The square matrix (d, d) or array of square matrices (..., d, d)
            to be expanded in *basis*
        hermitian: bool (default: False)
            If M is hermitian along its last two axes, the result will be
            real.
        tidyup: bool {False}
            Whether to set values below the floating point eps to zero.

        See Also
        --------
        expand : The function corresponding to this method.
        """
        if self.btype == 'GGM' and self.iscomplete:
            return ggm_expand(M, traceless, hermitian, tidyup)
        return expand(M, self, self.isnorm, hermitian, tidyup)

    def normalize(self, copy: bool = False) -> Union[None, 'Basis']:
        """Normalize the basis."""
        if copy:
            return normalize(self)

        self /= _norm(self)
        self._invalidate_cached_properties()

    def tidyup(self, eps_scale: Optional[float] = None) -> None:
        """Wraps util.remove_float_errors."""
        if eps_scale is None:
            atol = self._atol
        else:
            atol = self._eps*eps_scale

        self.real[np.abs(self.real) <= atol] = 0
        self.imag[np.abs(self.imag) <= atol] = 0

        self._invalidate_cached_properties()

    @classmethod
    def pauli(cls, n: int) -> 'Basis':
        r"""
        Returns a Pauli basis for :math:`n` qubits, i.e. the basis spans
        the space :math:`\mathbb{C}^{d\times d}` with :math:`d = 2^n`:

        .. math::
            \mathcal{P} = \{I, X, Y, Z\}^{\otimes n}.

        The elements :math:`\sigma_i` are normalized with respect to the
        Hilbert-Schmidt inner product,

        .. math::

            \langle\sigma_i,\sigma_j\rangle
                &= \mathrm{Tr}\,\sigma_i^\dagger\sigma_j \\
                &= \delta_{ij}.

        Parameters
        ----------
        n: int
            The number of qubits.

        Returns
        -------
        basis: Basis
            The Basis object representing the Pauli basis.
        """
        normalization = np.sqrt(2**n)
        combinations = np.indices((4,)*n).reshape(n, 4**n)
        sigma = util.tensor(*util.paulis[combinations], rank=2)
        sigma /= normalization
        return cls(sigma, btype='Pauli',
                   labels=[''.join(tup) for tup in product(['I', 'X', 'Y', 'Z'], repeat=n)])

    @classmethod
    def ggm(cls, d: int) -> 'Basis':
        r"""
        Returns a generalized Gell-Mann basis in :math:`d` dimensions
        [Bert08]_ where the elements :math:`\Lambda_i` are normalized
        with respect to the Hilbert-Schmidt inner product,

        .. math::

            \langle\Lambda_i,\Lambda_j\rangle
                &= \mathrm{Tr}\,\Lambda_i^\dagger\Lambda_j \\
                &= \delta_{ij}.

        Parameters
        ----------
        d: int
            The dimensionality of the space spanned by the basis

        Returns
        -------
        basis: Basis
            The Basis object representing the GGM.

        References
        ----------
        .. [Bert08]
            Bertlmann, R. A., & Krammer, P. (2008). Bloch vectors for
            qudits. Journal of Physics A: Mathematical and Theoretical,
            41(23). https://doi.org/10.1088/1751-8113/41/23/235303

        """
        n_sym = int(d*(d - 1)/2)
        sym_rng = np.arange(1, n_sym + 1)
        diag_rng = np.arange(1, d)

        # Indices for offdiagonal elements
        j = np.repeat(np.arange(d - 1), np.arange(d - 1, 0, -1))
        k = np.arange(1, n_sym+1) - (j*(2*d - j - 3)/2).astype(int)
        j_offdiag = tuple(j)
        k_offdiag = tuple(k)
        # Indices for diagonal elements
        j_diag = tuple(i for l in range(d) for i in range(l))
        l_diag = tuple(i for i in range(1, d))

        inv_sqrt2 = 1/np.sqrt(2)
        Lambda = np.zeros((d**2, d, d), dtype=complex)
        Lambda[0] = np.eye(d)/np.sqrt(d)
        # First n matrices are symmetric
        Lambda[sym_rng, j_offdiag, k_offdiag] = inv_sqrt2
        Lambda[sym_rng, k_offdiag, j_offdiag] = inv_sqrt2
        # Second n matrices are antisymmetric
        Lambda[sym_rng+n_sym, j_offdiag, k_offdiag] = -1j*inv_sqrt2
        Lambda[sym_rng+n_sym, k_offdiag, j_offdiag] = 1j*inv_sqrt2
        # Remaining matrices have entries on the diagonal only
        Lambda[np.repeat(diag_rng, diag_rng)+2*n_sym, j_diag, j_diag] = 1
        Lambda[diag_rng + 2*n_sym, l_diag, l_diag] = -diag_rng
        # Normalize
        Lambda[2*n_sym + 1:, range(d), range(d)] /= np.tile(
            np.sqrt(diag_rng*(diag_rng + 1))[:, None], (1, d)
        )

        return cls(Lambda, btype='GGM', labels=[rf'$\Lambda_{{{i}}}$' for i in range(len(Lambda))])

    @classmethod
    def from_partial(cls, partial_basis_array: Sequence,
                     traceless: Optional[bool] = None,
                     btype: Optional[str] = None,
                     labels: Optional[Sequence[str]] = None) -> 'Basis':
        r"""Generate complete and orthonormal basis from a partial set.

        The basis is completed using singular value decomposition to
        determine the null space of the expansion coefficients of the
        partial basis with respect to another complete basis.

        Parameters
        ----------
        partial_basis_array: array_like
            A sequence of basis elements.
        traceless: bool, optional
            If a traceless basis should be generated (i.e. the first
            element is the identity and all the others have trace zero).
        btype: str, optional
            A custom identifier.
        labels: Sequence[str], optional
            A list of custom labels for each element. If
            `len(labels) == len(partial_basis_array)`, the newly created
            elements get labels 'C_i'.

        Returns
        -------
        basis: Basis, shape (d**2, d, d)
            The orthonormal basis.

        Raises
        ------
        ValueError
            If the given elements are not orthonormal.
        ValueError
            If the given elements are not traceless but
            `traceless==True`.
        ValueError
            If not len(partial_basis_array) or d**2 labels were given.

        """
        if btype is None:
            btype = 'From partial'

        if (labels is None
                and hasattr(partial_basis_array, 'labels')
                and len(partial_basis_array.labels) == len(partial_basis_array)):
            # Need to check if labels and array are same length as indexing
            # is unaware of our custom attributes
            labels = partial_basis_array.labels

        basis, labels = _full_from_partial(partial_basis_array, traceless, labels)
        return cls(basis, btype=btype, labels=labels)


def _full_from_partial(elems: Sequence, traceless: bool, labels: Sequence[str]) -> Basis:
    """
    Internal function to parse the basis elements *elems*. By default,
    checks are performed for orthogonality and linear independence. If
    either fails an exception is raised. Returns a full hermitian and
    orthonormal basis.
    """
    # Convert elems to basis to have access to its handy attributes
    elems = Basis(elems)
    elems.normalize(copy=False)

    if not elems.isherm:
        warn("(Some) elems not hermitian! The resulting basis also won't be.")

    if not elems.isorthogonal:
        raise ValueError("The basis elements are not orthogonal!")

    if traceless is None:
        traceless = elems.istraceless
    elif traceless and not elems.istraceless:
        raise ValueError("The basis elements are not traceless (up to an identity element) "
                         + "but a traceless basis was requested!")

    if labels is not None and len(labels) not in (len(elems), elems.d**2):
        raise ValueError(f'Got {len(labels)} labels but expected {len(elems)} or {elems.d**2}')

    # Get a Generalized Gell-Mann basis to expand in (fulfills the desired
    # properties hermiticity and orthonormality, and therefore also linear
    # combinations, ie basis expansions, of it will). Split off the identity so
    # that for traceless bases we can put it in the front.
    ggm = Basis.ggm(elems.d)
    coeffs = ggm.expand(elems, traceless=traceless, hermitian=elems.isherm, tidyup=True)

    if traceless:
        Id, ggm = np.split(ggm, [1])
        coeffs = coeffs[..., 1:]

    # Throw out coefficient vectors that are all zero (should only happen for
    # the identity)
    coeffs = coeffs[(coeffs != 0).any(axis=-1)]
    if coeffs.size != 0:
        # Get d**2 - len(coeffs) vectors spanning the nullspace of coeffs.
        # Those together with coeffs span the whole space, and therefore also
        # the linear combinations of GGMs weighted with the coefficients will
        # span the whole matrix space
        coeffs = np.concatenate((coeffs, sla.null_space(coeffs).T))
        # Our new basis is given by linear combinations of GGMs with coeffs
        basis = np.einsum('ij,jkl', coeffs, ggm)
    else:
        # Resulting array is of size zero, i.e. we can just return the GGMs
        basis = ggm

    # Add the identity again and normalize the new basis
    if traceless:
        basis = np.concatenate((Id, basis)).view(Basis)
    else:
        basis = basis.view(Basis)

    # Clean up
    basis.tidyup()

    if labels is not None and len(labels) == len(elems):
        # Fill up labels for newly generated elements
        labels = list(labels)
        if traceless:
            # sort Identity label to the front, default to first if not found
            # (should not happen since traceless checks that it is present)
            id_idx = next((i for i, elem in enumerate(elems)
                           if np.allclose(Id.view(np.ndarray), elem.view(np.ndarray),
                                          rtol=elems._rtol, atol=elems._atol)), 0)
            labels.insert(0, labels.pop(id_idx))

        labels.extend('$C_{{{}}}$'.format(i) for i in range(len(labels), len(basis)))

    return basis, labels


def _norm(b: Sequence) -> np.ndarray:
    """Frobenius norm with two singleton dimensions inserted at the end."""
    b = np.asanyarray(b)
    norm = nla.norm(b, axis=(-1, -2))
    return norm[..., None, None]


def normalize(b: Basis) -> Basis:
    r"""
    Return a copy of the basis *b* normalized with respect to the
    Frobenius norm [Gol85]_:

        :math:`||A||_F = \left[\sum_{i,j} |a_{i,j}|^2\right]^{1/2}`

    or equivalently, with respect to the Hilbert-Schmidt inner product
    as implemented by :func:`~filter_functions.util.dot_HS`.

    References
    ----------
    .. [Gol85]
        G. H. Golub and C. F. Van Loan, *Matrix Computations*,
        Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    """
    return (b/_norm(b)).squeeze().reshape(b.shape).view(Basis)


def expand(M: Union[np.ndarray, Basis], basis: Union[np.ndarray, Basis],
           normalized: bool = True, hermitian: bool = False, tidyup: bool = False) -> np.ndarray:
    r"""
    Expand the array *M* in the basis given by *basis*.

    Parameters
    ----------
    M: array_like
        The square matrix (d, d) or array of square matrices (..., d, d)
        to be expanded in *basis*
    basis: array_like
        The basis of shape (m, d, d) in which to expand.
    normalized: bool {True}
        Wether the basis is normalized.
    hermitian: bool (default: False)
        If M is hermitian along its last two axes, the result will be
        real.
    tidyup: bool {False}
        Whether to set values below the floating point eps to zero.

    Returns
    -------
    coefficients: ndarray
        The coefficient array with shape (..., m) or (m,) if *M* was 2-d

    Notes
    -----
    For an orthogonal matrix basis :math:`\mathcal{C} = \big\{C_k\in
    \mathbb{C}^{d\times d}: \langle C_k,C_l\rangle_\mathrm{HS} \propto
    \delta_{kl}\big\}_{k=0}^{d^2-1}` with the Hilbert-Schmidt inner
    product as implemented by :func:`~filter_functions.util.dot_HS` and
    :math:`M\in\mathbb{C}^{d\times d}`, the expansion of
    :math:`M` in terms of :math:`\mathcal{C}` is given by

    .. math::
        M &= \sum_j c_j C_j, \\
        c_j &= \frac{\mathrm{tr}\big(M C_j\big)}
                    {\mathrm{tr}\big(C_j^\dagger C_j\big)}.

    """

    def cast(arr):
        return arr.real if hermitian and basis.isherm else arr

    coefficients = cast(np.tensordot(M, basis, axes=[(-2, -1), (-1, -2)]))
    if not normalized:
        coefficients /= cast(np.einsum('bij,bji->b', basis, basis))

    return util.remove_float_errors(coefficients) if tidyup else coefficients


def ggm_expand(M: Union[np.ndarray, Basis], traceless: bool = False,
               hermitian: bool = False, tidyup: bool = False) -> np.ndarray:
    r"""
    Expand the matrix *M* in a Generalized Gell-Mann basis [Bert08]_.
    This function makes use of the explicit construction prescription of
    the basis and thus makes do without computing the expansion
    coefficients as the overlap between the matrix and each basis
    element.

    Parameters
    ----------
    M: array_like
        The square matrix (d, d) or array of square matrices (..., d, d)
        to be expanded in a GGM basis.
    traceless: bool (default: False)
        Include the basis element proportional to the identity in the
        expansion. If it is known beforehand that M is traceless, the
        corresponding coefficient is zero and thus doesn't need to be
        computed.
    hermitian: bool (default: False)
        If M is hermitian along its last two axes, the result will be
        real.
    tidyup: bool {False}
        Whether to set values below the floating point eps to zero.

    Returns
    -------
    coefficients: ndarray
        The real coefficient array with shape (d**2,) or (..., d**2)

    References
    ----------
    .. [Bert08]
        Bertlmann, R. A., & Krammer, P. (2008). Bloch vectors for
        qudits. Journal of Physics A: Mathematical and Theoretical,
        41(23). https://doi.org/10.1088/1751-8113/41/23/235303
    """
    if M.shape[-1] != M.shape[-2]:
        raise ValueError('M should be square in its last two axes')

    def cast(arr):
        return arr.real if hermitian else arr

    # Squeeze out an extra dimension to be shape agnostic
    square = M.ndim < 3
    if square:
        M = M[None, ...]

    d = M.shape[-1]

    n_sym = int(d*(d - 1)/2)
    sym_rng = np.arange(1, n_sym + 1)
    diag_rng = np.arange(1, d)

    # Map linear index to index tuple of upper triangle
    j = np.repeat(np.arange(d - 1), np.arange(d - 1, 0, -1))
    k = np.arange(1, n_sym+1) - (j*(2*d - j - 3)/2).astype(int)
    offdiag_idx = [tuple(j), tuple(k)]
    # Indices for upper triangular part of M
    triu_idx = tuple([...] + offdiag_idx)
    # Indices for lower triangular part of M
    tril_idx = tuple([...] + offdiag_idx[::-1])
    # Indices for diagonal elements of M up to first to last
    diag_idx = tuple([...] + [tuple(i for i in range(d - 1))]*2)
    # Indices of diagonal elements of M starting from first
    diag_idx_shifted = tuple([...] + [tuple(i for i in range(1, d))]*2)

    # Compute the coefficients
    coeffs = np.zeros((*M.shape[:-2], d**2), dtype=float if hermitian else complex)
    if not traceless:
        # First element is proportional to the trace of M
        coeffs[..., 0] = cast(M.trace(0, -1, -2))/np.sqrt(d)

    # Elements proportional to the symmetric GGMs
    coeffs[..., sym_rng] = cast(M[triu_idx] + M[tril_idx])/np.sqrt(2)
    # Elements proportional to the antisymmetric GGMs
    coeffs[..., sym_rng + n_sym] = cast(1j*(M[triu_idx] - M[tril_idx]))/np.sqrt(2)
    # Elements proportional to the diagonal GGMs
    coeffs[..., diag_rng + 2*n_sym] = cast(M[diag_idx].cumsum(axis=-1)
                                           - diag_rng*M[diag_idx_shifted])
    coeffs[..., diag_rng + 2*n_sym] /= cast(np.sqrt(diag_rng*(diag_rng + 1)))

    if square:
        coeffs = coeffs.squeeze()
    if tidyup:
        coeffs = util.remove_float_errors(coeffs)
    return coeffs


def equivalent_pauli_basis_elements(idx: Union[Sequence[int], int], N: int) -> np.ndarray:
    """
    Get the indices of the equivalent (up to identities tensored to it)
    basis elements of Pauli bases of qubits at position idx in the total
    Pauli basis for N qubits.
    """
    idx = [idx] if isinstance(idx, int) else idx
    multi_index = np.ix_(*[range(4) if i in idx else [0]
                           for i in range(N)])
    elem_idx = np.ravel_multi_index(multi_index, [4]*N).ravel()
    return elem_idx


def remap_pauli_basis_elements(order: Sequence[int], N: int) -> np.ndarray:
    """
    For a N-qubit Pauli basis, transpose the order of the subsystems and
    return the indices that permute the old basis to the new.
    """
    # Index tuples for single qubit paulis that give the n-qubit paulis when
    # tensored together
    pauli_idx = np.indices((4,)*N).reshape(N, 4**N).T
    # Indices of the N-qubit basis reordered according to order
    linear_idx = [np.ravel_multi_index([idx_tup[i] for i in order], (4,)*N)
                  for idx_tup in pauli_idx]

    return np.array(linear_idx)

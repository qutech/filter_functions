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

from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np
import opt_einsum as oe
from numpy import linalg as nla
from numpy.core import ndarray
from scipy import linalg as sla
from sparse import COO

from . import util

__all__ = ['Basis', 'expand', 'ggm_expand', 'normalize']


class Basis(ndarray):
    r"""
    Class for operator bases. There are several ways to instantiate a
    Basis object:

        - by just calling this constructor with a (possibly incomplete)
          array of basis matrices. In this case, it is attempted to
          construct from the input a complete basis with the following
          properties that retains all original (input) elements:

              - hermitian
              - orthonormal
              - [traceless] (can be controlled by a flag)

        - by calling one of the classes alternative constructors
          (classmethods) for predefined bases:

              - :meth:`pauli`: Pauli operator basis
              - :meth:`ggm`: Generalized Gell-Mann basis
              - :meth:`partial` (not implemented)

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
    skip_check: bool, optional (default: ``False``)
        Skip the internal routine for checking ``basis_array``'s
        orthonormality and completeness. Use with caution.

    Attributes
    ----------
    Other than the attributes inherited from ``ndarray``, a ``Basis``
    instance has the following attributes:

    btype: str
        Basis type.
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
        Normalizes the basis in-place (used internally when creating a
        basis from elements)
    tidyup(eps_scale=None)
        Cleans up floating point errors in-place to make zeros actual
        zeros. ``eps_scale`` is an optional argument multiplied to the
        data type's ``eps`` to get the absolute tolerance.

    """

    def __new__(cls, basis_array: Sequence, traceless: Optional[bool] = None,
                btype: Optional[str] = None, skip_check: bool = False) -> 'Basis':
        """Constructor."""
        if not skip_check:
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

                basis = np.empty((len(basis_array), *basis_array[0].shape), dtype=complex)
                for i, elem in enumerate(basis_array):
                    if isinstance(elem, ndarray):   # numpy array
                        basis[i] = elem
                    elif hasattr(elem, 'full'):     # qutip.Qobj
                        basis[i] = elem.full()
                    elif hasattr(elem, 'todense'):  # sparse array
                        basis[i] = elem.todense()
                    else:
                        raise TypeError('At least one element invalid type!')

            d = basis.shape[-1]

            if len(basis) > d**2:
                raise ValueError('Given overcomplete set of basis matrices. '
                                 'Not linearly independent.')

            basis = _full_from_partial(basis, traceless)
        else:
            basis = np.asanyarray(basis_array)
            d = basis.shape[-1]

        basis = basis.view(cls)
        basis.btype = btype or 'Custom'
        basis.d = d
        return basis

    def __array_finalize__(self, basis: 'Basis') -> None:
        """Required for subclassing ndarray."""
        if basis is None:
            return

        self.btype = getattr(basis, 'btype', 'Custom')
        self.d = getattr(basis, 'd', basis.shape[-1])
        self._sparse = None
        self._four_element_traces = None
        self._isherm = None
        self._isorthonorm = None
        self._istraceless = None
        self._iscomplete = None
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

        return np.allclose(self.view(ndarray), other.view(ndarray),
                           atol=self._atol, rtol=self._rtol)

    def __contains__(self, item: ndarray) -> bool:
        """Implement 'in' operator."""
        return any(np.all(np.isclose(item.view(ndarray), self.view(ndarray),
                                     rtol=self._rtol, atol=self._atol),
                          axis=(1, 2)))

    def __array_wrap__(self, out_arr, context=None):
        """
        Fixes problem that ufuncs return 0-d arrays instead of scalars.

        https://github.com/numpy/numpy/issues/5819#issue-72454838
        """
        if out_arr.ndim:
            return ndarray.__array_wrap__(self, out_arr, context)

    def _print_checks(self) -> None:
        """Print checks for debug purposes."""
        checks = ('isherm', 'istraceless', 'iscomplete', 'isorthonorm')
        for check in checks:
            print(check, ':\t', getattr(self, check))

    @property
    def isherm(self) -> bool:
        """Returns True if all basis elements are hermitian."""
        if self._isherm is None:
            self._isherm = (self.H == self)

        return self._isherm

    @property
    def isorthonorm(self) -> bool:
        """Returns True if basis is orthonormal."""
        if self._isorthonorm is None:
            # All the basis is orthonormal iff the matrix consisting of all
            # d**2 elements written as d**2-dimensional column vectors is
            # unitary.
            if self.ndim == 2:
                # Only one basis element
                self._isorthonorm = True
            else:
                # Size of the result after multiplication
                dim = self.shape[0]
                U = self.reshape((dim, -1))
                actual = U.conj() @ U.T
                target = np.identity(dim)
                atol = self._eps*(self.d**2)**3
                self._isorthonorm = np.allclose(actual.view(ndarray), target,
                                                atol=atol, rtol=self._rtol)

        return self._isorthonorm

    @property
    def istraceless(self) -> bool:
        """
        Returns True if basis is traceless except for possibly the identity.
        """
        if self._istraceless is None:
            trace = np.einsum('...jj', self)
            trace = util.remove_float_errors(trace, self.d)
            nonzero = trace.nonzero()
            if nonzero[0].size == 0:
                self._istraceless = True
            elif nonzero[0].size == 1:
                # Single element has nonzero trace, check if (proportional to)
                # identity
                elem = self[nonzero][0].view(ndarray)
                offdiag_nonzero = elem[~np.eye(self.d, dtype=bool)].nonzero()
                diag_equal = np.diag(elem) == elem[0, 0]
                if diag_equal.all() and not offdiag_nonzero[0].any():
                    # Element is (proportional to) the identity, this we define
                    # as 'traceless' since a complete basis cannot have only
                    # traceless elems.
                    self._istraceless = True
                else:
                    # Element not the identity, therefore not traceless
                    self._istraceless = False
            else:
                self._istraceless = False

        return self._istraceless

    @property
    def iscomplete(self) -> bool:
        """Returns True if basis is complete."""
        if self._iscomplete is None:
            A = self.reshape(self.shape[0], -1)
            rank = np.linalg.matrix_rank(A)
            self._iscomplete = rank == self.d**2

        return self._iscomplete

    @property
    def H(self) -> 'Basis':
        """Return the basis hermitian conjugated element-wise."""
        return self.T.conj()

    @property
    def T(self) -> 'Basis':
        """Return the basis transposed element-wise."""
        if self.ndim == 3:
            return self.transpose(0, 2, 1)

        if self.ndim == 2:
            return self.transpose(1, 0)

        return self

    @property
    def sparse(self) -> COO:
        """Return the basis as a sparse COO array"""
        if self._sparse is None:
            self._sparse = COO.from_numpy(self)

        return self._sparse

    @property
    def four_element_traces(self) -> COO:
        r"""
        Return all traces of the form
        :math:`\mathrm{tr}(C_i C_j C_k C_l)` as a sparse COO array for
        :math:`i,j,k,l > 0` (i.e. excluding the identity).
        """
        if self._four_element_traces is None:
            # Most of the traces are zero, therefore store the result in a
            # sparse array. For GGM bases, which are inherently sparse, it
            # makes sense for any dimension to also calculate with sparse
            # arrays. For Pauli bases, which are very dense, this is not so
            # efficient but unavoidable for d > 12.
            path = [(0, 1), (0, 1), (0, 1)]
            if self.btype == 'Pauli' and self.d <= 12:
                # For d == 12, the result is ~270 MB.
                self._four_element_traces = COO.from_numpy(oe.contract('iab,jbc,kcd,lda->ijkl',
                                                                       *(self,)*4, optimize=path))
            else:
                self._four_element_traces = oe.contract('iab,jbc,kcd,lda->ijkl', *(self.sparse,)*4,
                                                        backend='sparse', optimize=path)

        return self._four_element_traces

    @four_element_traces.setter
    def four_element_traces(self, traces):
        self._four_element_traces = traces

    def normalize(self) -> None:
        """Normalize the basis in-place"""
        if self.ndim == 2:
            self /= nla.norm(self)
        elif self.ndim == 3:
            np.einsum('ijk,i->ijk', self, 1/nla.norm(self, axis=(1, 2)),
                      out=self)

    def tidyup(self, eps_scale: Optional[float] = None) -> None:
        """Wraps util.remove_float_errors."""
        if eps_scale is None:
            atol = self._atol
        else:
            atol = self._eps*eps_scale

        self.real[np.abs(self.real) <= atol] = 0
        self.imag[np.abs(self.imag) <= atol] = 0

    @classmethod
    def partial(cls, basis_array, subspace_inds):
        raise NotImplementedError

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
        return cls(sigma, btype='Pauli', skip_check=True)

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

        return cls(Lambda, btype='GGM', skip_check=True)


def _full_from_partial(elems: Sequence, traceless: Union[None, bool]) -> Basis:
    """
    Internal function to parse the basis elements *elems*. By default,
    checks are performed for orthogonality and linear independence. If
    either fails an exception is raised. Returns a full hermitian and
    orthonormal basis.
    """
    elems = np.asanyarray(elems)
    if not isinstance(elems, Basis):
        # Convert elems to basis to have access to its handy attributes
        elems = normalize(elems.view(Basis))

    if not elems.isherm:
        warn("(Some) elems not hermitian! The resulting basis also won't be.")

    if not elems.isorthonorm:
        raise ValueError("The basis elements are not orthonormal!")

    if traceless is None:
        traceless = elems.istraceless
    else:
        if traceless and not elems.istraceless:
            raise ValueError("The basis elements are not traceless (up to " +
                             "an identity element) but a traceless basis " +
                             "was requested!")

    # Get a Generalized Gell-Mann basis to expand in (fulfills the desired
    # properties hermiticity and orthonormality, and therefore also linear
    # combinations, ie basis expansions, of it will). Split off the identity so
    # that for traceless bases we can put it in the front.
    if traceless:
        Id, ggm = np.split(Basis.ggm(elems.d), [1])
    else:
        ggm = Basis.ggm(elems.d)

    coeffs = expand(elems, ggm, tidyup=True)

    # Throw out coefficient vectors that are all zero (should only happen for
    # the identity)
    coeffs = coeffs[(coeffs != 0).any(axis=1)]
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

    return basis


def normalize(b: Sequence) -> Basis:
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
    b = np.asanyarray(b)
    if b.ndim == 2:
        return (b/nla.norm(b)).view(Basis)
    if b.ndim == 3:
        return np.einsum('ijk,i->ijk', b, 1/nla.norm(b, axis=(1, 2))).view(Basis)

    raise ValueError(f'Expected b.ndim to be either 2 or 3, not {b.ndim}.')


def expand(M: Union[ndarray, Basis], basis: Union[ndarray, Basis],
           normalized: bool = True, tidyup: bool = False) -> ndarray:
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
    coefficients = np.tensordot(M, basis, axes=[(-2, -1), (-1, -2)])

    if not normalized:
        coefficients /= np.einsum('bij,bji->b', basis, basis).real

    return util.remove_float_errors(coefficients) if tidyup else coefficients


def ggm_expand(M: Union[ndarray, Basis], traceless: bool = False) -> ndarray:
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

    Returns
    -------
    coefficients: ndarray
        The coefficient array with shape (d**2,) or (..., d**2)

    References
    ----------
    .. [Bert08]
        Bertlmann, R. A., & Krammer, P. (2008). Bloch vectors for
        qudits. Journal of Physics A: Mathematical and Theoretical,
        41(23). https://doi.org/10.1088/1751-8113/41/23/235303
    """
    if M.shape[-1] != M.shape[-2]:
        raise ValueError('M should be square in its last two axes')

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
    coeffs = np.zeros((*M.shape[:-2], d**2), dtype=complex)
    if not traceless:
        # First element is proportional to the trace of M
        coeffs[..., 0] = np.einsum('...jj', M)/np.sqrt(d)

    # Elements proportional to the symmetric GGMs
    coeffs[..., sym_rng] = (M[triu_idx] + M[tril_idx])/np.sqrt(2)
    # Elements proportional to the antisymmetric GGMs
    coeffs[..., sym_rng + n_sym] = 1j*(M[triu_idx] - M[tril_idx])/np.sqrt(2)
    # Elements proportional to the diagonal GGMs
    coeffs[..., diag_rng + 2*n_sym] = M[diag_idx].cumsum(axis=-1) - diag_rng*M[diag_idx_shifted]
    coeffs[..., diag_rng + 2*n_sym] /= np.sqrt(diag_rng*(diag_rng + 1))

    return coeffs.squeeze() if square else coeffs


def equivalent_pauli_basis_elements(idx: Union[Sequence[int], int], N: int) -> ndarray:
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


def remap_pauli_basis_elements(order: Sequence[int], N: int) -> ndarray:
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

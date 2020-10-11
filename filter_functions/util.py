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
This module provides various helper functions.

Functions
---------
:func:`abs2`
    Absolute value squared
:func:`get_indices_from_identifiers`
    The the indices of control or noise operators with given identifiers
    as they are saved in a ``PulseSequence``.
:func:`tensor`
    Fast, flexible tensor product of an arbitrary number of inputs using
    :func:`~numpy.einsum`
:func:`tensor_insert`
    For an array that is known to be a tensor product, insert arrays at
    a given position in the product chain
:func:`tensor_merge`
    For two arrays that are tensor products of known dimensions, merge
    them at arbitary positions in the product chain
:func:`tensor_transpose`
    For a tensor product, transpose the order of the constituents in the
    product chain
:func:`mdot`
    Multiple matrix product
:func:`remove_float_errors`
    Set entries whose absolute value is below a certain threshold to
    zero
:func:`oper_equiv`
    Determine if two vectors or operators are equal up to a global phase
:func:`dot_HS`
    Hilbert-Schmidt inner product
:func:`get_sample_frequencies`
    Get frequencies with typical infrared and ultraviolet cutoffs for a
    ``PulseSequence``
:func:`progressbar`
    A progress bar for loops. Uses tqdm if available and a simple custom
    one if not.
:func:`hash_array_along_axis`
    Return a list of hashes along a given axis
:func:`all_array_equal`
    Check if all arrays in an iterable are equal

Exceptions
----------
:class:`CalculationError`
    Exception raised if trying to fetch the pulse correlation function
    when it was not computed during concatenation

"""
import functools
import inspect
import json
import operator
import os
import re
import string
from itertools import zip_longest
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import linalg as nla
from numpy import ndarray

from .types import Operator, State

try:
    import jupyter_client
    import requests
    from notebook.notebookapp import list_running_servers
    from requests.compat import urljoin

    def _get_notebook_name() -> str:
        """
        Return the full path of the jupyter notebook.

        See https://github.com/jupyter/notebook/issues/1000

        Jupyter notebook is licensed as follows:

        This project is licensed under the terms of the Modified BSD License
        (also known as New or Revised or 3-Clause BSD), as follows:

        - Copyright (c) 2001-2015, IPython Development Team
        - Copyright (c) 2015-, Jupyter Development Team

        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are
        met:

        Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

        Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

        Neither the name of the Jupyter Development Team nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
        IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
        TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
        PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
        OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
        DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
        THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        """
        try:
            connection_file = jupyter_client.find_connection_file()
        except OSError:
            return ''

        kernel_id = re.search('kernel-(.*).json', connection_file).group(1)
        servers = list_running_servers()
        for ss in servers:
            response = requests.get(urljoin(ss['url'], 'api/sessions'),
                                    params={'token': ss.get('token', '')})
            for nn in json.loads(response.text):
                try:
                    if nn['kernel']['id'] == kernel_id:
                        try:
                            relative_path = nn['notebook']['path']
                            return os.path.join(ss['notebook_dir'], relative_path)
                        except KeyError:
                            return ''
                except TypeError:
                    return ''

        return ''

    _NOTEBOOK_NAME = _get_notebook_name()
except ImportError:
    _NOTEBOOK_NAME = ''

if _NOTEBOOK_NAME:
    from tqdm.notebook import tqdm as _tqdm
else:
    # Either not running notebook or not able to determine
    from tqdm import tqdm as _tqdm

__all__ = ['paulis', 'abs2', 'all_array_equal', 'dot_HS', 'get_sample_frequencies',
           'hash_array_along_axis', 'mdot', 'oper_equiv', 'progressbar', 'remove_float_errors',
           'tensor', 'tensor_insert', 'tensor_merge', 'tensor_transpose']

# Pauli matrices
paulis = np.array([
    [[1, 0],
     [0, 1]],
    [[0, 1],
     [1, 0]],
    [[0, -1j],
     [1j, 0]],
    [[1, 0],
     [0, -1]],
])


def abs2(x: ndarray) -> ndarray:
    r"""
    Fast function to calculate the absolute value squared,

    .. math::

        |\cdot|^2 := \Re(\cdot)^2 + \Im(\cdot)^2

    Equivalent to::

        np.abs(x)**2
    """
    return x.real**2 + x.imag**2


def cexp(x: ndarray, out=None, where=True) -> ndarray:
    r"""Fast complex exponential.

    Parameters
    ----------
    x: ndarray
        Argument of the complex exponential :math:`\exp(i x)`.

    Returns
    -------
    y: ndarray
        Complex exponential :math:`y = \exp(i x)`.

    References
    ----------
    https://software.intel.com/en-us/forums/intel-distribution-for-python/topic/758148
    """
    out = np.empty(x.shape, dtype=np.complex128) if out is None else out
    out.real = np.cos(x, out=out.real, where=where)
    out.imag = np.sin(x, out=out.imag, where=where)
    return out


def parse_optional_parameters(params_dict: Dict[str, Sequence]) -> Callable:
    """Decorator factory to parse optional parameter with certain legal values.

    For ``params_dict = {name: allowed, ...}``: If the parameter value
    corresponding to ``name`` (either in args or kwargs of the decorated
    function) is not contained in ``allowed`` a ``ValueError`` is raised.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            parameters = inspect.signature(func).parameters
            for name, allowed in params_dict.items():
                idx = tuple(parameters).index(name)
                try:
                    value = args[idx]
                except IndexError:
                    value = kwargs.get(name, parameters[name].default)

                if value not in allowed:
                    raise ValueError(f"Invalid value for {name}: {value}. " +
                                     f"Should be one of {allowed}.")

            return func(*args, **kwargs)
        return wrapper
    return decorator


parse_which_FF_parameter = parse_optional_parameters({'which': ('fidelity', 'generalized')})


def _tensor_product_shape(shape_A: Sequence[int], shape_B: Sequence[int], rank: int):
    """Get shape of the tensor product between A and B of rank rank"""
    broadcast_shape = ()
    # Loop over dimensions from last to first, filling the 'shorter' shape
    # with 1's once it is exhausted
    for dims in zip_longest(shape_A[-rank-1::-1], shape_B[-rank-1::-1], fillvalue=1):
        if 1 in dims:
            # Broadcast 1-d of argument to dimension of other
            broadcast_shape = (max(dims),) + broadcast_shape
        elif len(set(dims)) == 1:
            # Both arguments have same dimension on axis.
            broadcast_shape = dims[:1] + broadcast_shape
        else:
            raise ValueError(f'Incompatible shapes {shape_A} and {shape_B} ' +
                             f'for tensor product of rank {rank}.')

    # Shape of the actual tensor product is product of each dimension,
    # again broadcasting if need be
    tensor_shape = tuple(
        functools.reduce(operator.mul, dimensions)
        for dimensions in zip_longest(shape_A[:-rank-1:-1],
                                      shape_B[:-rank-1:-1],
                                      fillvalue=1)
    )[::-1]

    return broadcast_shape + tensor_shape


def _parse_dims_arg(name: str, dims: Sequence[Sequence[int]], rank: int) -> None:
    """Check if dimension arg for a tensor_* function is correct format"""
    if not len(dims) == rank:
        raise ValueError(f'{name}_dims should be of length rank = {rank}, not {len(dims)}')

    if not len(set(len(dim) for dim in dims)) == 1:
        # Not all nested lists the same length as required
        raise ValueError(f'Require all lists in {name}_dims to be of same length!')


@parse_optional_parameters({'kind': ('noise', 'control')})
def get_indices_from_identifiers(pulse: 'PulseSequence',
                                 identifiers: Union[None, str, Sequence[str]],
                                 kind: str) -> Tuple[Sequence[int],
                                                     Sequence[str]]:
    """Get the indices of operators for given identifiers.

    Parameters
    ----------
    pulse: PulseSequence
        The PulseSequence instance for which to get the indices.
    identifiers: str or sequence of str
        The identifiers whose indices to get.
    kind: str
        Whether to get 'control' or 'noise' operator indices.
    """
    if kind == 'noise':
        pulse_identifiers = pulse.n_oper_identifiers
    else:
        # kind == 'control'
        pulse_identifiers = pulse.c_oper_identifiers

    identifier_to_index_table = {identifier: index for index, identifier
                                 in enumerate(pulse_identifiers)}
    if identifiers is None:
        inds = np.arange(len(pulse_identifiers))
    else:
        try:
            if isinstance(identifiers, str):
                inds = np.array([identifier_to_index_table[identifiers]])
            else:
                inds = np.array([identifier_to_index_table[identifier]
                                 for identifier in identifiers])
        except KeyError:
            raise ValueError('Invalid identifiers given. All available ones ' +
                             f'are: {pulse_identifiers}')

    return inds


def tensor(*args, rank: int = 2, optimize: Union[bool, str] = False) -> ndarray:
    """
    Fast, flexible tensor product using einsum. The product is taken
    over the last *rank* axes and broadcast over the remaining axes
    which thus need to follow numpy broadcasting rules. Note that
    vectors are treated as rank 2 tensors with shape (1, x) or (x, 1).

    For example, the following shapes are compatible:

     - ``rank == 2`` (e.g. matrices or vectors)::

        (a, b, c, d, d), (a, b, c, e, e) -> (a, b, c, d*e, d*e)
        (a, b, c), (a, d, e) -> (a, b*d, c*e)
        (a, b), (c, d, e) -> (c, a*d, b*e)
        (1, a), (b, 1, c) -> (b, 1, a*c)
     - ``rank == 1``::

        (a, b), (a, c) -> (a, b*c)
        (a, b, 1), (a, c) -> (a, b, c)

    Parameters
    ----------
    args: array_like
        The elements of the tensor product
    rank: int, optional (default: 2)
        The rank of the tensors. E.g., for a Kronecker product between
        two matrices ``rank == 2``. The remaining axes are broadcast
        over.
    optimize: bool|str, optional (default: False)
        Optimize the tensor contraction order. Passed through to
        :func:`numpy.einsum`.

    Examples
    --------
    >>> Z = np.diag([1, -1])
    >>> np.array_equal(tensor(Z, Z), np.kron(Z, Z))
    True

    >>> A, B = np.arange(2), np.arange(2, 5)
    >>> tensor(A, B, rank=1)
    array([[0, 0, 0, 2, 3, 4]])

    >>> args = np.random.randn(4, 10, 3, 2)
    >>> result = tensor(*args, rank=1)
    >>> result.shape == (10, 3, 2**4)
    True
    >>> result = tensor(*args, rank=2)
    >>> result.shape == (10, 3**4, 2**4)
    True

    >>> A, B = np.random.randn(1, 3), np.random.randn(3, 4)
    >>> result = tensor(A, B)
    >>> result.shape == (1*3, 3*4)
    True

    >>> A, B = np.random.randn(3, 1, 2), np.random.randn(2, 2, 2)
    >>> try:
    ...     result = tensor(A, B, rank=2)
    ... except ValueError as err:  # cannot broadcast over axis 0
    ...     print(err)
    Incompatible shapes (3, 1, 2) and (2, 2, 2) for tensor product of rank 2.
    >>> result = tensor(A, B, rank=3)
    >>> result.shape == (3*2, 1*2, 2*2)
    True

    See Also
    --------
    numpy.kron: NumPy tensor product.
    tensor_insert: Insert array at given position in tensor product chain.
    tensor_merge: Merge tensor product chains.
    tensor_transpose: Transpose the order of a tensor product chain.
    """
    chars = string.ascii_letters
    # All the subscripts we need
    A_chars = chars[:rank]
    B_chars = chars[rank:2*rank]
    subscripts = '...{},...{}->...{}'.format(
        A_chars, B_chars, ''.join(i + j for i, j in zip(A_chars, B_chars))
    )

    def binary_tensor(A, B):
        """Compute the Kronecker product of two tensors"""
        # Add dimensions so that each arg has at least ndim == rank
        while A.ndim < rank:
            A = A[None, :]

        while B.ndim < rank:
            B = B[None, :]

        outshape = _tensor_product_shape(A.shape, B.shape, rank)
        return np.einsum(subscripts, A, B, optimize=optimize).reshape(outshape)

    # Compute the tensor products in a binary tree-like structure, calculating
    # the product of two leaves and working up. This is more memory-efficient
    # than reduce(binary_tensor, args) which computes the products
    # left-to-right.
    n = len(args)
    bit = n % 2
    while n > 1:
        args = args[:bit] + tuple(binary_tensor(*args[i:i+2]) for i in range(bit, n, 2))
        n = len(args)
        bit = n % 2

    return args[0]


def tensor_insert(arr: ndarray, *args, pos: Union[int, Sequence[int]],
                  arr_dims: Sequence[Sequence[int]], rank: int = 2,
                  optimize: Union[bool, str] = False) -> ndarray:
    r"""
    For a tensor product *arr*, insert *args* into the product chain at
    *pos*. E.g, if :math:`\verb|arr|\equiv A\otimes B\otimes C` and
    :math:`\verb|pos|\equiv 2`, the result will be the tensor product

    .. math::
        A\otimes B\otimes\left[\bigotimes_{X\in\verb|args|}X\right]
        \otimes C.

    This function works in a similar way to :func:`numpy.insert` and the
    following would be functionally equivalent in the case that the
    constituent tensors of the product *arr* are known:

    >>> tensor_insert(tensor(*arrs, rank=rank), *args, pos=pos, arr_dims=...,
    ...               rank=rank)

    >>> tensor(*np.insert(arrs, pos, args, axis=0), rank=rank)


    Parameters
    ----------
    arr: ndarray
        The tensor product in whose chain the other args should be
        inserted
    *args: ndarray
        The tensors to be inserted in the product chain
    pos: int|sequence of ints
        The position(s) at which the args are inserted in the product
        chain. If an int and ``len(args) > 1``, it is repeated so that
        all args are inserted in a row. If a sequence, it should
        indicate the indices in the original tensor product chain that
        led to *arr* before which *args* should be inserted.
    arr_dims: array_like, shape (rank, n_const)
        The last *rank* dimensions of the *n_const* constituent tensors
        of the tensor product *arr* as a list of lists with the list at
        position *i* containing the *i*-th relevant dimension of all
        args. Since the remaing axes are broadcast over, their shape is
        irrelevant.

        For example, if ``arr = tensor(a, b, c, rank=2)`` and ``a,b,c``
        have shapes ``(2, 3, 4), (5, 2, 2, 1), (2, 2)``,
        ``arr_dims = [[3, 2, 2], [4, 1, 2]]``.
    rank: int, optional (default: 2)
        The rank of the tensors. E.g., for a Kronecker product between
        two vectors, ``rank == 1``, and between two matrices
        ``rank == 2``. The remaining axes are broadcast over.
    optimize: bool|str, optional (default: False)
        Optimize the tensor contraction order. Passed through to
        :func:`numpy.einsum`.


    Examples
    --------
    >>> I, X, Y, Z = paulis
    >>> arr = tensor(X, I)
    >>> r = tensor_insert(arr, Y, Z, arr_dims=[[2, 2], [2, 2]], pos=0)
    >>> np.allclose(r, tensor(Y, Z, X, I))
    True
    >>> r = tensor_insert(arr, Y, Z, arr_dims=[[2, 2], [2, 2]], pos=1)
    >>> np.allclose(r, tensor(X, Y, Z, I))
    True
    >>> r = tensor_insert(arr, Y, Z, arr_dims=[[2, 2], [2, 2]], pos=2)
    >>> np.allclose(r, tensor(X, I, Y, Z))
    True

    Other ranks and different dimensions:

    >>> from numpy.random import randn
    >>> A, B, C = randn(2, 3, 1, 2), randn(2, 2, 2, 2), randn(3, 2, 1)
    >>> arr = tensor(A, C, rank=3)
    >>> r = tensor_insert(arr, B, pos=1, rank=3,
    ...                   arr_dims=[[3, 3], [1, 2], [2, 1]])
    >>> np.allclose(r, tensor(A, B, C, rank=3))
    True

    >>> arrs, args = randn(2, 2, 2), randn(2, 2, 2)
    >>> arr_dims = [[2, 2], [2, 2]]
    >>> r = tensor_insert(tensor(*arrs), *args, pos=(0, 1), arr_dims=arr_dims)
    >>> np.allclose(r, tensor(args[0], arrs[0], args[1], arrs[1]))
    True
    >>> r = tensor_insert(tensor(*arrs), *args, pos=(0, 0), arr_dims=arr_dims)
    >>> np.allclose(r, tensor(*args, *arrs))
    True
    >>> r = tensor_insert(tensor(*arrs), *args, pos=(1, 2), arr_dims=arr_dims)
    >>> np.allclose(r, tensor(*np.insert(arrs, (1, 2), args, axis=0)))
    True

    See Also
    --------
    numpy.insert: NumPy array insertion with similar syntax.
    numpy.kron: NumPy tensor product.
    tensor_insert: Insert array at given position in tensor product chain.
    tensor_merge: Merge tensor product chains.
    tensor_transpose: Transpose the order of a tensor product chain.
    """
    if len(args) == 0:
        raise ValueError('Require nonzero number of args!')

    if isinstance(pos, int):
        pos = (pos,)
        if len(args) > 1:
            # Inserting all args at same position, perform their tensor product
            # using tensor and insert the result instead of iteratively insert
            # one by one
            args = (tensor(*args, rank=rank, optimize=optimize),)
    else:
        if not len(pos) == len(args):
            raise ValueError('Expected pos to be either an int or a sequence of the same length ' +
                             f'as the number of args, not length {len(pos)}')

    _parse_dims_arg('arr', arr_dims, rank)

    def _tensor_insert_subscripts(ndim, pos, rank):
        """Get einsum string for the contraction"""
        ins_chars = string.ascii_letters[:rank]
        arr_chars = string.ascii_letters[rank:(ndim+1)*rank]
        subscripts = '...{},...{}->...{}'.format(
            ins_chars,
            arr_chars,
            arr_chars[:pos] + ''.join(
                ins_chars[i] + arr_chars[pos+i*ndim:pos+(i+1)*ndim]
                for i in range(rank)
            )
        )

        return subscripts

    def single_tensor_insert(arr, ins, arr_dims, pos):
        """Insert a single tensor *ins* into *arr* at position *pos*."""
        subscripts = _tensor_insert_subscripts(len(arr_dims[0]), pos, rank)
        outshape = _tensor_product_shape(ins.shape, arr.shape, rank)
        # Need to reshape arr to the rank*ndim-dimensional shape that's the
        # output of the regular tensor einsum call
        flat_arr_dims = [dim for axis in arr_dims for dim in axis]
        reshaped_arr = arr.reshape(*arr.shape[:-rank], *flat_arr_dims)
        result = np.einsum(subscripts, ins, reshaped_arr, optimize=optimize).reshape(outshape)

        return result

    # Insert args one after the other, starting at lowest index
    result = arr.copy()
    # Make a deep copy of arr_dims and pos as we modify them
    carr_dims = [list(axis[:]) for axis in arr_dims]
    cpos = list(pos).copy()
    # Number of constituent tensors of the tensor product arr
    ndim = len(arr_dims[0])
    divs, pos = zip(*[divmod(p, ndim) if p != ndim else (0, p) for p in pos])
    for i, (p, div, arg, arg_counter) in enumerate(sorted(
            zip_longest(pos, divs, args, range(len(args)), fillvalue=pos[0]),
            key=operator.itemgetter(0))):

        if div not in (-1, 0):
            raise IndexError(f'Invalid position {cpos[i]} specified. Must ' +
                             f'be between -{ndim} and {ndim}.')

        # Insert argument arg at position p+i (since every iteration the index
        # shifts by 1)
        try:
            result = single_tensor_insert(result, arg, carr_dims, p+i)
        except ValueError as err:
            raise ValueError(f'Could not insert arg {arg_counter} with shape {result.shape} ' +
                             f'into the array with shape {arg.shape} at position {p}.') from err

        # Update arr_dims
        for axis, d in zip(carr_dims, arg.shape[-rank:]):
            axis.insert(p, d)

    return result


def tensor_merge(arr: ndarray, ins: ndarray, pos: Sequence[int],
                 arr_dims: Sequence[Sequence[int]], ins_dims: Sequence[Sequence[int]],
                 rank: int = 2, optimize: Union[bool, str] = False) -> ndarray:
    r"""
    For two tensor products *arr* and *ins*, merge *ins* into the
    product chain at indices *pos*. E.g, if
    :math:`\verb|arr|\equiv A\otimes B\otimes C`,
    :math:`\verb|ins|\equiv D\otimes E`, and
    :math:`\verb|pos|\equiv [1, 2]`, the result will be the tensor
    product

    .. math::
        A\otimes D\otimes B\otimes E\otimes C.

    This function works in a similar way to :func:`numpy.insert` and
    :func:`tensor_insert`.

    Parameters
    ----------
    arr: ndarray
        The tensor product in whose chain the other args should be
        inserted
    ins: ndarray
        The tensor product to be inserted in the product chain
    pos: sequence of ints
        The positions at which the constituent tensors of *ins* are
        inserted in the product chain. Should indicate the indices in
        the original tensor product chain that led to *arr* before which
        the constituents of *ins* should be inserted.
    arr_dims: array_like, shape (rank, n_const)
        The last *rank* dimensions of the *n_const* constituent tensors
        of the tensor product *arr* as a list of lists with the list at
        position *i* containing the *i*-th relevant dimension of all
        args. Since the remaing axes are broadcast over, their shape is
        irrelevant.

        For example, if ``arr = tensor(a, b, c, rank=2)`` and ``a,b,c``
        have shapes ``(2, 3, 4), (5, 2, 2, 1), (2, 2)``,
        ``arr_dims = [[3, 2, 2], [4, 1, 2]]``.
    ins_dims: array_like, shape (rank, n_const)
        The last *rank* dimensions of the *n_const* constituent tensors
        of the tensor product *ins* as a list of lists with the list at
        position *i* containing the *i*-th relevant dimension of *ins*.
        Since the remaing axes are broadcast over, their shape is
        irrelevant.
    rank: int, optional (default: 2)
        The rank of the tensors. E.g., for a Kronecker product between
        two vectors, ``rank == 1``, and between two matrices
        ``rank == 2``. The remaining axes are broadcast over.
    optimize: bool|str, optional (default: False)
        Optimize the tensor contraction order. Passed through to
        :func:`numpy.einsum`.

    Examples
    --------
    >>> I, X, Y, Z = paulis
    >>> arr = tensor(X, Y, Z)
    >>> ins = tensor(I, I)
    >>> r1 = tensor_merge(arr, ins, pos=[1, 2], arr_dims=[[2]*3, [2]*3],
    ...                   ins_dims=[[2]*2, [2]*2])
    >>> np.allclose(r1, tensor(X, I, Y, I, Z))
    True
    >>> r2 = tensor_merge(ins, arr, pos=[0, 1, 2], arr_dims=[[2]*2, [2]*2],
    ...                   ins_dims=[[2]*3, [2]*3])
    >>> np.allclose(r1, r2)
    True

    :func:`tensor_insert` can provide the same functionality in some
    cases:

    >>> arr = tensor(Y, Z)
    >>> ins = tensor(I, X)
    >>> r1 = tensor_merge(arr, ins, pos=[0, 0], arr_dims=[[2]*2, [2]*2],
    ...                   ins_dims=[[2]*2, [2]*2])
    >>> r2 = tensor_insert(arr, I, X, pos=[0, 0], arr_dims=[[2]*2, [2]*2])
    >>> np.allclose(r1, r2)
    True

    Also tensors of rank other than 2 and numpy broadcasting are
    supported:

    >>> arr = np.random.randn(2, 10, 3, 4)
    >>> ins = np.random.randn(2, 10, 3, 2)
    >>> r = tensor_merge(tensor(*arr, rank=1), tensor(*ins, rank=1), [0, 1],
    ...                  arr_dims=[[4, 4]], ins_dims=[[2, 2]], rank=1)
    >>> np.allclose(r, tensor(ins[0], arr[0], ins[1], arr[1], rank=1))
    True

    See Also
    --------
    numpy.insert: NumPy array insertion with similar syntax.
    numpy.kron: NumPy tensor product.
    tensor: Fast tensor product with broadcasting.
    tensor_insert: Insert array at given position in tensor product chain.
    tensor_transpose: Transpose the order of a tensor product chain.
    """
    # Parse dimension args
    for arg_name, arg_dims in zip(('arr', 'ins'), (arr_dims, ins_dims)):
        _parse_dims_arg(arg_name, arg_dims, rank)

    ins_ndim = len(ins_dims[0])
    arr_ndim = len(arr_dims[0])
    ins_chars = string.ascii_letters[:ins_ndim*rank]
    arr_chars = string.ascii_letters[ins_ndim*rank:(ins_ndim+arr_ndim)*rank]
    out_chars = ''
    for r in range(rank):
        arr_part = arr_chars[r*arr_ndim:(r+1)*arr_ndim]
        ins_part = ins_chars[r*ins_ndim:(r+1)*ins_ndim]
        for i, (p, ins_p) in enumerate(sorted(zip(pos, ins_part))):
            if p != arr_ndim:
                div, p = divmod(p, arr_ndim)
                if div not in (-1, 0):
                    raise IndexError(f'Invalid position {pos[i]} specified. Must be between ' +
                                     f'-{arr_ndim} and {arr_ndim}.')
            arr_part = arr_part[:p+i] + ins_p + arr_part[p+i:]

        out_chars += arr_part

    subscripts = f'...{ins_chars},...{arr_chars}->...{out_chars}'

    outshape = _tensor_product_shape(ins.shape, arr.shape, rank)
    # Need to reshape arr to the rank*ndim-dimensional shape that's the
    # output of the regular tensor einsum call
    flat_arr_dims = [dim for axis in arr_dims for dim in axis]
    flat_ins_dims = [dim for axis in ins_dims for dim in axis]

    # Catch exceptions from wrong ins/arr_dims arguments
    try:
        ins_reshaped = ins.reshape(*ins.shape[:-rank], *flat_ins_dims)
    except ValueError as err:
        raise ValueError('ins_dims not compatible with ins.shape[-rank:] = ' +
                         f'{ins.shape[-rank:]}') from err
    try:
        arr_reshaped = arr.reshape(*arr.shape[:-rank], *flat_arr_dims)
    except ValueError as err:
        raise ValueError('arr_dims not compatible with arr.shape[-rank:] = ' +
                         f'{arr.shape[-rank:]}') from err

    result = np.einsum(subscripts, ins_reshaped, arr_reshaped, optimize=optimize).reshape(outshape)

    return result


def tensor_transpose(arr: ndarray, order: Sequence[int], arr_dims: Sequence[Sequence[int]],
                     rank: int = 2) -> ndarray:
    r"""
    Transpose the order of a tensor product chain.

    Parameters
    ----------
    arr: ndarray
        The tensor product whose chain should be reordered.
    order: sequence of ints
        The transposition order. If ``arr == tensor(A, B)`` and
        ``order == (1, 0)``, the result will be ``tensor(B, A)``.
    arr_dims: array_like, shape (rank, n_const)
        The last *rank* dimensions of the *n_const* constituent tensors
        of the tensor product *arr* as a list of lists with the list at
        position *i* containing the *i*-th relevant dimension of all
        args. Since the remaing axes are broadcast over, their shape is
        irrelevant.

        For example, if ``arr = tensor(a, b, c, rank=2)`` and ``a,b,c``
        have shapes ``(2, 3, 4), (5, 2, 2, 1), (2, 2)``,
        ``arr_dims = [[3, 2, 2], [4, 1, 2]]``.
    rank: int, optional (default: 2)
        The rank of the tensors. E.g., for a Kronecker product between
        two vectors, ``rank == 1``, and between two matrices
        ``rank == 2``. The remaining axes are broadcast over.

    Returns
    -------
    transposed_arr: ndarray
        The tensor product *arr* with its order transposed according to
        *order*

    Examples
    --------
    >>> I, X, Y, Z = paulis
    >>> arr = tensor(X, Y, Z)
    >>> transposed = tensor_transpose(arr, [1, 2, 0], arr_dims=[[2, 2, 2]]*2)
    >>> np.allclose(transposed, tensor(Y, Z, X))
    True

    See Also
    --------
    numpy.insert: NumPy array insertion with similar syntax.
    numpy.kron: NumPy tensor product.
    tensor: Fast tensor product with broadcasting.
    tensor_insert: Insert array at given position in tensor product chain.
    tensor_merge: Merge tensor product chains.
    """
    _parse_dims_arg('arr', arr_dims, rank)

    ndim = len(arr_dims[0])
    # Number of axes that are broadcast over
    n_broadcast = len(arr.shape[:-rank])
    transpose_axes = ([i for i in range(n_broadcast)] +
                      [n_broadcast + r*ndim + o for r in range(rank) for o in order])

    # Need to reshape arr to the rank*ndim-dimensional shape that's the
    # output of the regular tensor einsum call
    flat_arr_dims = [dim for axis in arr_dims for dim in axis]

    # Catch exceptions from wrong arr_dims argument
    try:
        arr_reshaped = arr.reshape(*arr.shape[:-rank], *flat_arr_dims)
    except ValueError as err:
        raise ValueError('arr_dims not compatible with arr.shape[-rank:] = ' +
                         f'{arr.shape[-rank:]}') from err

    try:
        result = arr_reshaped.transpose(*transpose_axes).reshape(arr.shape)
    except TypeError as type_err:
        raise TypeError("Could not transpose the order. Are all elements of " +
                        "'order' integers?") from type_err
    except ValueError as val_err:
        raise ValueError("Could not transpose the order. Are all elements " +
                         "of 'order' unique and match the array?") from val_err

    return result


def mdot(arr: Sequence, axis: int = 0) -> ndarray:
    """Multiple matrix products along axis"""
    return functools.reduce(np.matmul, np.swapaxes(arr, 0, axis))


def remove_float_errors(arr: ndarray, eps_scale: Optional[float] = None):
    """
    Clean up arr by removing floating point numbers smaller than the
    dtype's precision multiplied by eps_scale. Treats real and imaginary
    parts separately.
    """
    if eps_scale is None:
        atol = np.finfo(arr.dtype).eps*arr.shape[-1]
    else:
        atol = np.finfo(arr.dtype).eps*eps_scale

    # Hack around arr.imag sometimes not being writable
    if arr.dtype == complex:
        arr = arr.real + 1j*arr.imag
        arr.real[np.abs(arr.real) <= atol] = 0
        arr.imag[np.abs(arr.imag) <= atol] = 0
    else:
        arr = arr.real
        arr.real[np.abs(arr.real) <= atol] = 0

    return arr


def oper_equiv(psi: Union[Operator, State], phi: Union[Operator, State],
               eps: Optional[float] = None, normalized: bool = False) -> Tuple[bool, float]:
    r"""
    Checks whether psi and phi are equal up to a global phase, i.e.

    .. math::
        |\psi\rangle = e^{i\chi}|\phi\rangle \Leftrightarrow
        \langle \phi|\psi\rangle = e^{i\chi},

    and returns the phase. If the first return value is false, the
    second is meaningless in this context. psi and phi can also be
    operators.

    Parameters
    ----------
    psi, phi: qutip.Qobj or array_like
        Vectors or operators to be compared
    eps: float
        The tolerance below which the two objects are treated as equal,
        i.e., the function returns ``True`` if
        ``abs(1 - modulus) <= eps``.
    normalized: bool
        Flag indicating if *psi* and *phi* are normalized with respect
        to the Hilbert-Schmidt inner product :func:`dot_HS`.

    Examples
    --------
    >>> psi = paulis[1]
    >>> phi = paulis[1]*np.exp(1j*1.2345)
    >>> oper_equiv(psi, phi)
    (True, 1.2345)
    """
    # Convert qutip.Qobj's to numpy arrays
    psi, phi = [obj.full() if hasattr(obj, 'full') else obj for obj in (psi, phi)]

    psi, phi = np.atleast_2d(psi, phi)

    if eps is None:
        # Tolerance the floating point eps times the # of flops for the matrix
        # multiplication, i.e. for psi and phi n x m matrices 2*n**2*m
        eps = (max(np.finfo(psi.dtype).eps, np.finfo(phi.dtype).eps) *
               np.prod(psi.shape)*phi.shape[-1]*2)
        if not normalized:
            # normalization introduces more floating point error
            eps *= (np.prod(psi.shape[-2:])*phi.shape[-1]*2)**2

    try:
        inner_product = (psi.swapaxes(-1, -2).conj() @ phi).trace(0, -1, -2)
    except ValueError as err:
        raise ValueError('psi and phi have incompatible dimensions!') from err

    if normalized:
        norm = 1
    else:
        norm = nla.norm(psi, axis=(-1, -2))*nla.norm(phi, axis=(-1, -2))

    phase = np.angle(inner_product)
    modulus = abs(inner_product)

    return abs(norm - modulus) <= eps, phase


def dot_HS(U: Operator, V: Operator, eps: Optional[float] = None) -> float:
    r"""Return the Hilbert-Schmidt inner product of U and V,

    .. math::
        \langle U, V\rangle_\mathrm{HS} := \mathrm{tr}(U^\dagger V).

    Parameters
    ----------
    U, V: qutip.Qobj or ndarray
        Objects to compute the inner product of.

    Returns
    -------
    result: float, complex
        The result rounded to precision eps.

    Examples
    --------
    >>> U, V = paulis[1:3]
    >>> dot_HS(U, V)
    0.0
    >>> dot_HS(U, U)
    2.0
    """
    # Convert qutip.Qobj's to numpy arrays
    if hasattr(U, 'full'):
        U = U.full()
    if hasattr(V, 'full'):
        V = V.full()

    if eps is None:
        # Tolerance is the dtype precision times the number of flops for the
        # matrix multiplication times two to be on the safe side
        try:
            eps = np.finfo(U.dtype).eps*np.prod(U.shape)*V.shape[-1]*2
        except ValueError:
            # dtype is int and therefore exact
            eps = 0

    if eps == 0:
        decimals = 0
    else:
        decimals = abs(int(np.log10(eps)))

    res = np.round(np.einsum('...ij,...ij', U.conj(), V), decimals)
    return res if res.imag.any() else res.real


@parse_optional_parameters({'spacing': ('log', 'linear')})
def get_sample_frequencies(pulse: 'PulseSequence', n_samples: int = 300,
                           spacing: str = 'log',
                           include_quasistatic: bool = False) -> ndarray:
    """Get *n_samples* sample frequencies spaced 'linear' or 'log'.

    The ultraviolet cutoff is taken to be two orders of magnitude larger
    than the timescale of the pulse tau. In the case of log spacing, the
    values are clipped in the infrared at two orders of magnitude below
    the timescale of the pulse.

    Parameters
    ----------
    pulse: PulseSequence
        The pulse to get frequencies for.
    n_samples: int, optional
        The number of frequency samples. Default is 300.
    spacing: str, optional
        The spacing of the frequencies. Either 'log' or 'linear',
        default is 'log'.
    include_quasistatic: bool, optional
        Include zero frequency. Default is False.

    Returns
    -------
    omega: ndarray
        The frequencies.
    """
    if spacing == 'linear':
        xspace = np.linspace
    else:
        # spacing == 'log'
        xspace = np.geomspace

    if include_quasistatic:
        freqs = xspace(1e-2/pulse.tau, 1e2/pulse.tau, n_samples - 1)
        freqs = np.insert(freqs, 0, 0)
    else:
        freqs = xspace(1e-2/pulse.tau, 1e2/pulse.tau, n_samples)

    return 2*np.pi*freqs


def hash_array_along_axis(arr: ndarray, axis: int = 0) -> List[int]:
    """Return the hashes of arr along the first axis"""
    return [hash(arr.tobytes()) for arr in np.swapaxes(arr, 0, axis)]


def all_array_equal(it: Iterable) -> bool:
    """
    Return ``True`` if all array elements of ``it`` are equal by hashing
    the bytes representation of each array. Note that this is not
    thread-proof.
    """
    return len(set(hash(i.tobytes()) for i in it)) == 1


def progressbar(iterable: Iterable, *args, **kwargs):
    """
    Progress bar for loops. Uses tqdm.

    Usage::

        for i in progressbar(range(10)):
            do_something()
    """
    return _tqdm(iterable, *args, **kwargs)


def progressbar_range(*args, show_progressbar: bool = True, **kwargs):
    """Wrapper for range() that shows a progressbar dependent on a kwarg.

    Parameters
    ----------
    *args :
        Positional arguments passed through to :func:`range`.
    show_progressbar: bool, optional
        Return a range iterator with or without a progressbar.
    **kwargs :
        Keyword arguments passed through to :func:`progressbar`.

    Returns
    -------
    it: Iterator
        Range iterator dressed with a progressbar if
        ``show_progressbar=True``.
    """
    if show_progressbar:
        return progressbar(range(*args), **kwargs)

    return range(*args)


class CalculationError(Exception):
    """Indicates a quantity could not be computed."""

    def __init__(self, message: str) -> None:
        super().__init__(message)

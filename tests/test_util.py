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
This module tests the utility functions in util.py
"""
import numpy as np
import qutip as qt
from numpy.random import randint, randn, random
from tests import testutil

from filter_functions import PulseSequence, util


class UtilTest(testutil.TestCase):

    def test_abs2(self):
        x = randn(20, 100) + 1j*randn(20, 100)
        self.assertArrayAlmostEqual(np.abs(x)**2, util.abs2(x))

    def test_cexp(self):
        """Fast complex exponential."""
        x = randn(50, 100)
        a = util.cexp(x)
        b = np.exp(1j*x)
        self.assertArrayAlmostEqual(a, b)

        a = util.cexp(-x)
        b = np.exp(-1j*x)
        self.assertArrayAlmostEqual(a, b)

    def test_get_indices_from_identifiers(self):
        pulse = PulseSequence(
            [[util.P_np[3], [2], 'Z'],
             [util.P_np[1], [1], 'X']],
            [[util.P_np[2], [2]]],
            [1]
        )
        idx = util.get_indices_from_identifiers(pulse, ['X'], 'control')
        self.assertArrayEqual(idx, [0])

        idx = util.get_indices_from_identifiers(pulse, ['Z', 'X'], 'control')
        self.assertArrayEqual(idx, [1, 0])

        idx = util.get_indices_from_identifiers(pulse, None, 'control')
        self.assertArrayEqual(idx, [0, 1])

        idx = util.get_indices_from_identifiers(pulse, ['B_0'], 'noise')
        self.assertArrayEqual(idx, [0])

        with self.assertRaises(ValueError):
            util.get_indices_from_identifiers(pulse, ['foobar'], 'noise')

    def test_tensor(self):
        shapes = [(1, 2, 3, 4, 5), (5, 4, 3, 2, 1)]
        A = randn(*shapes[0])
        B = randn(*shapes[1])
        with self.assertRaises(ValueError):
            util.tensor(A, B)

        shapes = [(3, 2, 1), (3, 4, 2)]
        A = randn(*shapes[0])
        B = randn(*shapes[1])
        with self.assertRaises(ValueError):
            util.tensor(A, B, rank=1)

        self.assertEqual(util.tensor(A, B, rank=2).shape, (3, 8, 2))
        self.assertEqual(util.tensor(A, B, rank=3).shape, (9, 8, 2))

        shapes = [(10, 1, 3, 2), (10, 1, 2, 3)]
        A = randn(*shapes[0])
        B = randn(*shapes[1])
        self.assertEqual(util.tensor(A, B).shape, (10, 1, 6, 6))

        shapes = [(3, 5, 4, 4), (3, 5, 4, 4)]
        A = randn(*shapes[0])
        B = randn(*shapes[1])
        self.assertEqual(util.tensor(A, B).shape, (3, 5, 16, 16))

        d = randint(2, 9)
        eye = np.eye(d)
        for i in range(d):
            for j in range(d):
                A, B = eye[i:i+1, :], eye[:, j:j+1]
                self.assertArrayEqual(util.tensor(A, B, rank=1), np.kron(A, B))

        i, j = randint(0, 4, (2,))
        A, B = util.P_np[i], util.P_np[j]
        self.assertArrayEqual(util.tensor(A, B), np.kron(A, B))

        args = [randn(4, 1, 2), randn(3, 2), randn(4, 3, 5)]
        self.assertEqual(util.tensor(*args, rank=1).shape, (4, 3, 20))
        self.assertEqual(util.tensor(*args, rank=2).shape, (4, 9, 20))
        self.assertEqual(util.tensor(*args, rank=3).shape, (16, 9, 20))

        args = [randn(2, 3, 4), randn(4, 3)]
        with self.assertRaises(ValueError) as err:
            util.tensor(*args, rank=1)

        msg = ('Incompatible shapes (2, 3, 4) and (4, 3) for tensor ' +
               'product of rank 1.')
        self.assertEqual(msg, str(err.exception))

    def test_tensor_insert(self):
        I, X, Y, Z = util.P_np
        arr = util.tensor(X, I)

        with self.assertRaises(ValueError):
            # Test exception for empty args
            util.tensor_insert(arr, np.array([[]]), pos=0, arr_dims=[])

        r = util.tensor_insert(arr, Y, Z, arr_dims=[[2, 2], [2, 2]], pos=0)
        self.assertArrayAlmostEqual(r, util.tensor(Y, Z, X, I))

        r = util.tensor_insert(arr, Y, Z, arr_dims=[[2, 2], [2, 2]], pos=1)
        self.assertArrayAlmostEqual(r, util.tensor(X, Y, Z, I))

        r = util.tensor_insert(arr, Y, Z, arr_dims=[[2, 2], [2, 2]], pos=2)
        self.assertArrayAlmostEqual(r, util.tensor(X, I, Y, Z))

        # Test pos being negative
        r = util.tensor_insert(arr, Y, Z, arr_dims=[[2, 2], [2, 2]], pos=-1)
        self.assertArrayAlmostEqual(r, util.tensor(X, Y, Z, I))

        # Test pos exception
        with self.assertRaises(IndexError) as err:
            util.tensor_insert(arr, Y, Z, arr_dims=[[2, 2], [2, 2]], pos=3)

        msg = 'Invalid position 3 specified. Must be between -2 and 2.'
        self.assertEqual(msg, str(err.exception))

        # Test broadcasting and rank != 2
        A, B, C = randn(2, 3, 1, 2), randn(2, 3, 1, 2), randn(3, 1, 3)

        arr = util.tensor(A, C, rank=1)
        r = util.tensor_insert(arr, B, pos=1, rank=1,
                               arr_dims=[[2, 3]])
        self.assertArrayAlmostEqual(r, util.tensor(A, B, C, rank=1))

        # Test exceptions for wrong arr_dims format
        with self.assertRaises(ValueError):
            util.tensor_insert(arr, B, pos=1, rank=1,
                               arr_dims=[[3, 3], [1, 2], [2, 1]])

        with self.assertRaises(ValueError):
            util.tensor_insert(arr, B, pos=1, rank=1,
                               arr_dims=[[2], [2, 1]])

        A, B, C = randn(2, 3, 1, 2), randn(2, 3, 2, 2), randn(3, 2, 1)
        arr = util.tensor(A, C, rank=3)
        r = util.tensor_insert(arr, B, pos=1, rank=3,
                               arr_dims=[[3, 3], [1, 2], [2, 1]])
        self.assertArrayAlmostEqual(r, util.tensor(A, B, C, rank=3))

        # Test exceptions for wrong arr_dims format
        with self.assertRaises(ValueError):
            util.tensor_insert(arr, B, pos=1, rank=3,
                               arr_dims=[[1, 2], [2, 1]])

        with self.assertRaises(ValueError):
            util.tensor_insert(arr, B, pos=1, rank=2,
                               arr_dims=[[3, 3, 1], [1, 2], [2]])

        A, B, C = randn(2, 1), randn(1, 2, 3), randn(1)

        arr = util.tensor(A, C, rank=1)
        r = util.tensor_insert(arr, B, pos=0, rank=1, arr_dims=[[1, 1]])
        self.assertArrayAlmostEqual(r, util.tensor(A, B, C, rank=1))

        arrs, args = randn(2, 2, 2), randn(2, 2, 2)
        arr_dims = [[2, 2], [2, 2]]

        r = util.tensor_insert(util.tensor(*arrs), *args, pos=(0, 1),
                               arr_dims=arr_dims)
        self.assertArrayAlmostEqual(
            r, util.tensor(args[0], arrs[0], args[1], arrs[1])
        )

        r = util.tensor_insert(util.tensor(*arrs), *args, pos=(0, 0),
                               arr_dims=arr_dims)
        self.assertArrayAlmostEqual(r, util.tensor(*args, *arrs))

        r = util.tensor_insert(util.tensor(*arrs), *args, pos=(1, 2),
                               arr_dims=arr_dims)
        self.assertArrayAlmostEqual(
            r, util.tensor(*np.insert(arrs, (1, 2), args, axis=0))
        )

        # Test exception for wrong pos argument
        with self.assertRaises(ValueError):
            util.tensor_insert(util.tensor(*arrs), *args, pos=(0, 1, 2),
                               arr_dims=arr_dims)

        # Test exception for wrong shapes
        arrs, args = randn(2, 4, 3, 2), randn(2, 2, 3, 4)
        with self.assertRaises(ValueError) as err:
            util.tensor_insert(util.tensor(*arrs), *args, pos=(1, 2),
                               arr_dims=[[3, 3], [2, 2]])

        err_msg = ('Could not insert arg 0 with shape (4, 9, 4) into the ' +
                   'array with shape (2, 3, 4) at position 1.')
        cause_msg = ('Incompatible shapes (2, 3, 4) and (4, 9, 4) for ' +
                     'tensor product of rank 2.')

        self.assertEqual(err_msg, str(err.exception))
        self.assertEqual(cause_msg, str(err.exception.__cause__))

        # Do some random tests
        for rank, n_args, n_broadcast in zip(randint(1, 4, 10),
                                             randint(3, 6, 10),
                                             randint(1, 11, 10)):
            arrs = randn(n_args, n_broadcast, *[2]*rank)
            split_idx = randint(1, n_args-1)
            ins_idx = randint(split_idx-n_args, n_args-split_idx)
            ins_arrs = arrs[:split_idx]
            arr = util.tensor(*arrs[split_idx:], rank=rank)
            sorted_arrs = np.insert(arrs[split_idx:], ins_idx, ins_arrs,
                                    axis=0)

            arr_dims = [[2]*(n_args-split_idx)]*rank
            r = util.tensor_insert(arr, *ins_arrs, pos=ins_idx, rank=rank,
                                   arr_dims=arr_dims)
            self.assertArrayAlmostEqual(
                r, util.tensor(*sorted_arrs, rank=rank))

            pos = randint(-split_idx+1, split_idx, split_idx)
            r = util.tensor_insert(arr, *ins_arrs, pos=pos, rank=rank,
                                   arr_dims=arr_dims)
            sorted_arrs = np.insert(arrs[split_idx:], pos, ins_arrs, axis=0)
            self.assertArrayAlmostEqual(
                r, util.tensor(*sorted_arrs, rank=rank),
                atol=1e-10
            )

    def test_tensor_merge(self):
        # Test basic functionality
        I, X, Y, Z = util.P_np
        arr = util.tensor(X, Y, Z)
        ins = util.tensor(I, I)
        r1 = util.tensor_merge(arr, ins, pos=[1, 2], arr_dims=[[2]*3, [2]*3],
                               ins_dims=[[2]*2, [2]*2])
        r2 = util.tensor_merge(ins, arr, pos=[0, 1, 2],
                               arr_dims=[[2]*2, [2]*2],
                               ins_dims=[[2]*3, [2]*3])

        self.assertArrayAlmostEqual(r1, util.tensor(X, I, Y, I, Z))
        self.assertArrayAlmostEqual(r1, r2)

        # Test if tensor_merge and tensor_insert produce same results
        arr = util.tensor(Y, Z)
        ins = util.tensor(I, X)
        r1 = util.tensor_merge(arr, ins, pos=[0, 0], arr_dims=[[2]*2, [2]*2],
                               ins_dims=[[2]*2, [2]*2])
        r2 = util.tensor_insert(arr, I, X, pos=[0, 0], arr_dims=[[2]*2, [2]*2])
        self.assertArrayAlmostEqual(r1, r2)

        # Test pos being negative
        r = util.tensor_merge(arr, ins, arr_dims=[[2, 2], [2, 2]],
                              ins_dims=[[2, 2], [2, 2]], pos=(-1, -2))
        self.assertArrayAlmostEqual(r, util.tensor(X, Y, I, Z))

        # Test exceptions
        # Wrong dims format
        with self.assertRaises(ValueError):
            util.tensor_merge(arr, ins, pos=(1, 2),
                              arr_dims=[[2, 2], [2, 2], [2, 2]],
                              ins_dims=[[2, 2], [2, 2]])

        with self.assertRaises(ValueError):
            util.tensor_merge(arr, ins, pos=(1, 2),
                              arr_dims=[[2, 2], [2, 2]],
                              ins_dims=[[2, 2], [2, 2], [2, 2]])

        with self.assertRaises(ValueError):
            util.tensor_merge(arr, ins, pos=(1, 2),
                              arr_dims=[[2, 2], [2, 2, 2]],
                              ins_dims=[[2, 2], [2, 2]])

        with self.assertRaises(ValueError):
            util.tensor_merge(arr, ins, pos=(1, 2),
                              arr_dims=[[2, 2], [2, 2]],
                              ins_dims=[[2, 2], [2, 2, 2]])

        # Wrong pos
        with self.assertRaises(IndexError):
            util.tensor_merge(arr, ins, pos=(1, 3),
                              arr_dims=[[2, 2], [2, 2]],
                              ins_dims=[[2, 2], [2, 2]])

        # Wrong dimensions given
        with self.assertRaises(ValueError):
            util.tensor_merge(arr, ins, pos=(1, 2),
                              arr_dims=[[2, 3], [2, 2]],
                              ins_dims=[[2, 2], [2, 2]])

        with self.assertRaises(ValueError):
            util.tensor_merge(arr, ins, pos=(1, 2),
                              arr_dims=[[2, 2], [2, 2]],
                              ins_dims=[[2, 3], [2, 2]])

        # Incompatible shapes
        arrs, args = randn(2, 4, 3, 2), randn(2, 2, 3, 4)
        with self.assertRaises(ValueError) as err:
            util.tensor_merge(util.tensor(*arrs), util.tensor(*args),
                              pos=(1, 2), arr_dims=[[3, 3], [2, 2]],
                              ins_dims=[[3, 3], [4, 4]])

        msg = ('Incompatible shapes (2, 9, 16) and (4, 9, 4) for tensor ' +
               'product of rank 2.')

        self.assertEqual(msg, str(err.exception))

        # Test rank 1 and broadcasting
        arr = np.random.randn(2, 10, 3, 4)
        ins = np.random.randn(2, 10, 3, 2)
        r = util.tensor_merge(util.tensor(*arr, rank=1),
                              util.tensor(*ins, rank=1), pos=[0, 1],
                              arr_dims=[[4, 4]], ins_dims=[[2, 2]], rank=1)
        self.assertArrayAlmostEqual(
            r, util.tensor(ins[0], arr[0], ins[1], arr[1], rank=1)
        )

        # Do some random tests
        for rank, n_args, n_broadcast in zip(randint(1, 4, 10),
                                             randint(3, 6, 10),
                                             randint(1, 11, 10)):
            arrs = randn(n_args, n_broadcast, *[2]*rank)
            split_idx = randint(1, n_args-1)
            arr = util.tensor(*arrs[split_idx:], rank=rank)
            ins = util.tensor(*arrs[:split_idx], rank=rank)
            pos = randint(0, split_idx, split_idx)
            sorted_arrs = np.insert(arrs[split_idx:], pos, arrs[:split_idx],
                                    axis=0)

            arr_dims = [[2]*(n_args-split_idx)]*rank
            ins_dims = [[2]*split_idx]*rank
            r = util.tensor_merge(arr, ins, pos=pos, rank=rank,
                                  arr_dims=arr_dims, ins_dims=ins_dims)
            self.assertArrayAlmostEqual(
                r, util.tensor(*sorted_arrs, rank=rank))

    def test_tensor_transpose(self):
        # Test basic functionality
        paulis = np.array(util.P_np)
        I, X, Y, Z = paulis
        arr = util.tensor(I, X, Y, Z)
        arr_dims = [[2]*4]*2
        order = np.arange(4)

        for _ in range(20):
            order = np.random.permutation(order)
            r = util.tensor_transpose(arr, order, arr_dims)
            self.assertArrayAlmostEqual(r, util.tensor(*paulis[order]))

        # Check exceptions
        with self.assertRaises(ValueError):
            # wrong arr_dims (too few dims)
            r = util.tensor_transpose(arr, order, [[2]*3]*2)

        with self.assertRaises(ValueError):
            # wrong arr_dims (too few dims)
            r = util.tensor_transpose(arr, order, [[2]*4]*1)

        with self.assertRaises(ValueError):
            # wrong arr_dims (dims too large)
            r = util.tensor_transpose(arr, order, [[3]*4]*2)

        with self.assertRaises(ValueError):
            # wrong order (too few axes)
            r = util.tensor_transpose(arr, (0, 1, 2), arr_dims)

        with self.assertRaises(ValueError):
            # wrong order (index 4 too large)
            r = util.tensor_transpose(arr, (1, 2, 3, 4), arr_dims)

        with self.assertRaises(ValueError):
            # wrong order (not unique axes)
            r = util.tensor_transpose(arr, (1, 1, 1, 1), arr_dims)

        with self.assertRaises(TypeError):
            # wrong order (floats instead of ints)
            r = util.tensor_transpose(arr, (0., 1., 2., 3.), arr_dims)

        # Random tests
        for rank, n_args, n_broadcast in zip(randint(1, 4, 10),
                                             randint(3, 6, 10),
                                             randint(1, 11, 10)):
            arrs = randn(n_args, n_broadcast, *[2]*rank)
            order = np.random.permutation(n_args)
            arr_dims = [[2]*n_args]*rank

            r = util.tensor_transpose(util.tensor(*arrs, rank=rank),
                                      order=order, arr_dims=arr_dims,
                                      rank=rank)
            self.assertArrayAlmostEqual(
                r, util.tensor(*arrs[order], rank=rank))

    def test_mdot(self):
        arr = randn(3, 2, 4, 4)
        self.assertArrayEqual(util.mdot(arr, 0), arr[0] @ arr[1] @ arr[2])
        self.assertArrayEqual(util.mdot(arr, 1), arr[:, 0] @ arr[:, 1])

    def test_remove_float_errors(self):
        for eps_scale in (None, 2):
            scale = 1 if eps_scale is None else eps_scale
            for dtype in (float, complex):
                arr = np.zeros((10, 10), dtype=dtype)
                arr += scale*np.finfo(arr.dtype).eps*random(arr.shape)
                arr[randint(0, 2, arr.shape, dtype=bool)] *= -1
                arr = util.remove_float_errors(arr, eps_scale)
                self.assertArrayEqual(arr, np.zeros(arr.shape, dtype=dtype))

    def test_oper_equiv(self):
        with self.assertRaises(ValueError):
            util.oper_equiv(*[np.ones((1, 2, 3))]*2)

        for d in randint(2, 10, (5,)):
            psi = qt.rand_ket(d)
            U = qt.rand_dm(d)
            phase = randn()

            result = util.oper_equiv(psi, psi*np.exp(1j*phase))
            self.assertTrue(result[0])
            self.assertAlmostEqual(result[1], phase, places=5)

            result = util.oper_equiv(psi*np.exp(1j*phase), psi)
            self.assertTrue(result[0])
            self.assertAlmostEqual(result[1], -phase, places=5)

            psi = psi.full()
            psi /= np.sqrt(np.linalg.norm(psi, ord=2))

            result = util.oper_equiv(psi, psi*np.exp(1j*phase),
                                     normalized=True, eps=1e-13)
            self.assertTrue(result[0])
            self.assertAlmostEqual(result[1], phase, places=5)

            result = util.oper_equiv(psi, psi+1)
            self.assertFalse(result[0])

            result = util.oper_equiv(U, U*np.exp(1j*phase))
            self.assertTrue(result[0])
            self.assertAlmostEqual(result[1], phase, places=5)

            result = util.oper_equiv(U*np.exp(1j*phase), U)
            self.assertTrue(result[0])
            self.assertAlmostEqual(result[1], -phase, places=5)

            U = U.full()
            U /= np.sqrt(util.dot_HS(U, U))
            result = util.oper_equiv(U, U*np.exp(1j*phase), normalized=True,
                                     eps=1e-10)
            self.assertTrue(result[0])
            self.assertAlmostEqual(result[1], phase)

            result = util.oper_equiv(U, U+1)
            self.assertFalse(result[0])

    def test_dot_HS(self):
        U, V = randint(0, 100, (2, 2, 2))
        S = util.dot_HS(U, V)
        T = util.dot_HS(U, V, eps=0)
        self.assertArrayEqual(S, T)

        for d in randint(2, 10, (5,)):
            U = qt.rand_herm(d)
            V = qt.rand_herm(d)
            self.assertArrayAlmostEqual(util.dot_HS(U, V), (U.dag()*V).tr())

            U = qt.rand_unitary(d)
            self.assertEqual(util.dot_HS(U, U), d)

            self.assertEqual(util.dot_HS(U, U + 1e-14, eps=1e-10), d)

    def test_all_array_equal(self):
        for n in randint(2, 10, (10,)):
            gen = (np.ones((10, 10)) for _ in range(n))
            lst = [np.ones((10, 10)) for _ in range(n)]
            self.assertTrue(util.all_array_equal(gen))
            self.assertTrue(util.all_array_equal(lst))

            gen = (np.arange(9).reshape(3, 3) + i for i in range(n))
            lst = [np.arange(9).reshape(3, 3) + i for i in range(n)]
            self.assertFalse(util.all_array_equal(gen))
            self.assertFalse(util.all_array_equal(lst))

    def test_get_sample_frequencies(self):
        pulse = PulseSequence(
            [[util.P_np[1], [np.pi/2]]],
            [[util.P_np[1], [1]]],
            [abs(np.random.randn())]
        )
        # Default args
        omega = util.get_sample_frequencies(pulse)
        self.assertAlmostEqual(omega[0], -2e2*np.pi/pulse.t[-1])
        self.assertAlmostEqual(omega[-1], 2e2*np.pi/pulse.t[-1])
        self.assertEqual(len(omega), 200)
        self.assertTrue((omega[:100] <= 0).all())
        self.assertLessEqual(np.var(np.diff(np.log(omega[100:]))), 1e-16)

        # custom args
        omega = util.get_sample_frequencies(pulse, spacing='linear',
                                            n_samples=50, symmetric=False)
        self.assertAlmostEqual(omega[0], 0)
        self.assertAlmostEqual(omega[-1], 2e2*np.pi/pulse.t[-1])
        self.assertEqual(len(omega), 50)
        self.assertTrue((omega >= 0).all())
        self.assertLessEqual(np.var(np.diff(omega)), 1e-16)

        # Exceptions
        with self.assertRaises(ValueError):
            omega = util.get_sample_frequencies(pulse, spacing='foo')

    def test_symmetrize_spectrum(self):
        pulse = PulseSequence(
            [[util.P_np[1], [np.pi/2]]],
            [[util.P_np[1], [1]]],
            [abs(np.random.randn())]
        )

        asym_omega = util.get_sample_frequencies(pulse, symmetric=False,
                                                 n_samples=100)
        sym_omega = util.get_sample_frequencies(pulse, symmetric=True,
                                                n_samples=200)

        S_symmetrized, omega_symmetrized = util.symmetrize_spectrum(
            1/asym_omega**0.7, asym_omega)
        self.assertArrayEqual(omega_symmetrized, sym_omega)
        self.assertArrayEqual(S_symmetrized[99::-1], S_symmetrized[100:])
        self.assertArrayEqual(S_symmetrized[100:]*2, 1/asym_omega**0.7)

    def test_simple_progressbar(self):
        with self.assertRaises(TypeError):
            for i in util._simple_progressbar((i for i in range(10))):
                pass

        for i in util._simple_progressbar(range(10), prefix = "foo", size=10,
                                          count=5):
            pass

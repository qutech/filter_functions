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
This module tests the operator basis module.
"""

from itertools import product

import numpy as np
from sparse import COO
from tests import testutil

import filter_functions as ff


class BasisTest(testutil.TestCase):

    def test_basis_constructor(self):
        """Test the constructor for several failure modes"""

        # Constructing from given elements should check for __getitem__
        with self.assertRaises(TypeError):
            _ = ff.Basis(1)

        # All elements should be either COO, Qobj, or ndarray
        elems = [ff.util.P_np[1], ff.util.P_qt[2],
                 COO.from_numpy(ff.util.P_np[3]), ff.util.P_qt[0].data]
        with self.assertRaises(TypeError):
            _ = ff.Basis(elems)

        # Excluding the fast_csr element should work
        self.assertEqual(ff.Basis.pauli(1), ff.Basis(elems[:-1]))

        # Too many elements
        with self.assertRaises(ValueError):
            _ = ff.Basis(np.random.randn(5, 2, 2))

        # Properly normalized
        self.assertEqual(ff.Basis.pauli(1), ff.Basis(ff.util.P_np))

        # Non traceless elems but traceless basis requested
        with self.assertRaises(ValueError):
            _ = ff.Basis(np.ones((2, 2)), traceless=True)

        # Calling with only the identity should work with traceless true or
        # false
        self.assertEqual(ff.Basis(np.eye(2), traceless=False),
                         ff.Basis(np.eye(2), traceless=True))

        # Constructing a basis from a basis should work
        _ = ff.Basis(ff.Basis.ggm(2)[1:])

    def test_basis_properties(self):
        """Basis orthonormal and of correct dimensions"""
        d = np.random.randint(2, 17)
        n = np.random.randint(1, 5)

        ggm_basis = ff.Basis.ggm(d)
        pauli_basis = ff.Basis.pauli(n)
        custom_basis = ff.Basis(testutil.rand_herm(d), traceless=False)

        btypes = ('Pauli', 'GGM', 'Custom')
        bases = (pauli_basis, ggm_basis, custom_basis)
        for btype, base in zip(btypes, bases):
            base.tidyup(eps_scale=0)
            self.assertTrue(base == base)
            self.assertFalse(base == ff.Basis.ggm(d+1))
            self.assertEqual(btype, base.btype)
            if not btype == 'Pauli':
                self.assertEqual(d, base.d)
                # Check if __contains__ works as expected
                self.assertTrue(base[np.random.randint(0, d**2)] in base)
            else:
                self.assertEqual(2**n, base.d)
                # Check if __contains__ works as expected
                self.assertTrue(base[np.random.randint(0, (2**n)**2)] in base)
            # Check if all elements of each basis are orthonormal and hermitian
            self.assertArrayEqual(base.T,
                                  base.view(np.ndarray).swapaxes(-1, -2))
            self.assertTrue(base.isorthonorm)
            self.assertTrue(base.isherm)
            # Check if basis spans the whole space and all elems are traceless
            if not btype == 'Custom':
                self.assertTrue(base.istraceless)
            else:
                self.assertFalse(base.istraceless)

            self.assertTrue(base.iscomplete)
            # Check sparse representation
            self.assertArrayEqual(base.sparse.todense(), base)
            # Test sparse cache
            self.assertArrayEqual(base.sparse.todense(), base)

            if base.d < 8:
                # Test very resource intense
                self.assertArrayAlmostEqual(base.four_element_traces.todense(),
                                            np.einsum('iab,jbc,kcd,lda',
                                                      *(base,)*4),
                                            atol=1e-16)

            base._print_checks()

        basis = ff.util.P_np[1].view(ff.Basis)
        self.assertTrue(basis.isorthonorm)
        self.assertArrayEqual(basis.T, basis.view(np.ndarray).T)

    def test_basis_expansion_and_normalization(self):
        """Correct expansion of operators and normalization of bases"""
        for _ in range(10):
            d = np.random.randint(2, 16)
            ggm_basis = ff.Basis.ggm(d)
            basis = ff.Basis(
                np.einsum('i,ijk->ijk', np.random.randn(d**2), ggm_basis),
                skip_check=True
            )
            M = np.random.randn(d, d) + 1j*np.random.randn(d, d)
            M -= np.trace(M)/d
            coeffs = ff.basis.expand(M, basis, normalized=False)
            self.assertArrayAlmostEqual(M, np.einsum('i,ijk', coeffs, basis))
            self.assertArrayAlmostEqual(ff.basis.expand(M, ggm_basis),
                                        ff.basis.ggm_expand(M),
                                        atol=1e-14)
            self.assertArrayAlmostEqual(ff.basis.ggm_expand(M),
                                        ff.basis.ggm_expand(M, traceless=True),
                                        atol=1e-14)

            n = np.random.randint(1, 50)
            M = np.random.randn(n, d, d) + 1j*np.random.randn(n, d, d)
            coeffs = ff.basis.expand(M, basis, normalized=False)
            self.assertArrayAlmostEqual(M, np.einsum('li,ijk->ljk', coeffs,
                                                     basis))
            self.assertArrayAlmostEqual(ff.basis.expand(M, ggm_basis),
                                        ff.basis.ggm_expand(M), atol=1e-14)

            # Argument to ggm_expand not square in last two dimensions
            with self.assertRaises(ValueError):
                ff.basis.ggm_expand(basis[..., 0])

            self.assertTrue(ff.basis.normalize(basis).isorthonorm)

            # Basis method and function should give the same
            normalized = ff.basis.normalize(basis)
            basis.normalize()
            self.assertEqual(normalized, basis)

            # normalize single element
            elem = basis[1]
            normalized = ff.basis.normalize(elem)
            elem.normalize()
            self.assertEqual(normalized, elem)

            # Not matrix or sequence of matrices
            with self.assertRaises(ValueError):
                ff.basis.normalize(basis[0, 0])

    def test_basis_generation_from_partial_ggm(self):
        """"Generate complete basis from partial elements of a GGM basis"""
        # Do 100 test runs with random elements from a GGM basis in (2 ... 8)
        # dimensions
        for _ in range(50):
            d = np.random.randint(2, 9)
            b = ff.Basis.ggm(d)
            inds = [i for i in range(d**2)]
            tup = tuple(inds.pop(np.random.randint(0, len(inds)))
                        for _ in range(np.random.randint(1, d**2)))
            elems = b[tup, ...]
            basis = ff.Basis(elems)
            self.assertTrue(basis.isorthonorm)
            self.assertTrue(basis.isherm)
            self.assertTrue(basis.istraceless)
            self.assertTrue(basis.iscomplete)
            self.assertTrue(all(elem in basis for elem in elems))

    def test_basis_generation_from_partial_pauli(self):
        """"Generate complete basis from partial elements of a Pauli basis"""
        # Do 100 test runs with random elements from a Pauli basis in (2 ... 8)
        # dimensions
        for _ in range(50):
            n = np.random.randint(1, 4)
            d = 2**n
            b = ff.Basis.pauli(n)
            inds = [i for i in range(d**2)]
            tup = tuple(inds.pop(np.random.randint(0, len(inds)))
                        for _ in range(np.random.randint(1, d**2)))
            elems = b[tup, ...]
            basis = ff.Basis(elems)
            self.assertTrue(basis.isorthonorm)
            self.assertTrue(basis.isherm)
            self.assertTrue(basis.istraceless)
            self.assertTrue(basis.iscomplete)
            self.assertTrue(all(elem in basis for elem in elems))

            with self.assertWarns(UserWarning):
                b = [basis[0], 1j*basis[1]]
                ff.Basis(b)

            with self.assertRaises(ValueError):
                b = [basis[0], basis[0] + basis[1]]
                ff.Basis(b)

    def test_basis_generation_from_partial_random(self):
        """"Generate complete basis from partial elements of a random basis"""
        # Do 25 test runs with random elements from a random basis in
        # (2 ... 8) dimensions
        for _ in range(25):
            d = np.random.randint(2, 7)
            # Get a random traceless hermitian operator
            oper = testutil.rand_herm_traceless(d)
            # ... and build a basis from it
            b = ff.Basis(np.array([oper]))
            self.assertTrue(b.isorthonorm)
            self.assertTrue(b.isherm)
            self.assertTrue(b.istraceless)
            self.assertTrue(b.iscomplete)
            # Choose random elements from that basis and generate a new basis
            # from it
            inds = [i for i in range(d**2)]
            tup = tuple(inds.pop(np.random.randint(0, len(inds)))
                        for _ in range(np.random.randint(1, d**2)))
            elems = b[tup, ...]
            basis = ff.Basis(elems)
            self.assertTrue(basis.isorthonorm)
            self.assertTrue(basis.isherm)
            self.assertTrue(basis.istraceless)
            self.assertTrue(basis.iscomplete)
            self.assertTrue(all(elem in basis for elem in elems))

        # Test runs with non-traceless opers
        for _ in range(25):
            d = np.random.randint(2, 7)
            # Get a random hermitian operator
            oper = testutil.rand_herm(d)
            # ... and build a basis from it
            b = ff.Basis(np.array([oper]))
            self.assertTrue(b.isorthonorm)
            self.assertTrue(b.isherm)
            self.assertFalse(b.istraceless)
            self.assertTrue(b.iscomplete)
            # Choose random elements from that basis and generate a new basis
            # from it
            inds = [i for i in range(d**2)]
            tup = tuple(inds.pop(np.random.randint(0, len(inds)))
                        for _ in range(np.random.randint(1, d**2)))
            elems = b[tup, ...]
            basis = ff.Basis(elems)
            self.assertTrue(basis.isorthonorm)
            self.assertTrue(basis.isherm)
            self.assertFalse(basis.istraceless)
            self.assertTrue(basis.iscomplete)
            self.assertTrue(all(elem in basis for elem in elems))

    def test_filter_functions(self):
        """Filter functions equal for different bases"""
        # Set up random Hamiltonian
        c_oper = testutil.rand_herm(4)
        c_opers = (c_oper,)
        c_coeffs = ([0, 1, 0],)
        H_c = list(zip(c_opers, c_coeffs))

        dt = np.ones(3)

        n_oper = testutil.rand_herm_traceless(4)
        n_oper[np.diag_indices(n_oper.shape[-1])] = 0
        n_oper = ff.basis.normalize(n_oper)

        H_n = [[n_oper, np.ones_like(dt)]]

        omega = np.concatenate([-np.logspace(3, -3, 100),
                                np.logspace(-3, 3, 100)])

        pauli_basis = ff.Basis.pauli(2)
        ggm_basis = ff.Basis.ggm(4)
        from_random_basis = ff.Basis([n_oper])
        bases = (pauli_basis, ggm_basis, from_random_basis)

        # Get Pulses
        pulses = [ff.PulseSequence(H_c, H_n, dt, basis=b) for b in bases]
        F = [pulse.get_filter_function(omega).sum(0) for pulse in pulses]
        for pair in product(F, F):
            self.assertArrayAlmostEqual(*pair)

    def test_control_matrix(self):
        """Test control matrix for traceless and non-traceless bases"""
        c_opers = testutil.rand_herm(3, 4)
        c_coeffs = np.random.randn(4, 10)

        n_opers_traceless = testutil.rand_herm_traceless(3, 4)
        n_opers = testutil.rand_herm(3, 4)
        n_coeffs = np.abs(np.random.randn(4, 10))

        dt = np.abs(np.random.randn(10))

        basis = ff.Basis(testutil.rand_herm(3), traceless=False)
        basis_traceless = ff.Basis(testutil.rand_herm_traceless(3),
                                   traceless=True)

        omega = np.logspace(-1, 1, 51)

        for i, base in enumerate((basis, basis_traceless)):
            for j, n_ops in enumerate((n_opers, n_opers_traceless)):
                pulse = ff.PulseSequence(list(zip(c_opers, c_coeffs)),
                                         list(zip(n_ops, n_coeffs)),
                                         dt, base)

                R = pulse.get_control_matrix(omega)

                if i == 0 and j == 0:
                    # base not traceless, nopers not traceless
                    self.assertTrue((R[:, 0] != 0).all())
                elif i == 0 and j == 1:
                    # base not traceless, nopers traceless
                    self.assertTrue((R[:, 0] != 0).all())
                elif i == 1 and j == 0:
                    # base traceless, nopers not traceless
                    self.assertTrue((R[:, 0] != 0).all())
                elif i == 1 and j == 1:
                    # base traceless, nopers traceless
                    self.assertTrue(np.allclose(R[:, 0], 0))

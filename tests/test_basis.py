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
from copy import copy
from itertools import product

import numpy as np
import pytest
from opt_einsum import contract
from sparse import COO

import filter_functions as ff
from filter_functions import util
from tests import testutil
from tests.testutil import rng

from . import qutip


class BasisTest(testutil.TestCase):

    def test_basis_constructor(self):
        """Test the constructor for several failure modes"""
        # Constructing from given elements should check for __getitem__
        with self.assertRaises(TypeError):
            _ = ff.Basis(1)

        # All elements should be either sparse, Qobj, or ndarray
        elems = [util.paulis[1], COO.from_numpy(util.paulis[3]), [[0, 1], [1, 0]]]
        with self.assertRaises(TypeError):
            _ = ff.Basis(elems)

        # Too many elements
        with self.assertRaises(ValueError):
            _ = ff.Basis(rng.standard_normal((5, 2, 2)))

        # Non traceless elems but traceless basis requested
        with self.assertRaises(ValueError):
            _ = ff.Basis.from_partial(np.ones((2, 2)), traceless=True)

        # Incorrect number of labels for default constructor
        with self.assertRaises(ValueError):
            _ = ff.Basis(util.paulis, labels=['a', 'b', 'c'])

        # Incorrect number of labels for from_partial constructor
        with self.assertRaises(ValueError):
            _ = ff.Basis.from_partial(util.paulis[:2], labels=['a', 'b', 'c'])

        # from_partial constructor should move identity label to the front
        basis = ff.Basis.from_partial([util.paulis[1], util.paulis[0]], labels=['x', 'i'])
        self.assertEqual(basis.labels[:2], ['i', 'x'])
        self.assertEqual(basis.labels[2:], ['$C_{2}$', '$C_{3}$'])

        # from_partial constructor should copy labels if it can
        partial_basis = ff.Basis.pauli(1)[[1, 3]]
        partial_basis.labels = [partial_basis.labels[1], partial_basis.labels[3]]
        basis = ff.Basis.from_partial(partial_basis)
        self.assertEqual(basis.labels[:2], partial_basis.labels)
        self.assertEqual(basis.labels[2:], ['$C_{2}$', '$C_{3}$'])

        # Default constructor should return 3d array also for single 2d element
        basis = ff.Basis(rng.standard_normal((2, 2)))
        self.assertEqual(basis.shape, (1, 2, 2))

        # from_partial constructor should return same basis for 2d or 3d input
        elems = testutil.rand_herm(3)
        basis1 = ff.Basis.from_partial(elems, labels=['weif'])
        basis2 = ff.Basis.from_partial(elems.squeeze(), labels=['weif'])
        self.assertEqual(basis1, basis2)

        # Calling with only the identity should work with traceless true or false
        self.assertEqual(ff.Basis(np.eye(2), traceless=False), ff.Basis(np.eye(2), traceless=True))

        # Constructing a basis from a basis should work
        _ = ff.Basis(ff.Basis.ggm(2)[1:])

        # Constructing should not change the elements
        elems = rng.standard_normal((6, 3, 3))
        basis = ff.Basis(elems)
        self.assertArrayEqual(elems, basis)

    def test_basis_properties(self):
        """Basis orthonormal and of correct dimensions"""
        d = rng.integers(2, 17)
        n = rng.integers(1, 5)

        ggm_basis = ff.Basis.ggm(d)
        pauli_basis = ff.Basis.pauli(n)
        from_partial_basis = ff.Basis.from_partial(testutil.rand_herm(d), traceless=False)
        custom_basis = ff.Basis(testutil.rand_herm_traceless(d))
        custom_basis /= np.linalg.norm(custom_basis, ord='fro', axis=(-1, -2))

        btypes = ('Pauli', 'GGM', 'From partial', 'Custom')
        bases = (pauli_basis, ggm_basis, from_partial_basis, custom_basis)
        for btype, base in zip(btypes, bases):
            base.tidyup(eps_scale=0)
            self.assertTrue(base == base)
            self.assertFalse(base == ff.Basis.ggm(d+1))
            self.assertEqual(btype, base.btype)
            if not btype == 'Pauli':
                self.assertEqual(d, base.d)
                # Check if __contains__ works as expected
                self.assertTrue(base[rng.integers(0, len(base))] in base)
            else:
                self.assertEqual(2**n, base.d)
                # Check if __contains__ works as expected
                self.assertTrue(base[rng.integers(0, len(base))] in base)
            # Check if all elements of each basis are orthonormal and hermitian
            self.assertArrayEqual(base.T, base.view(np.ndarray).swapaxes(-1, -2))
            self.assertTrue(base.isnorm)
            self.assertTrue(base.isorthogonal)
            self.assertTrue(base.isorthonorm)
            self.assertTrue(base.isherm)
            # Check if basis spans the whole space and all elems are traceless
            if not btype == 'From partial':
                self.assertTrue(base.istraceless)
            else:
                self.assertFalse(base.istraceless)

            if not btype == 'Custom':
                self.assertTrue(base.iscomplete)
            else:
                self.assertFalse(base.iscomplete)

            # Check sparse representation
            self.assertArrayEqual(base.sparse.todense(), base)
            # Test sparse cache
            self.assertArrayEqual(base.sparse.todense(), base)

            if base.d < 8:
                # Test very resource intense
                ref = contract('iab,jbc,kcd,lda', *(base,)*4)
                self.assertArrayAlmostEqual(base.four_element_traces.todense(), ref, atol=1e-16)

                # Test setter
                base._four_element_traces = None
                base.four_element_traces = ref
                self.assertArrayEqual(base.four_element_traces, ref)

            base._print_checks()

        # single element always considered orthogonal
        orthogonal = rng.normal(size=(d, d))
        self.assertTrue(orthogonal.view(ff.Basis).isorthogonal)

        orthogonal /= np.linalg.norm(orthogonal, 'fro', axis=(-1, -2))
        self.assertTrue(orthogonal.view(ff.Basis).isorthonorm)

        nonorthogonal = rng.normal(size=(3, d, d)).view(ff.Basis)
        self.assertFalse(nonorthogonal.isnorm)
        nonorthogonal.normalize()
        self.assertTrue(nonorthogonal.isnorm)

        herm = testutil.rand_herm(d).squeeze()
        self.assertTrue(herm.view(ff.Basis).isherm)

        herm[0, 1] += 1
        self.assertFalse(herm.view(ff.Basis).isherm)

        traceless = testutil.rand_herm_traceless(d).squeeze()
        self.assertTrue(traceless.view(ff.Basis).istraceless)

        traceless[0, 0] += 1
        self.assertFalse(traceless.view(ff.Basis).istraceless)

    def test_transpose(self):
        arr = rng.normal(size=(2, 3, 3))
        b = arr.view(ff.Basis)
        self.assertArrayEqual(b.T, arr.transpose(0, 2, 1))

        arr = rng.normal(size=(2, 2))
        b = arr.view(ff.Basis)
        self.assertArrayEqual(b.T, arr.transpose(1, 0))

        # Edge cases for not officially intended usage
        arr = rng.normal(size=2)
        b = arr.view(ff.Basis)
        self.assertArrayEqual(b.T, arr)

        arr = rng.normal(size=(2, 3, 4, 4))
        b = arr.view(ff.Basis)
        self.assertArrayEqual(b.T, arr.transpose(0, 1, 3, 2))


    def test_basis_expansion_and_normalization(self):
        """Correct expansion of operators and normalization of bases"""
        # dtype
        b = ff.Basis.ggm(3)
        r = ff.basis.expand(rng.standard_normal((3, 3)), b, hermitian=False)
        self.assertTrue(r.dtype == 'complex128')
        r = ff.basis.expand(testutil.rand_herm(3), b, hermitian=True)
        self.assertTrue(r.dtype == 'float64')
        b.isherm = False
        r = ff.basis.expand(testutil.rand_herm(3), b, hermitian=True)
        self.assertTrue(r.dtype == 'complex128')

        r = ff.basis.ggm_expand(testutil.rand_herm(3), hermitian=True)
        self.assertTrue(r.dtype == 'float64')
        r = ff.basis.ggm_expand(rng.standard_normal((3, 3)), hermitian=False)
        self.assertTrue(r.dtype == 'complex128')

        # test the method
        pauli = ff.Basis.pauli(1)
        ggm = ff.Basis.ggm(2)

        M = testutil.rand_herm(2, 3)
        self.assertArrayAlmostEqual(pauli.expand(M, hermitian=True, traceless=False, tidyup=True),
                                    ggm.expand(M, hermitian=True, traceless=False, tidyup=True))

        M = testutil.rand_herm_traceless(2, 3)
        self.assertArrayAlmostEqual(pauli.expand(M, hermitian=True, traceless=True, tidyup=True),
                                    ggm.expand(M, hermitian=True, traceless=True, tidyup=True))

        M = testutil.rand_unit(2, 3)
        self.assertArrayAlmostEqual(pauli.expand(M, hermitian=False, traceless=False, tidyup=True),
                                    ggm.expand(M, hermitian=False, traceless=False, tidyup=True))

        for _ in range(10):
            d = rng.integers(2, 16)
            ggm_basis = ff.Basis.ggm(d)
            basis = ff.Basis(np.einsum('i,ijk->ijk', rng.standard_normal(d**2), ggm_basis))
            M = rng.standard_normal((d, d)) + 1j*rng.standard_normal((d, d))
            M -= np.trace(M)/d
            coeffs = ff.basis.expand(M, basis, normalized=False)
            self.assertArrayAlmostEqual(M, np.einsum('i,ijk', coeffs, basis))
            self.assertArrayAlmostEqual(ff.basis.expand(M, ggm_basis),
                                        ff.basis.ggm_expand(M), atol=1e-14)
            self.assertArrayAlmostEqual(ff.basis.ggm_expand(M),
                                        ff.basis.ggm_expand(M, traceless=True), atol=1e-14)

            n = rng.integers(1, 50)
            M = rng.standard_normal((n, d, d)) + 1j*rng.standard_normal((n, d, d))
            coeffs = ff.basis.expand(M, basis, normalized=False)
            self.assertArrayAlmostEqual(M, np.einsum('li,ijk->ljk', coeffs, basis))
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

            # Test copy
            arr = rng.standard_normal((3, 2, 2))
            basis = ff.Basis(arr)
            normalized = basis.normalize(copy=True)
            self.assertIsNot(normalized, basis)
            self.assertFalse(np.array_equal(normalized, basis))
            basis.normalize()
            self.assertArrayEqual(normalized, basis)

    def test_basis_generation_from_partial_ggm(self):
        """"Generate complete basis from partial elements of a GGM basis"""
        # Do test runs with random elements from a GGM basis in (2 ... 8)
        # dimensions
        for _ in range(50):
            d = rng.integers(2, 9)
            b = ff.Basis.ggm(d)
            inds = [i for i in range(d**2)]
            tup = tuple(inds.pop(rng.integers(0, len(inds))) for _ in range(rng.integers(1, d**2)))
            elems = b[tup, ...]
            basis = ff.Basis.from_partial(elems)
            self.assertTrue(basis.isorthonorm)
            self.assertTrue(basis.isherm)
            self.assertTrue(basis.istraceless)
            self.assertTrue(basis.iscomplete)
            self.assertTrue(all(elem in basis for elem in elems))

    def test_basis_generation_from_partial_pauli(self):
        """"Generate complete basis from partial elements of a Pauli basis"""
        # Do test runs with random elements from a Pauli basis in (2 ... 8)
        # dimensions
        for _ in range(50):
            n = rng.integers(1, 4)
            d = 2**n
            b = ff.Basis.pauli(n)
            inds = [i for i in range(d**2)]
            tup = tuple(inds.pop(rng.integers(0, len(inds))) for _ in range(rng.integers(1, d**2)))
            elems = b[tup, ...]
            basis = ff.Basis.from_partial(elems)
            self.assertTrue(basis.isorthonorm)
            self.assertTrue(basis.isherm)
            self.assertTrue(basis.istraceless)
            self.assertTrue(basis.iscomplete)
            self.assertTrue(all(elem in basis for elem in elems))

            with self.assertWarns(UserWarning):
                ff.Basis.from_partial([basis[0], 1j*basis[1]])

            with self.assertRaises(ValueError):
                ff.Basis.from_partial([basis[0], basis[0] + basis[1]])

    def test_basis_generation_from_partial_random(self):
        """"Generate complete basis from partial elements of a random basis"""
        # Do 25 test runs with random elements from a random basis in
        # (2 ... 8) dimensions
        for _ in range(25):
            d = rng.integers(2, 7)
            # Get a random traceless hermitian operator
            oper = testutil.rand_herm_traceless(d)
            # ... and build a basis from it
            b = ff.Basis.from_partial(oper)
            self.assertTrue(b.isorthonorm)
            self.assertTrue(b.isherm)
            self.assertTrue(b.istraceless)
            self.assertTrue(b.iscomplete)
            # Choose random elements from that basis and generate a new basis
            # from it
            inds = [i for i in range(d**2)]
            tup = tuple(inds.pop(rng.integers(0, len(inds))) for _ in range(rng.integers(1, d**2)))
            elems = b[tup, ...]
            basis = ff.Basis.from_partial(elems)
            self.assertTrue(basis.isorthonorm)
            self.assertTrue(basis.isherm)
            self.assertTrue(basis.istraceless)
            self.assertTrue(basis.iscomplete)
            self.assertTrue(all(elem in basis for elem in elems))

        # Test runs with non-traceless opers
        for _ in range(25):
            d = rng.integers(2, 7)
            # Get a random hermitian operator
            oper = testutil.rand_herm(d)
            # ... and build a basis from it
            b = ff.Basis.from_partial(oper)
            self.assertTrue(b.isorthonorm)
            self.assertTrue(b.isherm)
            self.assertFalse(b.istraceless)
            self.assertTrue(b.iscomplete)
            # Choose random elements from that basis and generate a new basis
            # from it
            inds = [i for i in range(d**2)]
            tup = tuple(inds.pop(rng.integers(0, len(inds))) for _ in range(rng.integers(1, d**2)))
            elems = b[tup, ...]
            basis = ff.Basis.from_partial(elems)
            self.assertTrue(basis.isorthonorm)
            self.assertTrue(basis.isherm)
            self.assertFalse(basis.istraceless)
            self.assertTrue(basis.iscomplete)
            self.assertTrue(all(elem in basis for elem in elems))

    def test_filter_functions(self):
        """Filter functions equal for different bases"""
        # Set up random Hamiltonian
        base_pulse = testutil.rand_pulse_sequence(4, 3, 1, 1)
        omega = util.get_sample_frequencies(base_pulse, n_samples=200)

        pauli_basis = ff.Basis.pauli(2)
        ggm_basis = ff.Basis.ggm(4)
        from_random_basis = ff.Basis.from_partial(base_pulse.n_opers)
        bases = (pauli_basis, ggm_basis, from_random_basis)

        # Get Pulses with different bases
        F = []
        for b in bases:
            pulse = copy(base_pulse)
            pulse.basis = b
            F.append(pulse.get_filter_function(omega).sum(0))

        for pair in product(F, F):
            self.assertArrayAlmostEqual(*pair)

    def test_control_matrix(self):
        """Test control matrix for traceless and non-traceless bases"""
        n_opers = testutil.rand_herm(3, 4)
        n_opers_traceless = testutil.rand_herm_traceless(3, 4)

        basis = ff.Basis.from_partial(testutil.rand_herm(3), traceless=False)
        basis_traceless = ff.Basis.from_partial(testutil.rand_herm_traceless(3), traceless=True)

        base_pulse = testutil.rand_pulse_sequence(3, 10, 4, 4)
        omega = np.logspace(-1, 1, 51)

        for i, base in enumerate((basis, basis_traceless)):
            for j, n_ops in enumerate((n_opers, n_opers_traceless)):
                pulse = copy(base_pulse)
                pulse.n_opers = n_ops
                pulse.basis = base
                B = pulse.get_control_matrix(omega)

                if i == 0 and j == 0:
                    # base not traceless, nopers not traceless
                    self.assertTrue((B[:, 0] != 0).all())
                elif i == 0 and j == 1:
                    # base not traceless, nopers traceless
                    self.assertTrue((B[:, 0] != 0).all())
                elif i == 1 and j == 0:
                    # base traceless, nopers not traceless
                    self.assertTrue((B[:, 0] != 0).all())
                elif i == 1 and j == 1:
                    # base traceless, nopers traceless
                    self.assertTrue(np.allclose(B[:, 0], 0))


@pytest.mark.skipif(
    qutip is None,
    reason='Skipping qutip compatibility test for build without qutip')
class QutipCompatibilityTest(testutil.TestCase):

    def test_constructor(self):
        """Test if can create basis from qutip objects."""

        elems_qutip = [qutip.sigmay(), qutip.qeye(2).data]
        elems_np = [qutip.sigmay().full(), qutip.qeye(2).full()]

        basis_qutip = ff.Basis(elems_qutip)
        basis_np = ff.Basis(elems_np)

        self.assertArrayEqual(basis_qutip, basis_np)

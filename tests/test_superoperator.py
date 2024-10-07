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
This module tests the superoperator module.
"""

import numpy as np

import filter_functions as ff
from filter_functions import superoperator
from tests import testutil
from tests.testutil import rng


class SuperoperatorTest(testutil.TestCase):

    def test_liouville_representation(self):
        """Test the calculation of the transfer matrix"""
        dd = np.arange(2, 18, 4)

        for d in dd:
            # Works with different bases
            if d == 4:
                basis = ff.Basis.pauli(2)
            else:
                basis = ff.Basis.ggm(d)

            U = testutil.rand_unit(d, 2)

            # Works on matrices and arrays of matrices
            U_liouville = superoperator.liouville_representation(U[0], basis)
            U_liouville = superoperator.liouville_representation(U, basis)

            # should have dimension d^2 x d^2
            self.assertEqual(U_liouville.shape, (U.shape[0], d**2, d**2))
            # Real
            self.assertTrue(np.isreal(U_liouville).all())
            # Hermitian for unitary input
            self.assertArrayAlmostEqual(
                U_liouville.swapaxes(-1, -2) @ U_liouville,
                np.tile(np.eye(d**2), (U.shape[0], 1, 1)),
                atol=5*np.finfo(float).eps*d**2
            )

            if d == 2:
                U_liouville = superoperator.liouville_representation(
                    ff.util.paulis[1:], basis)
                self.assertArrayAlmostEqual(U_liouville[0],
                                            np.diag([1, 1, -1, -1]),
                                            atol=basis._atol)
                self.assertArrayAlmostEqual(U_liouville[1],
                                            np.diag([1, -1, 1, -1]),
                                            atol=basis._atol)
                self.assertArrayAlmostEqual(U_liouville[2],
                                            np.diag([1, -1, -1, 1]),
                                            atol=basis._atol)

    def test_liouville_to_choi(self):
        """Test converting Liouville superops to choi matrices."""
        for d in rng.integers(2, 9, (15,)):
            # unitary channel
            U = testutil.rand_unit(d, rng.integers(1, 8)).squeeze()
            n = np.log2(d)
            if n % 1 == 0:
                basis = ff.Basis.pauli(int(n))
            else:
                basis = ff.Basis.ggm(d)

            U_sup = superoperator.liouville_representation(U, basis)
            choi = superoperator.liouville_to_choi(U_sup, basis).view(ff.Basis)

            self.assertTrue(choi.isherm)
            self.assertArrayAlmostEqual(np.einsum('...ii', choi), d)

            pulse = testutil.rand_pulse_sequence(d, 1)
            omega = ff.util.get_sample_frequencies(pulse)
            S = 1/abs(omega)**2

            U_sup = ff.error_transfer_matrix(pulse, S, omega)
            choi = superoperator.liouville_to_choi(U_sup, basis).view(ff.Basis)

            self.assertTrue(choi.isherm)
            self.assertAlmostEqual(np.einsum('ii', choi), d)

    def test_liouville_is_CP(self):
        def partial_transpose(A):
            d = A.shape[-1]
            sqd = int(np.sqrt(d))
            return A.reshape(-1, sqd, sqd, sqd, sqd).swapaxes(-1, -3).reshape(A.shape)

        # Partial transpose map should be non-CP
        basis = ff.Basis.pauli(2)
        Phi = ff.basis.expand(partial_transpose(basis), basis).T
        CP = superoperator.liouville_is_CP(Phi, basis)
        self.assertFalse(CP)

        for d in rng.integers(2, 9, (15,)):
            # unitary channel
            U = testutil.rand_unit(d, rng.integers(1, 8)).squeeze()
            n = np.log2(d)
            if n % 1 == 0:
                basis = ff.Basis.pauli(int(n))
            else:
                basis = ff.Basis.ggm(d)

            U_sup = superoperator.liouville_representation(U, basis)
            CP, (D, V) = superoperator.liouville_is_CP(U_sup, basis, True)

            _CP = superoperator.liouville_is_CP(U_sup, basis, False)

            self.assertArrayEqual(CP, _CP)
            self.assertTrue(np.all(CP))
            if U_sup.ndim == 2:
                self.assertIsInstance(CP, (bool, np.bool_))
            else:
                self.assertEqual(CP.shape[0], U_sup.shape[0])
            # Only one nonzero eigenvalue
            self.assertArrayAlmostEqual(D[..., :-1], 0, atol=basis._atol)

            pulse = testutil.rand_pulse_sequence(d, 1)
            omega = ff.util.get_sample_frequencies(pulse)
            S = 1/abs(omega)**2

            U_sup = ff.error_transfer_matrix(pulse, S, omega)
            CP = superoperator.liouville_is_CP(U_sup, pulse.basis)

            self.assertTrue(np.all(CP))
            self.assertIsInstance(CP, (bool, np.bool_))

    def test_liouville_is_cCP(self):
        for d in rng.integers(2, 9, (15,)):
            # (anti-) Hermitian generator, should always be cCP
            H = 1j*testutil.rand_herm(d, rng.integers(1, 8)).squeeze()
            n = np.log2(d)
            if n % 1 == 0:
                basis = ff.Basis.pauli(int(n))
            else:
                basis = ff.Basis.ggm(d)

            H_sup = (np.einsum('iab,...bc,jca', basis, H, basis,
                               optimize=['einsum_path', (0, 1), (0, 1)])
                     - np.einsum('iab,jbc,...ca', basis, basis, H,
                                 optimize=['einsum_path', (0, 2), (0, 1)]))
            cCP, (D, V) = superoperator.liouville_is_cCP(H_sup, basis, True)
            _cCP = superoperator.liouville_is_cCP(H_sup, basis, False)

            self.assertArrayEqual(cCP, _cCP)
            self.assertTrue(np.all(cCP))
            if H_sup.ndim == 2:
                self.assertIsInstance(cCP, (bool, np.bool_))
            else:
                self.assertEqual(cCP.shape[0], H_sup.shape[0])
            self.assertArrayAlmostEqual(D, 0, atol=1e-14)
            self.assertTrue(ff.util.oper_equiv(V, np.eye(d**2)[None, :, :],
                                               normalized=True))

            pulse = testutil.rand_pulse_sequence(d, 1)
            omega = ff.util.get_sample_frequencies(pulse)
            S = 1/abs(omega)**2

            K_sup = ff.numeric.calculate_cumulant_function(pulse, S, omega)
            cCP = superoperator.liouville_is_cCP(K_sup, pulse.basis, False,
                                                 atol=1e-13)

            self.assertTrue(np.all(cCP))
            if K_sup.ndim == 2:
                self.assertIsInstance(cCP, (bool, np.bool_))
            else:
                self.assertEqual(cCP.shape[0], K_sup.shape[0])

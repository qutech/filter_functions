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
This module tests the concatenation functionality for PulseSequence's.
"""

import string
from itertools import product
from random import sample

import numpy as np
from numpy.random import choice, randint, randn
from tests import testutil

import filter_functions as ff
from filter_functions.util import P_np, get_sample_frequencies, tensor


class ConcatenationTest(testutil.TestCase):

    def test_concatenate_without_filter_function(self):
        """Concatenate two Spin Echos without filter functions."""
        tau = 10
        tau_pi = 1e-4
        n = 1

        H_c_SE, dt_SE = testutil.generate_dd_hamiltonian(n, tau=tau,
                                                         tau_pi=tau_pi,
                                                         dd_type='cpmg')

        n_oper = P_np[3]
        H_n_SE = [[n_oper, np.ones_like(dt_SE)]]
        SE_1 = ff.PulseSequence(H_c_SE, H_n_SE, dt_SE)
        SE_2 = ff.PulseSequence(H_c_SE, H_n_SE, dt_SE)

        H_c_CPMG, dt_CPMG = testutil.generate_dd_hamiltonian(2*n, tau=2*tau,
                                                             tau_pi=tau_pi,
                                                             dd_type='cpmg')

        H_n_CPMG = [[n_oper, np.ones_like(dt_CPMG)]]
        CPMG = ff.PulseSequence(H_c_CPMG, H_n_CPMG, dt_CPMG)
        CPMG_concat = SE_1 @ SE_2

        self.assertEqual(CPMG_concat, CPMG)
        self.assertEqual(CPMG_concat._F, CPMG._F)   # Should still be None

        # Test if calculation of composite filter function can be enforced with
        # omega != None
        omega = get_sample_frequencies(SE_1)
        CPMG_concat = ff.concatenate((SE_1, SE_2), omega=omega)
        self.assertIsNotNone(CPMG_concat._F)

    def test_concatenate_with_filter_function_SE1(self):
        """
        Concatenate two Spin Echos with the first having a filter function.
        """
        tau = 10
        tau_pi = 1e-4
        omega = np.logspace(-1, 2, 500)
        n = 1

        H_c_SE, dt_SE = testutil.generate_dd_hamiltonian(n, tau=tau,
                                                         tau_pi=tau_pi,
                                                         dd_type='cpmg')

        H_n_SE = [[P_np[3], np.ones_like(dt_SE)]]
        SE_1 = ff.PulseSequence(H_c_SE, H_n_SE, dt_SE)
        SE_2 = ff.PulseSequence(H_c_SE, H_n_SE, dt_SE)

        H_c_CPMG, dt_CPMG = testutil.generate_dd_hamiltonian(2*n, tau=2*tau,
                                                             tau_pi=tau_pi,
                                                             dd_type='cpmg')

        H_n_CPMG = [[P_np[3], np.ones_like(dt_CPMG)]]
        CPMG = ff.PulseSequence(H_c_CPMG, H_n_CPMG, dt_CPMG)

        SE_1.cache_filter_function(omega)
        CPMG.cache_filter_function(omega)

        CPMG_concat = SE_1 @ SE_2

        self.assertIsNotNone(SE_1._total_phases)
        self.assertIsNotNone(SE_1._total_Q)
        self.assertIsNotNone(SE_1._total_Q_liouville)
        self.assertIsNotNone(CPMG._total_phases)
        self.assertIsNotNone(CPMG._total_Q)
        self.assertIsNotNone(CPMG._total_Q_liouville)
        self.assertIsNotNone(CPMG_concat._total_phases)
        self.assertIsNotNone(CPMG_concat._total_Q)
        self.assertIsNotNone(CPMG_concat._total_Q_liouville)

        self.assertEqual(CPMG_concat, CPMG)
        self.assertArrayAlmostEqual(CPMG_concat._F, CPMG._F, rtol=1e-11)

    def test_concatenate_with_filter_function_SE2(self):
        """
        Concatenate two Spin Echos with the second having a filter function.
        """
        tau = 10
        tau_pi = 1e-4
        omega = np.logspace(-1, 2, 500)
        n = 1

        H_c_SE, dt_SE = testutil.generate_dd_hamiltonian(n, tau=tau,
                                                         tau_pi=tau_pi,
                                                         dd_type='cpmg')

        H_n_SE = [[P_np[3], np.ones_like(dt_SE)]]
        SE_1 = ff.PulseSequence(H_c_SE, H_n_SE, dt_SE)
        SE_2 = ff.PulseSequence(H_c_SE, H_n_SE, dt_SE)

        H_c_CPMG, dt_CPMG = testutil.generate_dd_hamiltonian(2*n, tau=2*tau,
                                                             tau_pi=tau_pi,
                                                             dd_type='cpmg')

        H_n_CPMG = [[P_np[3], np.ones_like(dt_CPMG)]]
        CPMG = ff.PulseSequence(H_c_CPMG, H_n_CPMG, dt_CPMG)

        SE_2.cache_filter_function(omega)
        CPMG.cache_filter_function(omega)

        CPMG_concat = SE_1 @ SE_2

        self.assertIsNotNone(SE_2._total_phases)
        self.assertIsNotNone(SE_2._total_Q)
        self.assertIsNotNone(SE_2._total_Q_liouville)
        self.assertIsNotNone(CPMG._total_phases)
        self.assertIsNotNone(CPMG._total_Q)
        self.assertIsNotNone(CPMG._total_Q_liouville)
        self.assertIsNotNone(CPMG_concat._total_phases)
        self.assertIsNotNone(CPMG_concat._total_Q)
        self.assertIsNotNone(CPMG_concat._total_Q_liouville)

        self.assertEqual(CPMG_concat, CPMG)
        self.assertArrayAlmostEqual(CPMG_concat._F, CPMG._F, rtol=1e-11)

    def test_concatenate_with_filter_function_SE12(self):
        """Concatenate two Spin Echos with both having a filter function."""
        tau = 10
        tau_pi = 1e-4
        omega = np.logspace(-1, 2, 500)
        n = 1

        H_c_SE, dt_SE = testutil.generate_dd_hamiltonian(n, tau=tau,
                                                         tau_pi=tau_pi,
                                                         dd_type='cpmg')

        H_n_SE = [[P_np[3], np.ones_like(dt_SE)]]
        SE_1 = ff.PulseSequence(H_c_SE, H_n_SE, dt_SE)
        SE_2 = ff.PulseSequence(H_c_SE, H_n_SE, dt_SE)

        H_c_CPMG, dt_CPMG = testutil.generate_dd_hamiltonian(2*n, tau=2*tau,
                                                             tau_pi=tau_pi,
                                                             dd_type='cpmg')

        H_n_CPMG = [[P_np[3], np.ones_like(dt_CPMG)]]
        CPMG = ff.PulseSequence(H_c_CPMG, H_n_CPMG, dt_CPMG)

        SE_1.cache_filter_function(omega)
        SE_2.cache_filter_function(omega)
        CPMG.cache_filter_function(omega)

        CPMG_concat = SE_1 @ SE_2

        self.assertIsNotNone(SE_1._total_phases)
        self.assertIsNotNone(SE_1._total_Q)
        self.assertIsNotNone(SE_1._total_Q_liouville)
        self.assertIsNotNone(SE_2._total_phases)
        self.assertIsNotNone(SE_2._total_Q)
        self.assertIsNotNone(SE_2._total_Q_liouville)
        self.assertIsNotNone(CPMG._total_phases)
        self.assertIsNotNone(CPMG._total_Q)
        self.assertIsNotNone(CPMG._total_Q_liouville)
        self.assertIsNotNone(CPMG_concat._total_phases)
        self.assertIsNotNone(CPMG_concat._total_Q)
        self.assertIsNotNone(CPMG_concat._total_Q_liouville)

        self.assertEqual(CPMG_concat, CPMG)
        self.assertArrayAlmostEqual(CPMG_concat._F, CPMG._F, rtol=1e-11)

    def test_concatenate_4_spin_echos(self):
        """Concatenate four Spin Echos with a random one having a filter
        function
        """
        tau = 1
        tau_pi = 1e-4
        omega = np.logspace(-2, 1, 200)
        n = 1

        H_c_SE, dt_SE = testutil.generate_dd_hamiltonian(n, tau=tau,
                                                         tau_pi=tau_pi,
                                                         dd_type='cpmg')

        H_n_SE = [[P_np[3], np.ones_like(dt_SE)]]
        SE = [ff.PulseSequence(H_c_SE, H_n_SE, dt_SE) for _ in range(4)]

        H_c_CPMG, dt_CPMG = testutil.generate_dd_hamiltonian(4*n, tau=4*tau,
                                                             tau_pi=tau_pi,
                                                             dd_type='cpmg')

        H_n_CPMG = [[P_np[3], np.ones_like(dt_CPMG)]]
        CPMG = ff.PulseSequence(H_c_CPMG, H_n_CPMG, dt_CPMG)

        SE[randint(0, len(SE)-1)].cache_filter_function(omega)
        CPMG.cache_filter_function(omega)

        CPMG_concat_1 = ff.concatenate(SE)
        # Clean up so that we start from one SE with cached F again
        for se in SE:
            se.cleanup('all')

        SE[randint(0, len(SE)-1)].cache_filter_function(omega)
        CPMG_concat_2 = SE[0] @ SE[1] @ SE[2] @ SE[3]

        self.assertEqual(CPMG, CPMG_concat_1)
        self.assertEqual(CPMG, CPMG_concat_2)
        self.assertArrayAlmostEqual(CPMG_concat_1._F, CPMG._F, rtol=1e-10)
        self.assertArrayAlmostEqual(CPMG_concat_2._F, CPMG._F, rtol=1e-10)

    def test_concatenate_split_cnot(self):
        """Split up cnot and concatenate the parts."""
        c_opers, c_coeffs, dt = (testutil.subspace_opers, testutil.c_coeffs,
                                 testutil.dt)
        n_opers = c_opers
        n_coeffs = testutil.n_coeffs

        H_c = list(zip(c_opers, c_coeffs))
        H_n = list(zip(n_opers, n_coeffs))
        omega = np.logspace(-2, 1, 200)

        cnot_whole = ff.PulseSequence(H_c, H_n, dt)

        cnot_sliced = [
            ff.PulseSequence(list(zip(c_opers, [c[:10] for c in c_coeffs])),
                             list(zip(n_opers, [n[:10] for n in n_coeffs])),
                             dt[:10]),
            ff.PulseSequence(list(zip(c_opers, [c[10:15] for c in c_coeffs])),
                             list(zip(n_opers, [n[10:15] for n in n_coeffs])),
                             dt[10:15]),
            ff.PulseSequence(list(zip(c_opers, [c[15:100] for c in c_coeffs])),
                             list(zip(n_opers, [n[15:100] for n in n_coeffs])),
                             dt[15:100]),
            ff.PulseSequence(list(zip(c_opers,
                                      [c[100:245] for c in c_coeffs])),
                             list(zip(n_opers,
                                      [n[100:245] for n in n_coeffs])),
                             dt[100:245]),
            ff.PulseSequence(list(zip(c_opers, [c[245:] for c in c_coeffs])),
                             list(zip(n_opers, [n[245:] for n in n_coeffs])),
                             dt[245:])
        ]

        cnot_concatenated = ff.concatenate(cnot_sliced)

        self.assertEqual(cnot_whole, cnot_concatenated)
        self.assertEqual(cnot_whole._F, cnot_concatenated._F)

        cnot_concatenated.cache_filter_function(omega)
        cnot_whole.cache_filter_function(omega)

        self.assertEqual(cnot_whole, cnot_concatenated)
        self.assertArrayAlmostEqual(cnot_whole._F, cnot_concatenated._F)

        # Test concatenation if different child sequences have a filter
        # function already calculated
        cnot_sliced = [
            ff.PulseSequence(list(zip(c_opers, [c[:100] for c in c_coeffs])),
                             list(zip(n_opers, [n[:100] for n in n_coeffs])),
                             dt[:100]),
            ff.PulseSequence(list(zip(c_opers,
                                      [c[100:150] for c in c_coeffs])),
                             list(zip(n_opers,
                                      [n[100:150] for n in n_coeffs])),
                             dt[100:150]),
            ff.PulseSequence(list(zip(c_opers, [c[150:] for c in c_coeffs])),
                             list(zip(n_opers, [n[150:] for n in n_coeffs])),
                             dt[150:])
        ]

        atol = 1e-12
        rtol = 1e-10

        for slice_1, slice_2 in product(cnot_sliced, cnot_sliced):

            for cnot_slice in cnot_sliced:
                cnot_slice.cleanup('all')

            slice_1.cache_filter_function(omega)
            slice_2.cache_filter_function(omega)

            cnot_concatenated = ff.concatenate(cnot_sliced)

            self.assertArrayEqual(cnot_whole.omega, cnot_concatenated.omega)
            self.assertArrayAlmostEqual(cnot_whole._F, cnot_concatenated._F,
                                        rtol, atol)

    def test_different_n_opers(self):
        """Test behavior when concatenating with different n_opers"""
        for d, n_dt in zip(randint(2, 5, 20), randint(1, 11, 20)):
            opers = testutil.rand_herm_traceless(d, 10)
            letters = np.array(sample(list(string.ascii_letters), 10))
            n_idx = sample(range(10), randint(2, 5))
            c_idx = sample(range(10), randint(2, 5))
            n_opers = opers[n_idx]
            c_opers = opers[c_idx]
            n_coeffs = np.ones((n_opers.shape[0], n_dt))
            n_coeffs *= np.abs(randn(n_opers.shape[0], 1))
            c_coeffs = randn(c_opers.shape[0], n_dt)
            dt = np.abs(randn(n_dt))
            n_ids = np.array([''.join(l) for l in letters[n_idx]])
            c_ids = np.array([''.join(l) for l in letters[c_idx]])

            pulse_1 = ff.PulseSequence(list(zip(c_opers, c_coeffs, c_ids)),
                                       list(zip(n_opers, n_coeffs, n_ids)),
                                       dt)
            permutation = np.random.permutation(range(n_opers.shape[0]))
            pulse_2 = ff.PulseSequence(list(zip(c_opers, c_coeffs, c_ids)),
                                       list(zip(n_opers[permutation],
                                                n_coeffs[permutation],
                                                n_ids[permutation])),
                                       dt)
            more_n_idx = sample(range(10), randint(2, 5))
            more_n_opers = opers[more_n_idx]
            more_n_coeffs = np.ones((more_n_opers.shape[0], n_dt))
            more_n_coeffs *= np.abs(randn(more_n_opers.shape[0], 1))
            more_n_ids = np.array([''.join(l) for l in letters[more_n_idx]])
            pulse_3 = ff.PulseSequence(list(zip(c_opers, c_coeffs, c_ids)),
                                       list(zip(more_n_opers, more_n_coeffs,
                                                more_n_ids)),
                                       dt)

            nontrivial_n_coeffs = np.abs(randn(n_opers.shape[0], n_dt))
            pulse_4 = ff.PulseSequence(list(zip(c_opers, c_coeffs, c_ids)),
                                       list(zip(more_n_opers,
                                                nontrivial_n_coeffs,
                                                more_n_ids)),
                                       dt)

            omega = np.geomspace(.1, 10, 50)

            # Test caching
            with self.assertRaises(ValueError):
                ff.concatenate([pulse_1, pulse_3],
                               calc_filter_function=True)

            with self.assertRaises(ValueError):
                ff.concatenate([pulse_1, pulse_3],
                               calc_pulse_correlation_ff=True)

            pulse_1.cache_filter_function(omega)
            pulse_2.cache_filter_function(omega)
            pulse_3.cache_filter_function(omega)
            pulse_11 = ff.concatenate([pulse_1, pulse_1])
            pulse_12 = ff.concatenate([pulse_1, pulse_2])
            pulse_13_1 = ff.concatenate([pulse_1, pulse_3])
            pulse_13_2 = ff.concatenate([pulse_1, pulse_3],
                                        calc_filter_function=True)

            # concatenate pulses with different n_opers and nontrivial sens.
            subset = (set.issubset(set(pulse_1.n_oper_identifiers),
                                   set(pulse_4.n_oper_identifiers)) or
                      set.issubset(set(pulse_4.n_oper_identifiers),
                                   set(pulse_1.n_oper_identifiers)))

            if not subset and (len(pulse_1.dt) > 1 or len(pulse_4.dt) > 1):
                with self.assertRaises(ValueError):
                    pulse_1 @ pulse_4

            self.assertEqual(pulse_11, pulse_12)
            # Filter functions should be the same even though pulse_2 has
            # different n_oper ordering
            self.assertArrayAlmostEqual(pulse_11._R, pulse_12._R, atol=1e-12)
            self.assertArrayAlmostEqual(pulse_11._F, pulse_12._F, atol=1e-12)

            should_be_cached = False
            for i in n_idx:
                if i in more_n_idx:
                    should_be_cached = True
            self.assertEqual(should_be_cached, pulse_13_1.is_cached('F'))

            # Test forcibly caching
            self.assertTrue(pulse_13_2.is_cached('F'))

    def test_concatenation_periodic(self):
        """Test concatenation for periodic Hamiltonians"""
        X, Y, Z = P_np[1:]
        A = 0.01
        omega_0 = 1
        omega_d = omega_0
        tau = np.pi/A
        omega = np.logspace(np.log10(omega_0) - 3, np.log10(omega_0) + 3, 1001)

        t = np.linspace(0, tau, 1001)
        dt = np.diff(t)
        H_c = [[Z, [omega_0/2]*len(dt)],
               [X, A*np.cos(omega_d*t[1:])]]
        H_n = [[Z, np.ones_like(dt)],
               [X, np.ones_like(dt)]]

        NOT_LAB = ff.PulseSequence(H_c, H_n, dt)
        F_LAB = NOT_LAB.get_filter_function(omega)

        T = 2*np.pi/omega_d
        G = round(tau/T)
        t = np.linspace(0, T, int(T/NOT_LAB.dt[0])+1)
        dt = np.diff(t)

        H_c = [[Z, [omega_0/2]*len(dt)],
               [X, A*np.cos(omega_d*t[1:])]]
        H_n = [[Z, np.ones_like(dt)],
               [X, np.ones_like(dt)]]

        ATOMIC = ff.PulseSequence(H_c, H_n, dt)
        ATOMIC.cache_filter_function(omega)

        NOT_CC = ff.concatenate((ATOMIC for _ in range(G)))
        F_CC = NOT_CC.get_filter_function(omega)
        NOT_CC_PERIODIC = ff.concatenate_periodic(ATOMIC, G)
        F_CC_PERIODIC = NOT_CC_PERIODIC.get_filter_function(omega)

        # Have to do manual comparison due to floating point error. The high
        # rtol is due to 1e-21/1e-19 occurring once.
        attrs = ('dt', 'c_opers', 'c_coeffs', 'n_opers', 'n_coeffs')
        for attr in attrs:
            self.assertArrayAlmostEqual(getattr(NOT_LAB, attr),
                                        getattr(NOT_CC, attr),
                                        atol=1e-15, rtol=1e2)
            self.assertArrayAlmostEqual(getattr(NOT_LAB, attr),
                                        getattr(NOT_CC_PERIODIC, attr),
                                        atol=1e-15, rtol=1e2)

        # Check if stuff is cached
        self.assertIsNotNone(NOT_CC._total_phases)
        self.assertIsNotNone(NOT_CC._total_Q)
        self.assertIsNotNone(NOT_CC._total_Q_liouville)
        # concatenate_periodic does not cache phase factors
        self.assertIsNotNone(NOT_CC_PERIODIC._total_phases)
        self.assertIsNotNone(NOT_CC_PERIODIC._total_Q)
        self.assertIsNotNone(NOT_CC_PERIODIC._total_Q_liouville)

        self.assertArrayAlmostEqual(F_LAB, F_CC, atol=1e-13)
        self.assertArrayAlmostEqual(F_LAB, F_CC_PERIODIC, atol=1e-13)


class ExtensionTest(testutil.TestCase):

    def test_extend_with_identity(self):
        """Test extending a pulse to more qubits"""
        ID, X, Y, Z = P_np
        n_dt = 10
        coeffs = randn(3, n_dt)
        ids = ['X', 'Y', 'Z']
        pulse = ff.PulseSequence(
            list(zip((X, Y, Z), coeffs, ids)),
            list(zip((X, Y, Z), np.ones((3, n_dt)), ids)),
            np.ones(n_dt), basis=ff.Basis.pauli(1)
        )

        omega = get_sample_frequencies(pulse, spacing='log', n_samples=50)
        for N in randint(2, 5, 4):
            for target in randint(0, N-1, 2):
                pulse.cleanup('all')
                ext_opers = tensor(*np.insert(np.tile(ID, (N-1, 3, 1, 1)),
                                              target, (X, Y, Z), axis=0))

                # By default, extend should add the target qubit as suffix to
                # identifiers
                ext_ids = [i + '_{}'.format(target) for i in ids]
                ext_pulse = ff.PulseSequence(
                    list(zip(ext_opers, coeffs, ext_ids)),
                    list(zip(ext_opers, np.ones((3, n_dt)), ext_ids)),
                    np.ones(n_dt), basis=ff.Basis.pauli(N)
                )

                # Use custom mapping for identifiers and or labels
                letters = choice(list(string.ascii_letters), size=(3, 5))
                mapped_ids = np.array([''.join(l) for l in letters])
                mapping = {i: new_id for i, new_id in zip(ids, mapped_ids)}
                ext_pulse_mapped_identifiers = ff.PulseSequence(
                    list(zip(ext_opers, coeffs, mapped_ids, ext_ids)),
                    list(zip(ext_opers, np.ones((3, n_dt)), mapped_ids,
                             ext_ids)),
                    np.ones(n_dt), basis=ff.Basis.pauli(N)
                )
                ext_pulse_mapped_labels = ff.PulseSequence(
                    list(zip(ext_opers, coeffs, ext_ids, mapped_ids)),
                    list(zip(ext_opers, np.ones((3, n_dt)), ext_ids,
                             mapped_ids)),
                    np.ones(n_dt), basis=ff.Basis.pauli(N)
                )
                ext_pulse_mapped_identifiers_labels = ff.PulseSequence(
                    list(zip(ext_opers, coeffs, mapped_ids)),
                    list(zip(ext_opers, np.ones((3, n_dt)), mapped_ids)),
                    np.ones(n_dt), basis=ff.Basis.pauli(N)
                )

                calc_FF = randint(0, 2)
                if calc_FF:
                    # Expect things to be cached in extended pulse if original
                    # also was cached
                    pulse.cache_filter_function(omega)
                    ext_pulse.cache_filter_function(omega)

                test_ext_pulse = ff.extend([(pulse, target)], N, d_per_qubit=2)
                test_ext_pulse_mapped_identifiers = ff.extend(
                    [(pulse, target, mapping)], N, d_per_qubit=2
                )
                test_ext_pulse_mapped_labels = ff.extend(
                    [(pulse, target, None, mapping)], N, d_per_qubit=2
                )
                test_ext_pulse_mapped_identifiers_labels = ff.extend(
                    [(pulse, target, mapping, mapping)], N, d_per_qubit=2
                )

                self.assertEqual(ext_pulse, test_ext_pulse)
                self.assertEqual(ext_pulse_mapped_identifiers,
                                 test_ext_pulse_mapped_identifiers)
                self.assertEqual(ext_pulse_mapped_labels,
                                 test_ext_pulse_mapped_labels)
                self.assertEqual(ext_pulse_mapped_identifiers_labels,
                                 test_ext_pulse_mapped_identifiers_labels)

                if calc_FF:
                    self.assertCorrectDiagonalization(test_ext_pulse,
                                                      atol=1e-14)
                    self.assertArrayAlmostEqual(test_ext_pulse._Q,
                                                ext_pulse._Q, atol=1e-14)
                    self.assertArrayAlmostEqual(
                        test_ext_pulse._total_Q_liouville,
                        ext_pulse._total_Q_liouville,
                        atol=1e-14
                    )
                    self.assertArrayAlmostEqual(test_ext_pulse._total_Q,
                                                ext_pulse._total_Q, atol=1e-14)
                    self.assertArrayAlmostEqual(test_ext_pulse._total_phases,
                                                ext_pulse._total_phases)
                    self.assertArrayAlmostEqual(test_ext_pulse._R,
                                                ext_pulse._R, atol=1e-12)
                    self.assertArrayAlmostEqual(test_ext_pulse._F,
                                                ext_pulse._F, atol=1e-12)
                else:
                    self.assertIsNone(test_ext_pulse._HD)
                    self.assertIsNone(test_ext_pulse._HV)
                    self.assertIsNone(test_ext_pulse._Q)
                    self.assertIsNone(test_ext_pulse._total_Q_liouville)
                    self.assertIsNone(test_ext_pulse._total_Q)
                    self.assertIsNone(test_ext_pulse._total_phases)
                    self.assertIsNone(test_ext_pulse._R)
                    self.assertIsNone(test_ext_pulse._F)

                pulse.cleanup('all')
                ext_pulse.cleanup('all')

    def test_caching(self):
        """Test caching"""
        ID, X, Y, Z = P_np
        n_dt = 10
        coeffs = randn(3, n_dt)
        pulse_1 = ff.PulseSequence(
            list(zip((X, Y, Z), coeffs)),
            list(zip((X, Y, Z), np.ones((3, n_dt)))),
            np.ones(n_dt), basis=ff.Basis.pauli(1)
        )
        pulse_2 = ff.PulseSequence(
            list(zip((X, Y, Z), coeffs)),
            list(zip((X, Y, Z), np.ones((3, n_dt)))),
            np.ones(n_dt), basis=ff.Basis.pauli(1)
        )
        omega = get_sample_frequencies(pulse_1, 50)

        # diagonalize one pulse
        pulse_1.diagonalize()
        extended_pulse = ff.extend([(pulse_1, 0), (pulse_2, 1)])
        self.assertIsNone(extended_pulse._HD)
        self.assertIsNone(extended_pulse._HV)
        self.assertIsNone(extended_pulse._Q)
        self.assertIsNone(extended_pulse._total_Q)
        self.assertIsNone(extended_pulse._total_Q_liouville)
        self.assertIsNone(extended_pulse._total_phases)
        self.assertIsNone(extended_pulse._R)
        self.assertIsNone(extended_pulse._F)

        # override
        extended_pulse = ff.extend([(pulse_1, 0), (pulse_2, 1)],
                                   cache_diagonalization=True)
        self.assertIsNotNone(extended_pulse._HD)
        self.assertIsNotNone(extended_pulse._HV)
        self.assertIsNotNone(extended_pulse._Q)
        self.assertIsNotNone(extended_pulse._total_Q)
        self.assertIsNone(extended_pulse._total_Q_liouville)
        self.assertIsNone(extended_pulse._total_phases)
        self.assertIsNone(extended_pulse._R)
        self.assertIsNone(extended_pulse._F)

        # diagonalize both
        pulse_2.diagonalize()
        extended_pulse = ff.extend([(pulse_1, 0), (pulse_2, 1)])
        self.assertIsNotNone(extended_pulse._HD)
        self.assertIsNotNone(extended_pulse._HV)
        self.assertIsNotNone(extended_pulse._Q)
        self.assertIsNotNone(extended_pulse._total_Q)
        self.assertIsNone(extended_pulse._total_Q_liouville)
        self.assertIsNone(extended_pulse._total_phases)
        self.assertIsNone(extended_pulse._R)
        self.assertIsNone(extended_pulse._F)

        # override
        extended_pulse = ff.extend([(pulse_1, 0), (pulse_2, 1)],
                                   cache_diagonalization=False)
        self.assertIsNone(extended_pulse._HD)
        self.assertIsNone(extended_pulse._HV)
        self.assertIsNone(extended_pulse._Q)
        # Total_Q is still cached
        self.assertIsNotNone(extended_pulse._total_Q)
        self.assertIsNone(extended_pulse._total_Q_liouville)
        self.assertIsNone(extended_pulse._total_phases)
        self.assertIsNone(extended_pulse._R)
        self.assertIsNone(extended_pulse._F)

        # Get filter function for one pulse
        pulse_1.cache_filter_function(omega)
        extended_pulse = ff.extend([(pulse_1, 0), (pulse_2, 1)])
        self.assertIsNone(extended_pulse._total_Q_liouville)
        self.assertIsNone(extended_pulse._total_phases)
        self.assertIsNone(extended_pulse._R)
        self.assertIsNone(extended_pulse._F)

        # override
        extended_pulse = ff.extend([(pulse_1, 0), (pulse_2, 1)],
                                   cache_filter_function=True,
                                   omega=omega)
        self.assertIsNotNone(extended_pulse._total_Q_liouville)
        self.assertIsNotNone(extended_pulse._total_phases)
        self.assertIsNotNone(extended_pulse._R)
        self.assertIsNotNone(extended_pulse._F)

        # Get filter function for both
        pulse_2.cache_filter_function(omega)
        extended_pulse = ff.extend([(pulse_1, 0), (pulse_2, 1)])
        self.assertIsNotNone(extended_pulse._total_Q_liouville)
        self.assertIsNotNone(extended_pulse._total_phases)
        self.assertIsNotNone(extended_pulse._R)
        self.assertIsNotNone(extended_pulse._F)

        # override
        extended_pulse = ff.extend([(pulse_1, 0), (pulse_2, 1)],
                                   cache_filter_function=False)
        self.assertIsNone(extended_pulse._total_Q_liouville)
        self.assertIsNone(extended_pulse._total_phases)
        self.assertIsNone(extended_pulse._R)
        self.assertIsNone(extended_pulse._F)

    def test_accuracy(self):
        ID, X, Y, Z = P_np
        XI = tensor(X, ID)
        IX = tensor(ID, X)
        XII = tensor(X, ID, ID)
        IXI = tensor(ID, X, ID)
        IIX = tensor(ID, ID, X)
        XIII = tensor(X, ID, ID, ID)
        IXII = tensor(ID, X, ID, ID)
        IIXI = tensor(ID, ID, X, ID)
        IIIX = tensor(ID, ID, ID, X)
        YI = tensor(Y, ID)
        IY = tensor(ID, Y)
        YII = tensor(Y, ID, ID)
        IYI = tensor(ID, Y, ID)
        IIY = tensor(ID, ID, Y)
        YIII = tensor(Y, ID, ID, ID)
        IYII = tensor(ID, Y, ID, ID)
        IIYI = tensor(ID, ID, Y, ID)
        IIIY = tensor(ID, ID, ID, Y)
        ZI = tensor(Z, ID)
        IZ = tensor(ID, Z)
        ZII = tensor(Z, ID, ID)
        IZI = tensor(ID, Z, ID)
        ZIII = tensor(Z, ID, ID, ID)
        IZII = tensor(ID, Z, ID, ID)
        IIZI = tensor(ID, ID, Z, ID)
        IIIZ = tensor(ID, ID, ID, Z)

        IIZ = tensor(ID, ID, Z)
        XXX = tensor(X, X, X)

        n_dt = 10
        coeffs = np.random.randn(3, n_dt)
        X_pulse = ff.PulseSequence(
            [[X, coeffs[0], 'X']],
            list(zip((X, Y, Z), np.ones((3, n_dt)), ('X', 'Y', 'Z'))),
            np.ones(n_dt),
            basis=ff.Basis.pauli(1)
        )
        Y_pulse = ff.PulseSequence(
            [[Y, coeffs[1], 'Y']],
            list(zip((X, Y, Z), np.ones((3, n_dt)), ('X', 'Y', 'Z'))),
            np.ones(n_dt),
            basis=ff.Basis.pauli(1)
        )
        Z_pulse = ff.PulseSequence(
            [[Z, coeffs[2], 'Z']],
            list(zip((X, Y, Z), np.ones((3, n_dt)), ('X', 'Y', 'Z'))),
            np.ones(n_dt),
            basis=ff.Basis.pauli(1)
        )
        XZ_pulse = ff.PulseSequence(
            [[XI, coeffs[0], 'XI'],
             [IZ, coeffs[2], 'IZ']],
            list(zip((XI, YI, ZI, IX, IY, IZ), np.ones((6, n_dt)),
                     ('XI', 'YI', 'ZI', 'IX', 'IY', 'IZ'))),
            np.ones(n_dt),
            basis=ff.Basis.pauli(2)
        )
        XYZ_pulse = ff.PulseSequence(
            [[XII, coeffs[0], 'XII'],
             [IYI, coeffs[1], 'IYI'],
             [IIZ, coeffs[2], 'IIZ']],
            list(zip((XII, YII, ZII, IIX, IIY, IIZ, IXI, IYI, IZI, XXX),
                     np.ones((10, n_dt)),
                     ('XII', 'YII', 'ZII', 'IIX', 'IIY', 'IIZ', 'IXI', 'IYI',
                      'IZI', 'XXX'))),
            np.ones(n_dt),
            basis=ff.Basis.pauli(3)
        )
        ZYX_pulse = ff.PulseSequence(
            [[IIX, coeffs[0], 'IIX'],
             [IYI, coeffs[1], 'IYI'],
             [ZII, coeffs[2], 'ZII']],
            list(zip((IIX, IIY, IIZ, XII, YII, ZII, IXI, IYI, IZI),
                     np.ones((9, n_dt)),
                     ('IIX', 'IIY', 'IIZ', 'XII', 'YII', 'ZII', 'IXI', 'IYI',
                      'IZI'))),
            np.ones(n_dt),
            basis=ff.Basis.pauli(3)
        )
        XZXZ_pulse = ff.PulseSequence(
            [[XIII, coeffs[0], 'XIII'],
             [IZII, coeffs[2], 'IZII'],
             [IIXI, coeffs[0], 'IIXI'],
             [IIIZ, coeffs[2], 'IIIZ']],
            list(zip((XIII, YIII, ZIII, IXII, IYII, IZII,
                      IIXI, IIYI, IIZI, IIIX, IIIY, IIIZ),
                     np.ones((12, n_dt)),
                     ('XIII', 'YIII', 'ZIII', 'IXII', 'IYII', 'IZII',
                      'IIXI', 'IIYI', 'IIZI', 'IIIX', 'IIIY', 'IIIZ'))),
            np.ones(n_dt),
            basis=ff.Basis.pauli(4)
        )
        XXZZ_pulse = ff.PulseSequence(
            [[XIII, coeffs[0], 'XIII'],
             [IIZI, coeffs[2], 'IIZI'],
             [IXII, coeffs[0], 'IXII'],
             [IIIZ, coeffs[2], 'IIIZ']],
            list(zip((XIII, YIII, ZIII, IIXI, IIYI, IIZI,
                      IXII, IYII, IZII, IIIX, IIIY, IIIZ),
                     np.ones((12, n_dt)),
                     ('XIII', 'YIII', 'ZIII', 'IIXI', 'IIYI', 'IIZI',
                      'IXII', 'IYII', 'IZII', 'IIIX', 'IIIY', 'IIIZ'))),
            np.ones(n_dt),
            basis=ff.Basis.pauli(4)
        )

        # Cache
        omega = ff.util.get_sample_frequencies(XYZ_pulse, n_samples=50)
        X_pulse.cache_filter_function(omega)
        Y_pulse.cache_filter_function(omega)
        Z_pulse.cache_filter_function(omega)
        XZ_pulse.cache_filter_function(omega)
        XYZ_pulse.cache_filter_function(omega)
        ZYX_pulse.cache_filter_function(omega)
        XZXZ_pulse.cache_filter_function(omega)
        XXZZ_pulse.cache_filter_function(omega)

        # Test that mapping a pulse to itself returns the pulse itself and
        # issues a warning
        with self.assertWarns(UserWarning):
            pulse = ff.extend([(X_pulse, 0)])
            self.assertIs(pulse, X_pulse)

        with self.assertWarns(UserWarning):
            pulse = ff.extend([(XZ_pulse, (0, 1))])
            self.assertIs(pulse, XZ_pulse)

        # Test mapping two single-qubit pulses to a two-qubit pulse
        XZ_pulse_ext = ff.extend([
            (X_pulse, 0, {'X': 'XI', 'Y': 'YI', 'Z': 'ZI'}),
            (Z_pulse, 1, {'X': 'IX', 'Y': 'IY', 'Z': 'IZ'})
        ])

        self.assertEqual(XZ_pulse, XZ_pulse_ext)
        self.assertCorrectDiagonalization(XZ_pulse_ext, atol=1e-14)
        self.assertArrayAlmostEqual(XZ_pulse._Q, XZ_pulse_ext._Q, atol=1e-10)
        self.assertArrayAlmostEqual(XZ_pulse._R, XZ_pulse_ext._R, atol=1e-9)
        self.assertArrayAlmostEqual(XZ_pulse._F, XZ_pulse_ext._F, atol=1e-9)

        # Test additional noise Hamiltonian
        add_H_n = list(zip((XXX,), np.ones((1, n_dt)), ['XXX']))
        XYZ_pulse_ext = ff.extend(
            [(XZ_pulse, (0, 2),
              {i: i[0] + 'I' + i[1] for i in XZ_pulse.n_oper_identifiers}),
             (Y_pulse, 1,
              {i: 'I' + i[0] + 'I' for i in Y_pulse.n_oper_identifiers})],
            additional_noise_Hamiltonian=add_H_n
        )

        self.assertEqual(XYZ_pulse, XYZ_pulse_ext)
        self.assertCorrectDiagonalization(XYZ_pulse_ext, atol=1e-14)
        self.assertArrayAlmostEqual(XYZ_pulse._Q, XYZ_pulse_ext._Q, atol=1e-10)
        self.assertArrayAlmostEqual(XYZ_pulse._R, XYZ_pulse_ext._R, atol=1e-9)
        self.assertArrayAlmostEqual(XYZ_pulse._F, XYZ_pulse_ext._F, atol=1e-9)

        # Test remapping a two-qubit pulse
        ZYX_pulse_ext = ff.extend(
            [(XZ_pulse, (2, 0),
              {i: i[1] + 'I' + i[0] for i in XZ_pulse.n_oper_identifiers}),
             (Y_pulse, 1,
              {i: 'I' + i[0] + 'I' for i in Y_pulse.n_oper_identifiers})],
        )

        self.assertEqual(ZYX_pulse, ZYX_pulse_ext)
        self.assertCorrectDiagonalization(ZYX_pulse_ext, atol=1e-14)
        self.assertArrayAlmostEqual(ZYX_pulse._Q, ZYX_pulse_ext._Q, atol=1e-10)
        self.assertArrayAlmostEqual(ZYX_pulse._R, ZYX_pulse_ext._R, atol=1e-9)
        self.assertArrayAlmostEqual(ZYX_pulse._F, ZYX_pulse_ext._F, atol=1e-9)

        XZXZ_pulse_ext = ff.extend([
            (XZ_pulse, (0, 1),
             {i: i + 'II' for i in XZ_pulse.n_oper_identifiers}),
            (XZ_pulse, (2, 3),
             {i: 'II' + i for i in XZ_pulse.n_oper_identifiers})
        ], cache_diagonalization=True)
        self.assertEqual(XZXZ_pulse, XZXZ_pulse_ext)
        self.assertCorrectDiagonalization(XZXZ_pulse_ext, atol=1e-14)
        self.assertArrayAlmostEqual(XZXZ_pulse._Q, XZXZ_pulse_ext._Q,
                                    atol=1e-10)
        self.assertArrayAlmostEqual(XZXZ_pulse._R, XZXZ_pulse_ext._R,
                                    atol=1e-9)
        self.assertArrayAlmostEqual(XZXZ_pulse._F, XZXZ_pulse_ext._F,
                                    atol=1e-8)

        XZXZ_pulse_ext = ff.extend([
            (XZ_pulse, (0, 1),
             {i: i + 'II' for i in XZ_pulse.n_oper_identifiers}),
            (XZ_pulse, (2, 3),
             {i: 'II' + i for i in XZ_pulse.n_oper_identifiers})
        ], cache_diagonalization=False)
        self.assertEqual(XZXZ_pulse, XZXZ_pulse_ext)
        self.assertArrayAlmostEqual(XZXZ_pulse._total_Q,
                                    XZXZ_pulse_ext._total_Q,
                                    atol=1e-10)

        # Test merging with overlapping qubit ranges
        XXZZ_pulse_ext = ff.extend([
            (XZ_pulse, (0, 2),
             {i: i[0] + 'I' + i[1] + 'I'
              for i in XZ_pulse.n_oper_identifiers}),
            (XZ_pulse, (1, 3),
             {i: 'I' + i[0] + 'I' + i[1]
              for i in XZ_pulse.n_oper_identifiers}),
        ])
        self.assertEqual(XXZZ_pulse, XXZZ_pulse_ext)
        self.assertCorrectDiagonalization(XXZZ_pulse_ext, atol=1e-14)
        self.assertArrayAlmostEqual(XXZZ_pulse._Q, XXZZ_pulse_ext._Q,
                                    atol=1e-10)
        self.assertArrayAlmostEqual(XXZZ_pulse._R, XXZZ_pulse_ext._R,
                                    atol=1e-10)
        self.assertArrayAlmostEqual(XXZZ_pulse._F, XXZZ_pulse_ext._F,
                                    atol=1e-8)

    def test_exceptions(self):
        ID, X, Y, Z = P_np
        n_dt = 10
        coeffs = randn(3, n_dt)
        omega = np.linspace(0, 1, 50)
        pulse_1 = ff.PulseSequence(
            list(zip((X, Y, Z), coeffs)),
            list(zip((X, Y, Z), np.ones((3, n_dt)))),
            np.ones(n_dt), basis=ff.Basis.pauli(1)
        )
        pulse_1.cache_filter_function(omega)
        pulse_2 = ff.PulseSequence(
            list(zip((X, Y, Z), coeffs)),
            list(zip((X, Y, Z), np.ones((3, n_dt)))),
            np.ones(n_dt)*2
        )
        pulse_11 = ff.extend([[pulse_1, 0], [pulse_1, 1]])
        pulse_11.cache_filter_function(omega+1)

        with self.assertRaises(ValueError):
            # qubit indices don't match on pulse that is remapped
            ff.extend([(pulse_11, (2, 1, 0))])

        with self.assertRaises(ValueError):
            # wrong dimensions
            ff.extend([(pulse_1, (0, 1))])

        with self.assertRaises(ValueError):
            # wrong dimensions
            ff.extend([(pulse_1, (0,))], d_per_qubit=3)

        with self.assertRaises(ValueError):
            # wrong dimensions
            ff.extend([(pulse_11, (0,))])

        with self.assertRaises(ValueError):
            # different dt
            ff.extend([(pulse_1, 0), [pulse_2, 1]])

        with self.assertRaises(ValueError):
            # Multiple pulses mapped to same qubit
            ff.extend([(pulse_1, 0), [pulse_1, 0]])

        with self.assertRaises(ValueError):
            # N < max(qubits)
            ff.extend([(pulse_1, 2)], N=2)

        with self.assertRaises(ValueError):
            # cache_filter_function == True and unequal omega
            ff.extend([(pulse_1, 0), (pulse_11, (1, 2))],
                      cache_filter_function=True, omega=None)

        with self.assertRaises(ValueError):
            # cache_diagonalization == False and additional_noise_Hamiltonian
            # is not None
            additional_noise_Hamiltonian = [[tensor(X, X), np.ones(n_dt)]]
            ff.extend(
                [(pulse_1, 0), (pulse_1, 1)], cache_diagonalization=False,
                additional_noise_Hamiltonian=additional_noise_Hamiltonian
            )

        with self.assertRaises(ValueError):
            # additional noise Hamiltonian defines existing identifier
            additional_noise_Hamiltonian = [
                [tensor(X, X), np.ones(n_dt), 'foo'],
                [tensor(X, X), np.ones(n_dt), 'foo'],
            ]
            ff.extend(
                [(pulse_1, 0), (pulse_1, 1)],
                additional_noise_Hamiltonian=additional_noise_Hamiltonian
            )

        with self.assertRaises(ValueError):
            # additional_noise_Hamiltonian has wrong dimensions
            additional_noise_Hamiltonian = [[tensor(X, X, X), np.ones(n_dt)]]
            ff.extend(
                [(pulse_1, 0), (pulse_1, 1)],
                additional_noise_Hamiltonian=additional_noise_Hamiltonian
            )

        with self.assertWarns(UserWarning):
            # Non-pauli basis
            ff.extend([(pulse_2, 0)])


class RemappingTest(testutil.TestCase):

    def test_caching(self):
        I, X, Y, Z = P_np
        XY_pulse = ff.PulseSequence(
            [[tensor(X, Y), [np.pi/2]]],
            [[tensor(X, I), [1]],
             [tensor(Y, I), [1]],
             [tensor(I, X), [1]],
             [tensor(I, Y), [1]]],
            [1],
            ff.Basis.pauli(2)
        )
        GGM_XY_pulse = ff.PulseSequence(
            [[tensor(X, Y), [np.pi/2]]],
            [[tensor(X, I), [1]],
             [tensor(Y, I), [1]],
             [tensor(I, X), [1]],
             [tensor(I, Y), [1]]],
            [1],
            ff.Basis.ggm(4)
        )
        attrs = ('omega', 'HD', 'HV', 'Q', 'total_phases', 'total_Q', 'F',
                 'total_Q_liouville', 'R')

        XY_pulse.cleanup('all')
        remapped_XY_pulse = ff.remap(XY_pulse, (1, 0))
        for attr in attrs:
            self.assertEqual(XY_pulse.is_cached(attr),
                             remapped_XY_pulse.is_cached(attr))

        omega = get_sample_frequencies(XY_pulse, n_samples=50)
        XY_pulse.cache_filter_function(omega)
        remapped_XY_pulse = ff.remap(XY_pulse, (1, 0))
        for attr in attrs:
            self.assertEqual(XY_pulse.is_cached(attr),
                             remapped_XY_pulse.is_cached(attr))

        GGM_XY_pulse.cleanup('all')
        remapped_GGM_XY_pulse = ff.remap(GGM_XY_pulse, (1, 0))
        for attr in attrs:
            self.assertEqual(GGM_XY_pulse.is_cached(attr),
                             remapped_GGM_XY_pulse.is_cached(attr))

        omega = get_sample_frequencies(GGM_XY_pulse, n_samples=50)
        GGM_XY_pulse.cache_filter_function(omega)
        with self.assertWarns(UserWarning):
            remapped_GGM_XY_pulse = ff.remap(GGM_XY_pulse, (1, 0))

        for attr in attrs[:-2]:
            self.assertEqual(GGM_XY_pulse.is_cached(attr),
                             remapped_GGM_XY_pulse.is_cached(attr))

        for attr in attrs[-2:]:
            self.assertFalse(remapped_GGM_XY_pulse.is_cached(attr))

    def test_accuracy(self):
        paulis = np.array(P_np)
        I, X, Y, Z = paulis
        amps = randn(randint(1, 11))
        pulse = ff.PulseSequence(
            [[tensor(X, Y, Z), amps]],
            [[tensor(X, I, I), np.ones_like(amps), 'XII'],
             [tensor(I, X, I), np.ones_like(amps), 'IXI'],
             [tensor(I, I, X), np.ones_like(amps), 'IIX']],
            np.ones_like(amps),
            ff.Basis.pauli(3)
        )
        omega = get_sample_frequencies(pulse, 50)
        pulse.cache_filter_function(omega)

        for _ in range(100):
            order = np.random.permutation(range(3))
            reordered_pulse = ff.PulseSequence(
                [[tensor(*paulis[1:][order]), amps]],
                [[tensor(*paulis[[1, 0, 0]][order]), np.ones_like(amps),
                  (''.join(['XII'[o] for o in order]))],
                 [tensor(*paulis[[0, 1, 0]][order]), np.ones_like(amps),
                  (''.join(['IXI'[o] for o in order]))],
                 [tensor(*paulis[[0, 0, 1]][order]), np.ones_like(amps),
                  (''.join(['IIX'[o] for o in order]))]],
                np.ones_like(amps),
                ff.Basis.pauli(3)
            )
            reordered_pulse.cache_filter_function(omega)

            remapped_pulse = ff.remap(
                pulse, order,
                oper_identifier_mapping={
                    'A_0': 'A_0',
                    'XII': ''.join(['XII'[o] for o in order]),
                    'IXI': ''.join(['IXI'[o] for o in order]),
                    'IIX': ''.join(['IIX'[o] for o in order])
                }
            )

            self.assertEqual(reordered_pulse, remapped_pulse)
            self.assertArrayAlmostEqual(reordered_pulse.t, remapped_pulse.t)
            self.assertEqual(reordered_pulse.d, remapped_pulse.d)
            self.assertEqual(reordered_pulse.basis, remapped_pulse.basis)
            self.assertArrayAlmostEqual(reordered_pulse._omega,
                                        remapped_pulse.omega)
            self.assertArrayAlmostEqual(reordered_pulse._Q, remapped_pulse._Q,
                                        atol=1e-14)
            self.assertArrayAlmostEqual(reordered_pulse._total_Q,
                                        remapped_pulse._total_Q,
                                        atol=1e-14)
            self.assertArrayAlmostEqual(reordered_pulse._total_Q_liouville,
                                        remapped_pulse._total_Q_liouville,
                                        atol=1e-14)
            self.assertArrayAlmostEqual(reordered_pulse._total_phases,
                                        remapped_pulse._total_phases)
            self.assertArrayAlmostEqual(reordered_pulse._R, remapped_pulse._R,
                                        atol=1e-12)
            self.assertArrayAlmostEqual(reordered_pulse._F, remapped_pulse._F,
                                        atol=1e-12)

            # Test the eigenvalues and -vectors by the characteristic equation
            self.assertCorrectDiagonalization(remapped_pulse, atol=1e-14)

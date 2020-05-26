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
This module tests the core functionality of the package.
"""
import string
from copy import copy
from random import sample

import numpy as np

import filter_functions as ff
from filter_functions import numeric, util
from tests import testutil


class CoreTest(testutil.TestCase):

    def test_pulse_sequence_constructor(self):
        """Test constructing a PulseSequence."""
        base_pulse = testutil.rand_pulse_sequence(2, 5, 3, 3)
        H_c = list(zip(base_pulse.c_opers, base_pulse.c_coeffs,
                       base_pulse.c_oper_identifiers))
        H_n = list(zip(base_pulse.n_opers, base_pulse.n_coeffs,
                       base_pulse.n_oper_identifiers))
        dt = base_pulse.dt

        for i in range(3):
            H_c[i] = list(H_c[i])
            H_n[i] = list(H_n[i])

        with self.assertRaises(TypeError):
            # Not enough positional arguments
            ff.PulseSequence(H_c, H_n)

        with self.assertRaises(TypeError):
            # dt not a sequence
            ff.PulseSequence(H_c, H_n, dt[0])

        idx = testutil.rng.randint(0, 5)
        with self.assertRaises(ValueError):
            # negative dt
            dt[idx] *= -1
            ff.PulseSequence(H_c, H_n, dt)

        dt[idx] *= -1
        with self.assertRaises(ValueError):
            # imagniary dt
            dt = dt.astype(complex)
            dt[idx] += 1j
            ff.PulseSequence(H_c, H_n, dt)

        dt = dt.real
        basis = ff.Basis.pauli(1)
        with self.assertRaises(ValueError):
            # basis not Basis instance
            ff.PulseSequence(H_c, H_n, dt, basis.view(np.ndarray))

        with self.assertRaises(ValueError):
            # basis not square
            ff.PulseSequence(H_c, H_n, dt, basis.reshape(4, 1, 4))

        with self.assertRaises(TypeError):
            # Control Hamiltonian not list or tuple
            ff.PulseSequence(np.array(H_c), H_n, dt)

        with self.assertRaises(TypeError):
            # Noise Hamiltonian not list or tuple
            ff.PulseSequence(H_c, np.array(H_n), dt)

        idx = testutil.rng.randint(0, 3)
        with self.assertRaises(TypeError):
            # Control Hamiltonian element not list or tuple
            H_c[idx] = dict(H_c[idx])
            ff.PulseSequence(H_c, H_n, dt)

        H_c[idx] = list(H_c[idx])
        with self.assertRaises(TypeError):
            # Noise Hamiltonian element not list or tuple
            H_n[idx] = dict(H_n[idx])
            ff.PulseSequence(H_c, H_n, dt)

        H_n[idx] = list(H_n[idx])
        with self.assertRaises(TypeError):
            # Control operators wrong type
            oper = H_c[idx][0].copy()
            H_c[idx][0] = dict(H_c[idx][0])
            ff.PulseSequence(H_c, H_n, dt)

        H_c[idx][0] = oper
        with self.assertRaises(TypeError):
            # Noise operators wrong type
            oper = H_n[idx][0].copy()
            H_n[idx][0] = dict(H_n[idx][0])
            ff.PulseSequence(H_c, H_n, dt)

        H_n[idx][0] = oper
        with self.assertRaises(TypeError):
            # Control coefficients wrong type
            coeff = H_c[idx][1].copy()
            H_c[idx][1] = H_c[idx][1][0]
            ff.PulseSequence(H_c, H_n, dt)

        H_c[idx][1] = coeff
        with self.assertRaises(TypeError):
            # Noise coefficients wrong type
            coeff = H_n[idx][1].copy()
            H_n[idx][1] = H_n[idx][1][0]
            ff.PulseSequence(H_c, H_n, dt)

        H_n[idx][1] = coeff
        with self.assertRaises(TypeError):
            # Control operators weird dimensions
            H_c[idx][0] = H_c[idx][0][:, :, None]
            ff.PulseSequence(H_c, H_n, dt)

        H_c[idx][0] = H_c[idx][0].squeeze()
        with self.assertRaises(TypeError):
            # Noise operators weird dimensions
            H_n[idx][0] = H_n[idx][0][:, :, None]
            ff.PulseSequence(H_c, H_n, dt)

        H_n[idx][0] = H_n[idx][0].squeeze()
        with self.assertRaises(ValueError):
            # Control operators not 2d
            for hc in H_c:
                hc[0] = hc[0][:, :, None]
            ff.PulseSequence(H_c, H_n, dt)

        for hc in H_c:
            hc[0] = hc[0].squeeze()
        with self.assertRaises(ValueError):
            # Noise operators not 2d
            for hn in H_n:
                hn[0] = hn[0][:, :, None]
            ff.PulseSequence(H_c, H_n, dt)

        for hn in H_n:
            hn[0] = hn[0].squeeze()
        with self.assertRaises(ValueError):
            # Control operators not square
            for hc in H_c:
                hc[0] = hc[0].reshape(1, 4)
            ff.PulseSequence(H_c, H_n, dt)

        for hc in H_c:
            hc[0] = hc[0].reshape(2, 2)
        with self.assertRaises(ValueError):
            # Noise operators not square
            for hn in H_n:
                hn[0] = hn[0].reshape(1, 4)
            ff.PulseSequence(H_c, H_n, dt)

        for hn in H_n:
            hn[0] = hn[0].reshape(2, 2)
        with self.assertRaises(ValueError):
            # Control identifiers not unique
            identifier = H_c[idx][2]
            H_c[idx][2] = H_c[idx-1][2]
            ff.PulseSequence(H_c, H_n, dt)

        H_c[idx][2] = identifier
        with self.assertRaises(ValueError):
            # Noise identifiers not unique
            identifier = H_n[idx][2]
            H_n[idx][2] = H_n[idx-1][2]
            ff.PulseSequence(H_c, H_n, dt)

        H_n[idx][2] = identifier
        coeffs = []
        with self.assertRaises(ValueError):
            # Control coefficients not same length as dt
            for hc in H_c:
                coeffs.append(hc[1][-2:])
                hc[1] = hc[1][:-2]
            ff.PulseSequence(H_c, H_n, dt)

        for i, c in enumerate(coeffs):
            H_c[i][1] = np.concatenate((H_c[i][1], c))
        with self.assertRaises(ValueError):
            # Noise coefficients not same length as dt
            for hn in H_n:
                hn[1] = hn[1][:-2]
            ff.PulseSequence(H_c, H_n, dt)

        for i, c in enumerate(coeffs):
            H_n[i][1] = np.concatenate((H_n[i][1], c))
        pulse = ff.PulseSequence(H_c, H_n, dt)
        # Hit __str__ and __repr__ methods
        pulse
        print(pulse)

        # Hit __copy__ method
        _ = copy(pulse)

        # Fewer identifiers than opers
        pulse_2 = ff.PulseSequence(
            [[util.P_np[1], [1], 'X'],
             [util.P_np[2], [1]]],
            [[util.P_np[1], [1]],
             [util.P_np[2], [1], 'Y']],
            [1]
        )
        self.assertArrayEqual(pulse_2.c_oper_identifiers, ('A_1', 'X'))
        self.assertArrayEqual(pulse_2.n_oper_identifiers, ('B_0', 'Y'))

    def test_pulse_sequence_attributes(self):
        """Test attributes of single instance"""
        X, Y, Z = util.P_np[1:]
        n_dt = testutil.rng.randint(1, 10)

        # trivial case
        A = ff.PulseSequence([[X, testutil.rng.randn(n_dt), 'X']],
                             [[Z, testutil.rng.randn(n_dt), 'Z']],
                             np.abs(testutil.rng.randn(n_dt)))
        self.assertFalse(A == 1)
        self.assertTrue(A != 1)

        # different number of time steps
        B = ff.PulseSequence([[X, testutil.rng.randn(n_dt+1), 'X']],
                             [[Z, testutil.rng.randn(n_dt+1), 'Z']],
                             np.abs(testutil.rng.randn(n_dt+1)))
        self.assertFalse(A == B)
        self.assertTrue(A != B)

        # different time steps
        B = ff.PulseSequence(
            list(zip(A.c_opers, A.c_coeffs, A.c_oper_identifiers)),
            list(zip(A.n_opers, A.n_coeffs, A.n_oper_identifiers)),
            np.abs(testutil.rng.randn(n_dt))
        )
        self.assertFalse(A == B)
        self.assertTrue(A != B)

        # different control opers
        B = ff.PulseSequence(
            list(zip([Y], A.c_coeffs, A.c_oper_identifiers)),
            list(zip(A.n_opers, A.n_coeffs, A.n_oper_identifiers)),
            A.dt
        )
        self.assertFalse(A == B)
        self.assertTrue(A != B)

        # different control coeffs
        B = ff.PulseSequence(
            list(zip(A.c_opers, [testutil.rng.randn(n_dt)],
                     A.c_oper_identifiers)),
            list(zip(A.n_opers, A.n_coeffs, A.n_oper_identifiers)),
            A.dt
        )
        self.assertFalse(A == B)
        self.assertTrue(A != B)

        # different noise opers
        B = ff.PulseSequence(
            list(zip(A.c_opers, A.c_coeffs, A.c_oper_identifiers)),
            list(zip([Y], A.n_coeffs, A.n_oper_identifiers)),
            A.dt
        )
        self.assertFalse(A == B)
        self.assertTrue(A != B)

        # different noise coeffs
        B = ff.PulseSequence(
            list(zip(A.c_opers, A.c_coeffs, A.c_oper_identifiers)),
            list(zip(A.n_opers, [testutil.rng.randn(n_dt)],
                     A.n_oper_identifiers)),
            A.dt
        )
        self.assertFalse(A == B)
        self.assertTrue(A != B)

        # different control oper identifiers
        B = ff.PulseSequence(
            list(zip(A.c_opers, A.c_coeffs, ['foobar'])),
            list(zip(A.n_opers, A.n_coeffs, A.n_oper_identifiers)),
            A.dt
        )
        self.assertFalse(A == B)
        self.assertTrue(A != B)

        # different noise oper identifiers
        B = ff.PulseSequence(
            list(zip(A.c_opers, A.c_coeffs, A.c_oper_identifiers)),
            list(zip(A.n_opers, A.n_coeffs, ['foobar'])),
            A.dt
        )
        self.assertFalse(A == B)
        self.assertTrue(A != B)

        # different bases
        elem = testutil.rand_herm_traceless(2)
        B = ff.PulseSequence(
            list(zip(A.c_opers, A.c_coeffs, A.c_oper_identifiers)),
            list(zip(A.n_opers, A.n_coeffs, A.n_oper_identifiers)),
            A.dt,
            ff.Basis(elem)
        )
        self.assertFalse(A == B)
        self.assertTrue(A != B)

        # Test for attributes
        for attr in A.__dict__.keys():
            if not (attr.startswith('_') or '_' + attr in A.__dict__.keys()):
                # not a cached attribute
                with self.assertRaises(AttributeError):
                    _ = A.is_cached(attr)
            else:
                # set mock attribute at random
                if testutil.rng.randint(0, 2):
                    setattr(A, attr, 'foo')
                    assertion = self.assertTrue
                else:
                    setattr(A, attr, None)
                    assertion = self.assertFalse

                assertion(A.is_cached(attr))

        aliases = {'eigenvalues': '_HD',
                   'eigenvectors': '_HV',
                   'propagators': '_Q',
                   'total propagator': '_total_Q',
                   'total propagator liouville': '_total_Q_liouville',
                   'frequencies': '_omega',
                   'total phases': '_total_phases',
                   'filter function': '_F',
                   'fidelity filter function': '_F',
                   'generalized filter function': '_F_kl',
                   'pulse correlation filter function': '_F_pc',
                   'fidelity pulse correlation filter function': '_F_pc',
                   'generalized pulse correlation filter function': '_F_pc_kl',
                   'control matrix': '_R',
                   'pulse correlation control matrix': '_R_pc'}

        for alias, attr in aliases.items():
            # set mock attribute at random
            if testutil.rng.randint(0, 2):
                setattr(A, attr, 'foo')
                assertion = self.assertTrue
            else:
                setattr(A, attr, None)
                assertion = self.assertFalse

            assertion(A.is_cached(alias))
            assertion(A.is_cached(alias.upper()))
            assertion(A.is_cached(alias.replace(' ', '_')))

        A.cleanup('all')

        # Test cleanup
        C = ff.concatenate((A, A), calc_pulse_correlation_ff=True,
                           which='generalized',
                           omega=util.get_sample_frequencies(A))
        C.diagonalize()
        attrs = ['_HD', '_HV', '_Q']
        for attr in attrs:
            self.assertIsNotNone(getattr(C, attr))

        C.cleanup()
        for attr in attrs:
            self.assertIsNone(getattr(C, attr))

        C.diagonalize()
        C.cache_control_matrix(A.omega)
        attrs.extend(['_R', '_total_phases', '_total_Q', '_total_Q_liouville'])
        for attr in attrs:
            self.assertIsNotNone(getattr(C, attr))

        C.cleanup('greedy')
        for attr in attrs:
            self.assertIsNone(getattr(C, attr))

        C.cache_filter_function(A.omega, which='generalized')
        for attr in attrs + ['omega', '_F_kl', '_F_pc_kl']:
            self.assertIsNotNone(getattr(C, attr))

        C = ff.concatenate((A, A), calc_pulse_correlation_ff=True,
                           which='fidelity', omega=A.omega)
        C.diagonalize()
        C.cache_filter_function(A.omega, which='fidelity')
        attrs.extend(['omega', '_F', '_F_pc'])
        for attr in attrs:
            self.assertIsNotNone(getattr(C, attr))

        C.cleanup('all')
        for attr in attrs + ['_F_kl', '_F_pc_kl']:
            self.assertIsNone(getattr(C, attr))

        C.cache_filter_function(A.omega, which='fidelity')
        C.cleanup('frequency dependent')
        freq_attrs = {'omega', '_R', '_F', '_F_kl', '_F_pc', '_F_pc_kl',
                      '_total_phases'}
        for attr in freq_attrs:
            self.assertIsNone(getattr(C, attr))

        for attr in set(attrs).difference(freq_attrs):
            self.assertIsNotNone(getattr(C, attr))

    def test_pulse_sequence_attributes_concat(self):
        """Test attributes of concatenated sequence."""
        X, Y, Z = util.P_np[1:]
        n_dt_1 = testutil.rng.randint(5, 11)
        x_coeff_1 = testutil.rng.randn(n_dt_1)
        z_coeff_1 = testutil.rng.randn(n_dt_1)
        dt_1 = np.abs(testutil.rng.randn(n_dt_1))
        n_dt_2 = testutil.rng.randint(5, 11)
        y_coeff_2 = testutil.rng.randn(n_dt_2)
        z_coeff_2 = testutil.rng.randn(n_dt_2)
        dt_2 = np.abs(testutil.rng.randn(n_dt_2))
        pulse_1 = ff.PulseSequence([[X, x_coeff_1]],
                                   [[Z, z_coeff_1]],
                                   dt_1)
        pulse_2 = ff.PulseSequence([[Y, y_coeff_2]],
                                   [[Z, z_coeff_2]],
                                   dt_2)
        pulse_3 = ff.PulseSequence([[Y, testutil.rng.randn(2)],
                                    [X, testutil.rng.randn(2)]],
                                   [[Z, np.abs(testutil.rng.randn(2))]],
                                   [1, 1])

        pulse_12 = pulse_1 @ pulse_2
        pulse_21 = pulse_2 @ pulse_1

        with self.assertRaises(TypeError):
            _ = pulse_1 @ testutil.rng.randn(2, 2)

        # Concatenate pulses with same operators but different labels
        with self.assertRaises(ValueError):
            pulse_1 @ pulse_3

        # Test nbytes property
        _ = pulse_1.nbytes

        self.assertArrayEqual(pulse_12.dt, [*dt_1, *dt_2])
        self.assertArrayEqual(pulse_21.dt, [*dt_2, *dt_1])

        self.assertArrayEqual(pulse_12.c_opers, [X, Y])
        self.assertArrayEqual(pulse_21.c_opers, [Y, X])

        self.assertArrayEqual(pulse_12.c_oper_identifiers, ['A_0_0', 'A_0_1'])
        self.assertArrayEqual(pulse_21.c_oper_identifiers, ['A_0_0', 'A_0_1'])

        self.assertArrayEqual(pulse_12.c_coeffs,
                              [[*x_coeff_1, *np.zeros(n_dt_2)],
                               [*np.zeros(n_dt_1), *y_coeff_2]])
        self.assertArrayEqual(pulse_21.c_coeffs,
                              [[*y_coeff_2, *np.zeros(n_dt_1)],
                               [*np.zeros(n_dt_2), *x_coeff_1]])

        self.assertArrayEqual(pulse_12.n_opers, [Z])
        self.assertArrayEqual(pulse_21.n_opers, [Z])

        self.assertArrayEqual(pulse_12.n_oper_identifiers, ['B_0'])
        self.assertArrayEqual(pulse_21.n_oper_identifiers, ['B_0'])

        self.assertArrayEqual(pulse_12.n_coeffs, [[*z_coeff_1, *z_coeff_2]])
        self.assertArrayEqual(pulse_21.n_coeffs, [[*z_coeff_2, *z_coeff_1]])

        omega = np.linspace(-100, 100, 101)
        pulses = (pulse_1, pulse_2, pulse_12, pulse_21)
        for pulse in pulses:
            self.assertIsNone(pulse._total_phases)
            self.assertIsNone(pulse._total_Q)
            self.assertIsNone(pulse._total_Q_liouville)

            total_phases = pulse.get_total_phases(omega)
            total_Q = pulse.total_Q
            total_Q_liouville = pulse.total_Q_liouville

            self.assertArrayEqual(total_phases, pulse._total_phases)
            self.assertArrayEqual(total_Q, pulse._total_Q)
            self.assertArrayEqual(total_Q_liouville, pulse._total_Q_liouville)

        # Test custom identifiers
        letters = testutil.rng.choice(list(string.ascii_letters), size=(6, 5),
                                      replace=False)
        ids = [''.join(l) for l in letters[:3]]
        labels = [''.join(l) for l in letters[3:]]
        pulse = ff.PulseSequence(
            list(zip([X, Y, Z], testutil.rng.randn(3, 2), ids, labels)),
            list(zip([X, Y, Z], testutil.rng.randn(3, 2), ids, labels)),
            [1, 1]
        )

        self.assertArrayEqual(pulse.c_oper_identifiers, sorted(ids))
        self.assertArrayEqual(pulse.n_oper_identifiers, sorted(ids))

    def test_filter_function(self):
        """Test the filter function calculation and related methods"""
        for d, n_dt in zip(testutil.rng.randint(2, 10, (3,)),
                           testutil.rng.randint(10, 200, (3,))):
            total_pulse = testutil.rand_pulse_sequence(d, n_dt, 4, 6)
            c_opers, c_coeffs = total_pulse.c_opers, total_pulse.c_coeffs
            n_opers, n_coeffs = total_pulse.n_opers, total_pulse.n_coeffs
            dt = total_pulse.dt

            total_HD, total_HV, _ = numeric.diagonalize(
                np.einsum('il,ijk->ljk', c_coeffs, c_opers), total_pulse.dt
            )
            omega = util.get_sample_frequencies(total_pulse, n_samples=100)
            # Try the progress bar
            R = total_pulse.get_control_matrix(omega, show_progressbar=True)

            # Check that some attributes are cached
            self.assertIsNotNone(total_pulse._total_phases)
            self.assertIsNotNone(total_pulse._total_Q)
            self.assertIsNotNone(total_pulse._total_Q_liouville)

            # Calculate everything 'on foot'
            pulses = [ff.PulseSequence(list(zip(c_opers, c_coeffs[:, i:i+1])),
                                       list(zip(n_opers, n_coeffs[:, i:i+1])),
                                       dt[i:i+1])
                      for i in range(n_dt)]

            phases = np.empty((n_dt, len(omega)), dtype=complex)
            L = np.empty((n_dt, d**2, d**2))
            R_l = np.empty((n_dt, 6, d**2, len(omega)), dtype=complex)
            for l, pulse in enumerate(pulses):
                phases[l] = np.exp(1j*total_pulse.t[l]*omega)
                L[l] = numeric.liouville_representation(total_pulse.Q[l],
                                                        total_pulse.basis)
                R_l[l] = pulse.get_control_matrix(omega)

            # Check that both methods of calculating the control are the same
            R_from_atomic = numeric.calculate_control_matrix_from_atomic(
                phases, R_l, L)
            R_from_scratch = numeric.calculate_control_matrix_from_scratch(
                HD=total_HD,
                HV=total_HV,
                Q=total_pulse.Q,
                omega=omega,
                basis=total_pulse.basis,
                n_opers=n_opers,
                n_coeffs=n_coeffs,
                dt=total_pulse.dt,
                t=total_pulse.t
            )
            self.assertArrayAlmostEqual(R, R_from_scratch)
            # first column (identity element) always zero but susceptible to
            # floating point error, increase atol
            self.assertArrayAlmostEqual(R_from_scratch, R_from_atomic,
                                        atol=1e-13)

            # Check if the filter functions for autocorrelated noise are real
            F = total_pulse.get_filter_function(omega)
            self.assertTrue(
                np.isreal(F[np.eye(len(n_opers), dtype=bool)]).all()
            )

    def test_pulse_correlation_filter_function(self):
        """
        Test calculation of pulse correlation filter function and control
        matrix.
        """
        X, Y, Z = util.P_np[1:]
        T = 1
        omega = np.linspace(-2e1, 2e1, 250)
        H_c, H_n, dt = dict(), dict(), dict()
        H_c['X'] = [[X, [np.pi/2/T]]]
        H_n['X'] = [[X, [1]],
                    [Y, [1]],
                    [Z, [1]]]
        dt['X'] = [T]
        H_c['Y'] = [[Y, [np.pi/4/T]]]
        H_n['Y'] = [[X, [1]],
                    [Y, [1]],
                    [Z, [1]]]
        dt['Y'] = [T]
        n_nops = 3

        # Check if an exception is raised if we want to calculate the PC-FF but
        # one pulse has different frequencies
        with self.assertRaises(ValueError):
            pulses = dict()
            for i, key in enumerate(('X', 'Y')):
                pulses[key] = ff.PulseSequence(H_c[key], H_n[key], dt[key])
                pulses[key].cache_filter_function(omega + i)

            ff.concatenate([pulses['X'], pulses['Y']],
                           calc_pulse_correlation_ff=True)

        # Get filter functions at same frequencies
        [pulse.cache_filter_function(omega) for pulse in pulses.values()]

        pulse_1 = pulses['X'] @ pulses['Y']
        pulse_2 = ff.concatenate([pulses['X'], pulses['Y']],
                                 calc_pulse_correlation_ff=True)
        pulse_3 = ff.concatenate([pulses['X'], pulses['Y']],
                                 calc_pulse_correlation_ff=True,
                                 which='generalized')

        # Check if the filter functions on the diagonals are real
        F = pulse_2.get_pulse_correlation_filter_function()
        diag_1 = np.eye(2, dtype=bool)
        diag_2 = np.eye(3, dtype=bool)
        self.assertTrue(np.isreal(F[diag_1][:, diag_2]).all())

        self.assertEqual(pulse_1, pulse_2)
        self.assertEqual(pulse_2.get_pulse_correlation_filter_function().shape,
                         (2, 2, n_nops, n_nops, len(omega)))
        self.assertArrayAlmostEqual(
            pulse_1.get_filter_function(omega),
            pulse_2.get_pulse_correlation_filter_function().sum((0, 1))
        )
        self.assertArrayAlmostEqual(pulse_1.get_filter_function(omega),
                                    pulse_2._F)

        # Test the behavior of the pulse correlation control matrix
        with self.assertRaises(ValueError):
            # R wrong dimension
            numeric.calculate_pulse_correlation_filter_function(pulse_1._R)

        with self.assertRaises(util.CalculationError):
            # not calculated
            pulse_1.get_pulse_correlation_control_matrix()

        R_pc = pulse_2.get_pulse_correlation_control_matrix()
        self.assertArrayEqual(
            F, numeric.calculate_pulse_correlation_filter_function(
                R_pc, 'fidelity'
            )
        )

        F = pulse_3.get_pulse_correlation_filter_function()
        R_pc = pulse_3.get_pulse_correlation_control_matrix()
        self.assertArrayEqual(
            F, numeric.calculate_pulse_correlation_filter_function(
                R_pc, 'fidelity'
            )
        )

        S = omega**0*1e-2
        with self.assertRaises(util.CalculationError):
            infid_1 = ff.infidelity(pulse_1, S, omega, which='correlations')

        with self.assertRaises(ValueError):
            infid_1 = ff.infidelity(pulse_1, S, omega, which='foobar')

        for _ in range(10):
            n_nops = testutil.rng.randint(1, 4)
            identifiers = sample(['B_0', 'B_1', 'B_2'], n_nops)

            infid_X = ff.infidelity(pulses['X'], S, omega, which='total',
                                    n_oper_identifiers=identifiers)
            infid_Y = ff.infidelity(pulses['Y'], S, omega, which='total',
                                    n_oper_identifiers=identifiers)
            infid_1 = ff.infidelity(pulse_1, S, omega, which='total',
                                    n_oper_identifiers=identifiers)
            infid_2 = ff.infidelity(pulse_2, S, omega, which='correlations',
                                    n_oper_identifiers=identifiers)

            self.assertAlmostEqual(infid_1.sum(), infid_2.sum())
            self.assertArrayAlmostEqual(infid_X, infid_2[0, 0])
            self.assertArrayAlmostEqual(infid_Y, infid_2[1, 1])

        # Test function for correlated noise spectra
        S = np.array([[1e-4/omega**2, 1e-4*np.exp(-omega**2)],
                      [1e-4*np.exp(-omega**2), 1e-4/omega**2]])
        infid_1 = ff.infidelity(pulse_1, S, omega, which='total',
                                n_oper_identifiers=['B_0', 'B_2'])
        infid_2 = ff.infidelity(pulse_2, S, omega, which='correlations',
                                n_oper_identifiers=['B_0', 'B_2'])

        self.assertAlmostEqual(infid_1.sum(), infid_2.sum())
        self.assertArrayAlmostEqual(infid_1, infid_2.sum(axis=(0, 1)))

    def test_calculate_error_vector_correlation_functions(self):
        """Test raises of numeric.error_transfer_matrix"""
        pulse = testutil.rand_pulse_sequence(2, 1, 1, 1)

        omega = testutil.rng.randn(43)
        # single spectrum
        S = testutil.rng.randn(78)
        for i in range(4):
            with self.assertRaises(ValueError):
                numeric.calculate_error_vector_correlation_functions(
                    pulse, np.tile(S, [1]*i), omega
                )

    def test_infidelity_convergence(self):
        import matplotlib
        matplotlib.use('Agg')

        omega = {
            'omega_IR': 0,
            'omega_UV': 2,
            'spacing': 'linear',
            'n_min': 10,
            'n_max': 50,
            'n_points': 4
        }

        def S(omega):
            return omega**0

        simple_pulse = testutil.rand_pulse_sequence(2, 1, 1, 1)
        complicated_pulse = testutil.rand_pulse_sequence(2, 100, 3, 3)

        with self.assertRaises(TypeError):
            n, infids, (fig, ax) = ff.infidelity(simple_pulse, S, [],
                                                 test_convergence=True)

        with self.assertRaises(TypeError):
            n, infids, (fig, ax) = ff.infidelity(simple_pulse, [1, 2, 3],
                                                 dict(spacing='foobar'),
                                                 test_convergence=True)

        with self.assertRaises(ValueError):
            n, infids, (fig, ax) = ff.infidelity(simple_pulse, S,
                                                 dict(spacing='foobar'),
                                                 test_convergence=True)

        # Test with default args
        n, infids, (fig, ax) = ff.infidelity(simple_pulse, S, {},
                                             test_convergence=True)

        # Test with non-default args
        identifiers = testutil.rng.choice(complicated_pulse.n_oper_identifiers,
                                          testutil.rng.randint(1, 4))

        n, infids, (fig, ax) = ff.infidelity(complicated_pulse,
                                             S, omega, test_convergence=True,
                                             n_oper_identifiers=identifiers)

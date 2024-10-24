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
import copy
import string
from random import sample

import numpy as np
import pytest
import sparse

import filter_functions as ff
from filter_functions import numeric, util
from tests import testutil
from tests.testutil import rng

from . import qutip


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

        idx = rng.integers(0, 5)
        with self.assertRaises(ValueError):
            # negative dt
            dt[idx] *= -1
            ff.PulseSequence(H_c, H_n, dt)

        dt[idx] *= -1
        with self.assertRaises(ValueError):
            # imaginary dt
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
            ff.PulseSequence(np.array(H_c, dtype=object), H_n, dt)

        with self.assertRaises(TypeError):
            # Noise Hamiltonian not list or tuple
            ff.PulseSequence(H_c, np.array(H_n, dtype=object), dt)

        with self.assertRaises(TypeError):
            # Element of control Hamiltonian not list or tuple
            ff.PulseSequence([np.array(H_c[0], dtype=object)], H_n, dt)

        with self.assertRaises(TypeError):
            # Element of noise Hamiltonian not list or tuple
            ff.PulseSequence(H_c, [np.array(H_n[0], dtype=object)], dt)

        idx = rng.integers(0, 3)
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
        with self.assertRaises(ValueError):
            # Control operators not 2d
            for hc in H_c:
                hc[0] = np.tile(hc[0], (rng.integers(2, 11), 1, 1))
            ff.PulseSequence(H_c, H_n, dt)

        for hc in H_c:
            hc[0] = hc[0][0]
        with self.assertRaises(ValueError):
            # Noise operators not 2d
            for hn in H_n:
                hn[0] = np.tile(hn[0], (rng.integers(2, 11), 1, 1))
            ff.PulseSequence(H_c, H_n, dt)

        for hn in H_n:
            hn[0] = hn[0][0]
        with self.assertRaises(ValueError):
            # Control operators not square
            for hc in H_c:
                hc[0] = np.tile(hc[0].reshape(1, 4), (2, 1))
            ff.PulseSequence(H_c, H_n, dt)

        for hc in H_c:
            hc[0] = hc[0][0].reshape(2, 2)
        with self.assertRaises(ValueError):
            # Noise operators not square
            for hn in H_n:
                hn[0] = np.tile(hn[0].reshape(1, 4), (2, 1))
            ff.PulseSequence(H_c, H_n, dt)

        for hn in H_n:
            hn[0] = hn[0][0].reshape(2, 2)
        with self.assertRaises(ValueError):
            # Control and noise operators not same dimension
            for hn in H_n:
                hn[0] = np.block([[hn[0], hn[0]], [hn[0], hn[0]]])
            ff.PulseSequence(H_c, H_n, dt)

        for hn in H_n:
            hn[0] = hn[0][:2, :2]
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

        # Fewer identifiers than opers
        pulse_2 = ff.PulseSequence(
            [[util.paulis[1], [1], 'X'],
             [util.paulis[2], [1]]],
            [[util.paulis[1], [1]],
             [util.paulis[2], [1], 'Y']],
            [1]
        )
        self.assertArrayEqual(pulse_2.c_oper_identifiers, ('A_1', 'X'))
        self.assertArrayEqual(pulse_2.n_oper_identifiers, ('B_0', 'Y'))

    def test_copy(self):
        pulse = testutil.rand_pulse_sequence(2, 2)
        old_copers = pulse.c_opers.copy()

        copied = copy.copy(pulse)
        deepcopied = copy.deepcopy(pulse)

        self.assertEqual(pulse, copied)
        self.assertEqual(pulse, deepcopied)

        pulse.c_opers[...] = rng.standard_normal(size=pulse.c_opers.shape)

        self.assertArrayEqual(pulse.c_opers, copied.c_opers)
        self.assertArrayEqual(old_copers, deepcopied.c_opers)

        self.assertEqual(pulse, copied)
        self.assertNotEqual(pulse, deepcopied)

    def test_pulse_sequence_attributes(self):
        """Test attributes of single instance"""
        X, Y, Z = util.paulis[1:]
        n_dt = rng.integers(1, 10)

        # trivial case
        A = ff.PulseSequence([[X, rng.standard_normal(n_dt), 'X']],
                             [[Z, rng.standard_normal(n_dt), 'Z']],
                             np.abs(rng.standard_normal(n_dt)))
        self.assertFalse(A == 1)
        self.assertTrue(A != 1)

        # different number of time steps
        B = ff.PulseSequence([[X, rng.standard_normal(n_dt+1), 'X']],
                             [[Z, rng.standard_normal(n_dt+1), 'Z']],
                             np.abs(rng.standard_normal(n_dt+1)))
        self.assertFalse(A == B)
        self.assertTrue(A != B)

        # different time steps
        B = ff.PulseSequence(
            list(zip(A.c_opers, A.c_coeffs, A.c_oper_identifiers)),
            list(zip(A.n_opers, A.n_coeffs, A.n_oper_identifiers)),
            np.abs(rng.standard_normal(n_dt))
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
            list(zip(A.c_opers, [rng.standard_normal(n_dt)],
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
            list(zip(A.n_opers, [rng.standard_normal(n_dt)],
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

        # Test sparse operators for whatever reason
        A = ff.PulseSequence([[util.paulis[1], [1]]],
                             [[sparse.COO.from_numpy(util.paulis[2]), [2]]],
                             [3])
        B = ff.PulseSequence([[sparse.COO.from_numpy(util.paulis[1]), [1]]],
                             [[util.paulis[2], [2]]],
                             [3])
        self.assertEqual(A, B)

        # Test for attributes
        for attr in A.__dict__.keys():
            if not (attr.startswith('_') or '_' + attr in A.__dict__.keys()):
                # not a cached attribute
                with self.assertRaises(AttributeError):
                    _ = A.is_cached(attr)
            else:
                # set mock attribute at random
                if rng.integers(0, 2):
                    setattr(A, attr, 'foo')
                    assertion = self.assertTrue
                else:
                    setattr(A, attr, None)
                    assertion = self.assertFalse

                assertion(A.is_cached(attr))

        # Diagonalization attributes
        A.diagonalize()
        self.assertIsNotNone(A.eigvals)
        self.assertIsNotNone(A.eigvecs)
        self.assertIsNotNone(A.propagators)

        A.cleanup('conservative')
        self.assertIsNotNone(A.eigvals)
        A.cleanup('conservative')
        self.assertIsNotNone(A.eigvecs)
        A.cleanup('conservative')
        self.assertIsNotNone(A.propagators)

        aliases = {'eigenvalues': '_eigvals',
                   'eigenvectors': '_eigvecs',
                   'total propagator': '_total_propagator',
                   'total propagator liouville': '_total_propagator_liouville',
                   'frequencies': '_omega',
                   'total phases': '_total_phases',
                   'filter function': '_filter_function',
                   'fidelity filter function': '_filter_function',
                   'generalized filter function': '_filter_function_gen',
                   'pulse correlation filter function': '_filter_function_pc',
                   'fidelity pulse correlation filter function': '_filter_function_pc',
                   'generalized pulse correlation filter function': '_filter_function_pc_gen',
                   'control matrix': '_control_matrix',
                   'pulse correlation control matrix': '_control_matrix_pc'}

        for alias, attr in aliases.items():
            # set mock attribute at random
            if rng.integers(0, 2):
                setattr(A, attr, 'foo')
                assertion = self.assertTrue
            else:
                setattr(A, attr, None)
                assertion = self.assertFalse

            assertion(A.is_cached(alias))
            assertion(A.is_cached(alias.upper()))
            assertion(A.is_cached(alias.replace(' ', '_')))

        A.cleanup('all')
        A._t = None
        A._tau = None

        # Test cleanup
        C = ff.concatenate((A, A), calc_pulse_correlation_FF=True,
                           which='generalized',
                           omega=util.get_sample_frequencies(A))
        C.diagonalize()
        attrs = ['_eigvals', '_eigvecs', '_propagators']
        for attr in attrs:
            self.assertIsNotNone(getattr(C, attr))

        C.cleanup()
        for attr in attrs:
            self.assertIsNone(getattr(C, attr))

        C.diagonalize()
        C.cache_control_matrix(A.omega)
        attrs.extend(['_control_matrix', '_total_phases', '_total_propagator',
                      '_total_propagator_liouville'])
        for attr in attrs:
            self.assertIsNotNone(getattr(C, attr))

        C.cleanup('greedy')
        for attr in attrs:
            self.assertIsNone(getattr(C, attr))

        C.cache_filter_function(A.omega, which='generalized')
        for attr in attrs + ['omega', '_filter_function_gen',
                             '_filter_function_pc_gen']:
            self.assertIsNotNone(getattr(C, attr))

        C = ff.concatenate((A, A), calc_pulse_correlation_FF=True,
                           which='fidelity', omega=A.omega)
        C.diagonalize()
        C.cache_filter_function(A.omega, which='fidelity')
        attrs.extend(['omega', '_filter_function', '_filter_function_pc'])
        for attr in attrs:
            self.assertIsNotNone(getattr(C, attr))

        C.cleanup('all')
        for attr in attrs + ['_filter_function_gen',
                             '_filter_function_pc_gen']:
            self.assertIsNone(getattr(C, attr))

        C.cache_filter_function(A.omega, which='fidelity')
        C.cleanup('frequency dependent')
        freq_attrs = {'omega', '_control_matrix', '_filter_function',
                      '_filter_function_gen', '_filter_function_pc',
                      '_filter_function_pc_gen', '_total_phases'}
        for attr in freq_attrs:
            self.assertIsNone(getattr(C, attr))

        for attr in set(attrs).difference(freq_attrs):
            self.assertIsNotNone(getattr(C, attr))

        # Test t, tau, and duration properties
        pulse = testutil.rand_pulse_sequence(2, 3, 1, 1)
        self.assertIs(pulse._t, None)
        self.assertIs(pulse._tau, None)
        self.assertArrayEqual(pulse.t, [0, *pulse.dt.cumsum()])
        self.assertEqual(pulse.tau, pulse.t[-1])
        self.assertEqual(pulse.duration, pulse.tau)

    def test_pulse_sequence_attributes_concat(self):
        """Test attributes of concatenated sequence."""
        X, Y, Z = util.paulis[1:]
        n_dt_1 = rng.integers(5, 11)
        x_coeff_1 = rng.standard_normal(n_dt_1)
        z_coeff_1 = rng.standard_normal(n_dt_1)
        dt_1 = np.abs(rng.standard_normal(n_dt_1))
        n_dt_2 = rng.integers(5, 11)
        y_coeff_2 = rng.standard_normal(n_dt_2)
        z_coeff_2 = rng.standard_normal(n_dt_2)
        dt_2 = np.abs(rng.standard_normal(n_dt_2))
        pulse_1 = ff.PulseSequence([[X, x_coeff_1]],
                                   [[Z, z_coeff_1]],
                                   dt_1)
        pulse_2 = ff.PulseSequence([[Y, y_coeff_2]],
                                   [[Z, z_coeff_2]],
                                   dt_2)
        pulse_3 = ff.PulseSequence([[Y, rng.standard_normal(2)],
                                    [X, rng.standard_normal(2)]],
                                   [[Z, np.abs(rng.standard_normal(2))]],
                                   [1, 1])
        pulse_4 = ff.PulseSequence([[Y, rng.standard_normal(2)],
                                    [X, rng.standard_normal(2)]],
                                   [[Z, np.ones(2)]],
                                   [1, 1])
        pulse_5 = ff.PulseSequence([[Y, np.zeros(5), 'A_0']],
                                   [[Y, np.zeros(5), 'B_1']],
                                   1 - rng.random(5))

        # Concatenate with different noise opers
        pulses = [testutil.rand_pulse_sequence(2, 1) for _ in range(2)]
        pulses[0].omega = np.arange(10)
        pulses[1].omega = np.arange(10)
        newpulse = ff.concatenate(pulses, calc_filter_function=True)
        self.assertTrue(newpulse.is_cached('filter function'))

        with self.assertRaises(TypeError):
            _ = pulse_1 @ rng.standard_normal((2, 2))

        # Concatenate pulses with same operators but different labels
        with self.assertRaises(ValueError):
            pulse_1 @ pulse_3

        # Test nbytes property
        _ = pulse_1.nbytes

        pulse_12 = pulse_1 @ pulse_2
        pulse_21 = pulse_2 @ pulse_1
        pulse_45 = pulse_4 @ pulse_5

        self.assertArrayEqual(pulse_12.dt, [*dt_1, *dt_2])
        self.assertArrayEqual(pulse_21.dt, [*dt_2, *dt_1])

        self.assertIs(pulse_12._t, None)
        self.assertIs(pulse_21._t, None)

        self.assertEqual(pulse_12._tau, pulse_1.tau + pulse_2.tau)
        self.assertEqual(pulse_21._tau, pulse_1.tau + pulse_2.tau)

        self.assertAlmostEqual(pulse_12.duration, pulse_1.duration + pulse_2.duration)
        self.assertAlmostEqual(pulse_21.duration, pulse_2.duration + pulse_1.duration)
        self.assertAlmostEqual(pulse_12.duration, pulse_21.duration)

        self.assertArrayAlmostEqual(pulse_12.t, [*pulse_1.t, *(pulse_2.t[1:] + pulse_1.tau)])
        self.assertArrayAlmostEqual(pulse_21.t, [*pulse_2.t, *(pulse_1.t[1:] + pulse_2.tau)])

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

        # Make sure zero coefficients are handled correctly
        self.assertFalse(np.any(np.isnan(pulse_45.c_coeffs)))
        self.assertFalse(np.any(np.isnan(pulse_45.n_coeffs)))
        self.assertArrayEqual(pulse_45.c_coeffs,
                              [[*pulse_4.c_coeffs[0], *np.zeros(5)],
                               [*pulse_4.c_coeffs[1], *np.zeros(5)]])
        self.assertArrayEqual(pulse_45.n_coeffs,
                              [[*pulse_4.n_coeffs[0], *[pulse_4.n_coeffs[0, 0]]*5],
                               [*[pulse_5.n_coeffs[0, 0]]*2, *pulse_5.n_coeffs[0]]])

        omega = np.linspace(-100, 100, 101)
        pulses = (pulse_1, pulse_2, pulse_12, pulse_21)
        for pulse in pulses:
            self.assertIsNone(pulse._total_phases)
            self.assertIsNone(pulse._total_propagator)
            self.assertIsNone(pulse._total_propagator_liouville)

            total_phases = pulse.get_total_phases(omega)
            total_propagator = pulse.total_propagator
            total_propagator_liouville = pulse.total_propagator_liouville

            self.assertArrayEqual(total_phases, pulse._total_phases)
            self.assertArrayEqual(total_propagator, pulse._total_propagator)
            self.assertArrayEqual(total_propagator_liouville,
                                  pulse._total_propagator_liouville)

        # Test custom identifiers
        letters = rng.choice(list(string.ascii_letters), size=(6, 5),
                             replace=False)
        ids = [''.join(c) for c in letters[:3]]
        labels = [''.join(c) for c in letters[3:]]
        pulse = ff.PulseSequence(
            list(zip([X, Y, Z], rng.standard_normal((3, 2)), ids, labels)),
            list(zip([X, Y, Z], rng.standard_normal((3, 2)), ids, labels)),
            [1, 1]
        )

        self.assertArrayEqual(pulse.c_oper_identifiers, sorted(ids))
        self.assertArrayEqual(pulse.n_oper_identifiers, sorted(ids))

        pulse = testutil.rand_pulse_sequence(2, 7, 1, 2)
        periodic_pulse = ff.concatenate_periodic(pulse, 7)

        self.assertIs(periodic_pulse._t, None)
        self.assertEqual(periodic_pulse._tau, pulse.tau * 7)
        self.assertArrayAlmostEqual(periodic_pulse.t, [0, *periodic_pulse.dt.cumsum()])

    def test_cache_intermediates_liouville(self):
        """Test caching of intermediate elements"""
        pulse = testutil.rand_pulse_sequence(3, 4, 2, 3)
        omega = util.get_sample_frequencies(pulse, 33, spacing='linear')
        ctrlmat = pulse.get_control_matrix(omega, cache_intermediates=True)
        filtfun = pulse.get_filter_function(omega, cache_intermediates=True)

        self.assertIsNotNone(pulse._intermediates)
        self.assertArrayAlmostEqual(pulse._intermediates['control_matrix_step'].sum(0), ctrlmat)
        self.assertArrayAlmostEqual(numeric.calculate_filter_function(ctrlmat), filtfun)
        self.assertArrayAlmostEqual(pulse._intermediates['n_opers_transformed'],
                                    numeric._transform_hamiltonian(pulse.eigvecs,
                                                                   pulse.n_opers,
                                                                   pulse.n_coeffs))
        eigvecs_prop = numeric._propagate_eigenvectors(pulse.propagators[:-1], pulse.eigvecs)
        basis_transformed = np.einsum('gba,kbc,gcd->gkad',
                                      eigvecs_prop.conj(), pulse.basis, eigvecs_prop)
        self.assertArrayAlmostEqual(pulse._intermediates['basis_transformed'], basis_transformed,
                                    atol=1e-14)
        self.assertArrayAlmostEqual(pulse._intermediates['phase_factors'],
                                    util.cexp(omega*pulse.t[:-1, None]))

    def test_cache_intermediates_hilbert(self):
        pulse = testutil.rand_pulse_sequence(3, 4, 2, 3)
        omega = util.get_sample_frequencies(pulse, 33, spacing='linear')
        unitary, intermediates = numeric.calculate_noise_operators_from_scratch(
            pulse.eigvals, pulse.eigvecs, pulse.propagators, omega, pulse.n_opers, pulse.n_coeffs,
            pulse.dt, pulse.t, cache_intermediates=True
        )

        pulse._intermediates.update(**intermediates)

        self.assertArrayAlmostEqual(pulse._intermediates['noise_operators_step'].sum(0), unitary)
        self.assertArrayAlmostEqual(pulse._intermediates['n_opers_transformed'],
                                    numeric._transform_hamiltonian(pulse.eigvecs,
                                                                   pulse.n_opers,
                                                                   pulse.n_coeffs))
        self.assertArrayAlmostEqual(pulse._intermediates['phase_factors'],
                                    util.cexp(omega*pulse.t[:-1, None]))


    def test_cache_filter_function(self):
        omega = rng.random(32)
        pulse = testutil.rand_pulse_sequence(2, 3, n_nops=2)
        F_fidelity = numeric.calculate_filter_function(pulse.get_control_matrix(omega),
                                                       'fidelity')
        F_generalized = numeric.calculate_filter_function(pulse.get_control_matrix(omega),
                                                          'generalized')

        pulse.cache_filter_function(omega, filter_function=F_generalized, which='generalized')
        self.assertTrue(pulse.is_cached('filter function'))
        self.assertTrue(pulse.is_cached('generalized filter function'))

        self.assertArrayEqual(pulse.get_filter_function(omega, which='generalized'), F_generalized)
        self.assertArrayEqual(pulse.get_filter_function(omega, which='fidelity'), F_fidelity)

    def test_filter_function(self):
        """Test the filter function calculation and related methods"""
        for d, n_dt in zip(rng.integers(2, 10, (3,)),
                           rng.integers(10, 200, (3,))):
            total_pulse = testutil.rand_pulse_sequence(d, n_dt, 4, 6)
            c_opers, c_coeffs = total_pulse.c_opers, total_pulse.c_coeffs
            n_opers, n_coeffs = total_pulse.n_opers, total_pulse.n_coeffs
            dt = total_pulse.dt

            total_eigvals, total_eigvecs, _ = numeric.diagonalize(
                np.einsum('il,ijk->ljk', c_coeffs, c_opers), total_pulse.dt
            )
            omega = util.get_sample_frequencies(total_pulse, n_samples=100)
            # Try the progress bar
            control_matrix = total_pulse.get_control_matrix(
                omega, show_progressbar=True)

            # Check that some attributes are cached
            self.assertIsNotNone(total_pulse._total_phases)
            self.assertIsNotNone(total_pulse._total_propagator)
            self.assertIsNotNone(total_pulse._total_propagator_liouville)

            # Calculate everything 'on foot'
            pulses = [ff.PulseSequence(list(zip(c_opers, c_coeffs[:, i:i+1])),
                                       list(zip(n_opers, n_coeffs[:, i:i+1])),
                                       dt[i:i+1])
                      for i in range(n_dt)]

            phases = np.empty((n_dt, len(omega)), dtype=complex)
            L = np.empty((n_dt, d**2, d**2))
            control_matrix_g = np.empty((n_dt, 6, d**2, len(omega)),
                                        dtype=complex)
            for g, pulse in enumerate(pulses):
                phases[g] = np.exp(1j*total_pulse.t[g]*omega)
                L[g] = ff.superoperator.liouville_representation(total_pulse.propagators[g],
                                                                 total_pulse.basis)
                control_matrix_g[g] = pulse.get_control_matrix(omega)

            # Check that both methods of calculating the control are the same
            control_matrix_from_atomic = numeric.calculate_control_matrix_from_atomic(
                phases, control_matrix_g, L
            )
            control_matrix_from_scratch = numeric.calculate_control_matrix_from_scratch(
                eigvals=total_eigvals,
                eigvecs=total_eigvecs,
                propagators=total_pulse.propagators,
                omega=omega,
                basis=total_pulse.basis,
                n_opers=n_opers,
                n_coeffs=n_coeffs,
                dt=total_pulse.dt
            )
            self.assertArrayAlmostEqual(control_matrix, control_matrix_from_scratch)
            # first column (identity element) always zero but susceptible to
            # floating point error, increase atol
            self.assertArrayAlmostEqual(control_matrix_from_scratch,
                                        control_matrix_from_atomic,
                                        atol=1e-13)

            # Check if the filter functions for autocorrelated noise are real
            filter_function = total_pulse.get_filter_function(omega)
            self.assertTrue(np.isreal(
                filter_function[np.eye(len(n_opers), dtype=bool)]
            ).all())

            # Check switch between fidelity and generalized filter function
            F_generalized = total_pulse.get_filter_function(
                omega, which='generalized')

            F_fidelity = total_pulse.get_filter_function(
                omega, which='fidelity')

            # Check that F_fidelity is correctly reduced from F_generalized
            self.assertArrayAlmostEqual(F_fidelity,
                                        F_generalized.trace(axis1=2, axis2=3))

            # Hit getters again to check caching functionality
            F_generalized = total_pulse.get_filter_function(
                omega, which='generalized')

            F_fidelity = total_pulse.get_filter_function(
                omega, which='fidelity')

            # Check that F_fidelity is correctly reduced from F_generalized
            self.assertArrayAlmostEqual(F_fidelity,
                                        F_generalized.trace(axis1=2, axis2=3))

            # Different set of frequencies than cached
            F_generalized = total_pulse.get_filter_function(
                omega + 1, which='generalized')

            F_fidelity = total_pulse.get_filter_function(
                omega + 1, which='fidelity')

            # Check that F_fidelity is correctly reduced from F_generalized
            self.assertArrayAlmostEqual(F_fidelity,
                                        F_generalized.trace(axis1=2, axis2=3))

    def test_second_order_filter_function(self):
        for d, n_nops in zip(rng.integers(2, 7, 5), rng.integers(1, 5, 5)):
            pulse = testutil.rand_pulse_sequence(d, 3, 2, n_nops)
            omega = util.get_sample_frequencies(pulse, n_samples=42)

            # Make sure result is the same with or without intermediates
            pulse.cache_control_matrix(omega, cache_intermediates=True)
            F = pulse.get_filter_function(omega, order=1)
            F_1 = pulse.get_filter_function(omega, order=2)
            # Test caching
            F_2 = pulse.get_filter_function(omega, order=2)
            F_3 = numeric.calculate_second_order_filter_function(
                pulse.eigvals, pulse.eigvecs, pulse.propagators, omega, pulse.basis, pulse.n_opers,
                pulse.n_coeffs, pulse.dt, show_progressbar=False, intermediates=None
            )
            # Make sure first and second order are of same order of magnitude
            rel = np.linalg.norm(F) / np.linalg.norm(F_1)

            self.assertIs(F_1, F_2)
            self.assertArrayEqual(F_1, F_3)
            self.assertEqual(F_1.shape, (n_nops, n_nops, d**2, d**2, 42))
            self.assertLessEqual(rel, 10)
            self.assertGreaterEqual(rel, 1/10)

    def test_pulse_correlation_filter_function(self):
        """
        Test calculation of pulse correlation filter function and control
        matrix.
        """
        X, Y, Z = util.paulis[1:]
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

            ff.concatenate([pulses['X'], pulses['Y']], calc_pulse_correlation_FF=True)

        # Get filter functions at same frequencies
        [pulse.cache_filter_function(omega) for pulse in pulses.values()]

        pulse_1 = pulses['X'] @ pulses['Y']
        pulse_2 = ff.concatenate([pulses['X'], pulses['Y']],
                                 calc_pulse_correlation_FF=True,
                                 which='fidelity')
        pulse_3 = ff.concatenate([pulses['X'], pulses['Y']],
                                 calc_pulse_correlation_FF=True,
                                 which='generalized')
        pulse_4 = copy.copy(pulse_3)
        pulse_4.cleanup('all')

        self.assertTrue(pulse_2.is_cached('control_matrix_pc'))
        self.assertTrue(pulse_2.is_cached('filter_function_pc'))
        self.assertTrue(pulse_3.is_cached('control_matrix_pc'))
        self.assertTrue(pulse_3.is_cached('filter_function_pc_gen'))

        # Check if the filter functions on the diagonals are real
        filter_function = pulse_2.get_pulse_correlation_filter_function()
        diag_1 = np.eye(2, dtype=bool)
        diag_2 = np.eye(3, dtype=bool)
        self.assertTrue(np.isreal(filter_function[diag_1][:, diag_2]).all())

        self.assertEqual(pulse_1, pulse_2)
        self.assertEqual(pulse_2.get_pulse_correlation_filter_function().shape,
                         (2, 2, n_nops, n_nops, len(omega)))
        self.assertArrayAlmostEqual(pulse_1.get_filter_function(omega),
                                    pulse_2.get_pulse_correlation_filter_function().sum((0, 1)))
        self.assertArrayAlmostEqual(pulse_1.get_filter_function(omega), pulse_2._filter_function)

        # Test the behavior of the pulse correlation control matrix
        with self.assertRaises(ValueError):
            # R wrong dimension
            numeric.calculate_pulse_correlation_filter_function(pulse_1._control_matrix)

        with self.assertRaises(util.CalculationError):
            # not calculated
            pulse_1.get_pulse_correlation_control_matrix()

        control_matrix_pc = pulse_2.get_pulse_correlation_control_matrix()
        self.assertArrayEqual(
            filter_function,
            numeric.calculate_pulse_correlation_filter_function(
                control_matrix_pc, 'fidelity'
            )
        )

        control_matrix_pc = pulse_3.get_pulse_correlation_control_matrix()
        filter_function = pulse_3.get_pulse_correlation_filter_function(which='fidelity')
        self.assertArrayEqual(
            filter_function,
            numeric.calculate_pulse_correlation_filter_function(
                control_matrix_pc, 'fidelity'
            )
        )

        filter_function = pulse_3.get_pulse_correlation_filter_function(which='generalized')
        self.assertArrayEqual(
            filter_function,
            numeric.calculate_pulse_correlation_filter_function(
                control_matrix_pc, 'generalized'
            )
        )

        # Test caching
        pulse_4.cache_filter_function(omega, control_matrix=control_matrix_pc, which='fidelity')
        self.assertTrue(pulse_4.is_cached('pulse correlation control matrix'))
        self.assertTrue(pulse_4.is_cached('pulse correlation filter function'))
        self.assertTrue(pulse_4.is_cached('filter function'))
        self.assertArrayAlmostEqual(pulse_3.get_pulse_correlation_control_matrix(),
                                    pulse_4.get_pulse_correlation_control_matrix())
        self.assertArrayAlmostEqual(pulse_3.get_pulse_correlation_filter_function(),
                                    pulse_4.get_pulse_correlation_filter_function())
        self.assertArrayAlmostEqual(pulse_3.get_filter_function(omega),
                                    pulse_4.get_filter_function(omega))

        pulse_4.cleanup('all')
        pulse_4.cache_filter_function(omega, control_matrix=control_matrix_pc, which='generalized')
        self.assertTrue(pulse_4.is_cached('pulse correlation control matrix'))
        self.assertTrue(pulse_4.is_cached('generalized pulse correlation filter function'))
        self.assertTrue(pulse_4.is_cached('generalized filter function'))
        self.assertTrue(pulse_4.is_cached('pulse correlation filter function'))
        self.assertTrue(pulse_4.is_cached('filter function'))
        self.assertArrayAlmostEqual(pulse_3.get_pulse_correlation_control_matrix(),
                                    pulse_4.get_pulse_correlation_control_matrix())
        self.assertArrayAlmostEqual(pulse_3.get_pulse_correlation_filter_function('fidelity'),
                                    pulse_4.get_pulse_correlation_filter_function('fidelity'))
        self.assertArrayAlmostEqual(pulse_3.get_pulse_correlation_filter_function('generalized'),
                                    pulse_4.get_pulse_correlation_filter_function('generalized'))
        self.assertArrayAlmostEqual(pulse_3.get_filter_function(omega, which='fidelity'),
                                    pulse_4.get_filter_function(omega, which='fidelity'))
        self.assertArrayAlmostEqual(pulse_3.get_filter_function(omega, which='generalized'),
                                    pulse_4.get_filter_function(omega, which='generalized'))

        # If for some reason filter_function_pc_xy is removed, check if
        # recovered from control_matrix_pc
        pulse_2._filter_function_pc = None
        pulse_3._filter_function_pc_gen = None

        control_matrix_pc = pulse_3.get_pulse_correlation_control_matrix()
        filter_function = pulse_3.get_pulse_correlation_filter_function(which='fidelity')
        self.assertArrayEqual(
            filter_function,
            numeric.calculate_pulse_correlation_filter_function(
                control_matrix_pc, 'fidelity'
            )
        )

        filter_function = pulse_3.get_pulse_correlation_filter_function(which='generalized')
        self.assertArrayEqual(
            filter_function,
            numeric.calculate_pulse_correlation_filter_function(
                control_matrix_pc, 'generalized'
            )
        )

        spectrum = omega**0*1e-2
        with self.assertRaises(util.CalculationError):
            infid_1 = ff.infidelity(pulse_1, spectrum, omega, which='correlations')

        with self.assertRaises(ValueError):
            infid_1 = ff.infidelity(pulse_1, spectrum, omega, which='foobar')

        for _ in range(10):
            n_nops = rng.integers(1, 4)
            identifiers = sample(['B_0', 'B_1', 'B_2'], n_nops)

            infid_X = ff.infidelity(pulses['X'], spectrum, omega,
                                    which='total',
                                    n_oper_identifiers=identifiers)
            infid_Y = ff.infidelity(pulses['Y'], spectrum, omega,
                                    which='total',
                                    n_oper_identifiers=identifiers)
            infid_1 = ff.infidelity(pulse_1, spectrum, omega, which='total',
                                    n_oper_identifiers=identifiers)
            infid_2 = ff.infidelity(pulse_2, spectrum, omega,
                                    which='correlations',
                                    n_oper_identifiers=identifiers)

            self.assertAlmostEqual(infid_1.sum(), infid_2.sum())
            self.assertArrayAlmostEqual(infid_X, infid_2[0, 0])
            self.assertArrayAlmostEqual(infid_Y, infid_2[1, 1])

        # Test function for correlated noise spectra
        spectrum = np.array([[1e-4/omega**2, 1e-4*np.exp(-omega**2)],
                             [1e-4*np.exp(-omega**2), 1e-4/omega**2]])
        infid_1 = ff.infidelity(pulse_1, spectrum, omega, which='total',
                                n_oper_identifiers=['B_0', 'B_2'])
        infid_2 = ff.infidelity(pulse_2, spectrum, omega, which='correlations',
                                n_oper_identifiers=['B_0', 'B_2'])

        self.assertAlmostEqual(infid_1.sum(), infid_2.sum())
        self.assertArrayAlmostEqual(infid_1, infid_2.sum(axis=(0, 1)))

    def test_calculate_decay_amplitudes(self):
        """Test raises of numeric.calculate_decay_amplitudes"""
        pulse = testutil.rand_pulse_sequence(2, 1, 1, 1)

        omega = rng.standard_normal(43)
        # single spectrum
        spectrum = rng.standard_normal(78)
        for i in range(4):
            with self.assertRaises(ValueError):
                numeric.calculate_decay_amplitudes(pulse, np.tile(spectrum, [1]*i), omega)

    def test_cumulant_function(self):
        pulse = testutil.rand_pulse_sequence(2, 1, 1, 1)
        omega = rng.standard_normal(43)
        spectrum = rng.standard_normal(43)
        Gamma = numeric.calculate_decay_amplitudes(pulse, spectrum, omega)
        Delta = numeric.calculate_frequency_shifts(pulse, spectrum, omega)
        K_1 = numeric.calculate_cumulant_function(pulse, spectrum, omega)
        K_2 = numeric.calculate_cumulant_function(pulse, decay_amplitudes=Gamma)
        K_3 = numeric.calculate_cumulant_function(pulse, spectrum, omega, second_order=True)
        K_4 = numeric.calculate_cumulant_function(pulse, decay_amplitudes=Gamma,
                                                  frequency_shifts=Delta, second_order=True)
        self.assertArrayAlmostEqual(K_1, K_2)
        self.assertArrayAlmostEqual(K_3, K_4)

        with self.assertRaises(ValueError):
            # Neither spectrum + frequencies nor decay amplitudes supplied
            numeric.calculate_cumulant_function(pulse, None, None, decay_amplitudes=None)

        with self.assertRaises(ValueError):
            # Neither spectrum + frequencies nor frequency shifts supplied
            numeric.calculate_cumulant_function(pulse, None, None, frequency_shifts=None,
                                                second_order=True)

        with self.assertRaises(ValueError):
            # Trying to get correlation cumulant function for second order
            numeric.calculate_cumulant_function(pulse, spectrum, omega, second_order=True,
                                                which='correlations')

        with self.assertRaises(ValueError):
            # Using precomputed frequency shifts or decay amplitudes but different shapes
            numeric.calculate_cumulant_function(pulse, spectrum, omega, second_order=True,
                                                decay_amplitudes=Gamma[1:])

        with self.assertWarns(UserWarning):
            # Memory parsimonious only works for decay amplitudes
            numeric.calculate_cumulant_function(pulse, spectrum, omega, second_order=True,
                                                memory_parsimonious=True)

        for d in [2, *rng.integers(2, 7, 5)]:
            pulse = testutil.rand_pulse_sequence(d, 3, 2, 2)
            omega = util.get_sample_frequencies(pulse, n_samples=42)
            spectrum = 4e-3/abs(omega)

            pulse.cache_control_matrix(omega, cache_intermediates=True)
            cumulant_function_first_order = numeric.calculate_cumulant_function(
                pulse, spectrum, omega, second_order=False
            )
            cumulant_function_second_order = numeric.calculate_cumulant_function(
                pulse, spectrum, omega, second_order=True
            )
            second_order_contribution = (cumulant_function_second_order
                                         - cumulant_function_first_order)

            # Second order terms should be anti-hermitian
            self.assertArrayAlmostEqual(second_order_contribution,
                                        - second_order_contribution.transpose(0, 2, 1),
                                        atol=1e-16)
            self.assertEqual(cumulant_function_first_order.shape,
                             cumulant_function_second_order.shape)

    def test_error_transfer_matrix(self):
        """Test raises of numeric.error_transfer_matrix."""
        pulse = testutil.rand_pulse_sequence(2, 1, 1, 1)
        omega = testutil.rng.standard_normal(43)
        spectrum = np.ones_like(omega)
        with self.assertRaises(ValueError):
            ff.error_transfer_matrix(pulse, spectrum)

        with self.assertRaises(TypeError):
            ff.error_transfer_matrix(cumulant_function=[1, 2, 3])

        with self.assertRaises((ValueError, np.linalg.LinAlgError)):
            ff.error_transfer_matrix(cumulant_function=testutil.rng.standard_normal((2, 3, 4)))

    def test_infidelity_convergence(self):
        omega = {
            'omega_IR': 0,
            'omega_UV': 2,
            'spacing': 'linear',
            'n_min': 10,
            'n_max': 50,
            'n_points': 4
        }

        def spectrum(omega):
            return omega**0

        simple_pulse = testutil.rand_pulse_sequence(2, 1, 1, 1)
        complicated_pulse = testutil.rand_pulse_sequence(2, 100, 3, 3)

        with self.assertRaises(TypeError):
            n, infids = ff.infidelity(simple_pulse, spectrum, [],
                                      test_convergence=True)

        with self.assertRaises(TypeError):
            n, infids = ff.infidelity(simple_pulse, [1, 2, 3],
                                      dict(spacing='foobar'),
                                      test_convergence=True)

        with self.assertRaises(ValueError):
            n, infids = ff.infidelity(simple_pulse, spectrum,
                                      dict(spacing='foobar'),
                                      test_convergence=True)

        # Test with default args
        n, infids = ff.infidelity(simple_pulse, spectrum, {},
                                  test_convergence=True)

        # Test with non-default args
        identifiers = rng.choice(complicated_pulse.n_oper_identifiers,
                                 rng.integers(1, 4))

        n, infids = ff.infidelity(complicated_pulse, spectrum, omega,
                                  test_convergence=True,
                                  n_oper_identifiers=identifiers)


@pytest.mark.skipif(
    qutip is None,
    reason='Skipping qutip compatibility tests for build without qutip')
class QutipCompatibilityTest(testutil.TestCase):

    def test_pulse_sequence_constructor(self):
        X, Y, Z = qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()
        pulse_1 = ff.PulseSequence(
            [[X, [1, 2, 3], 'X'],
             [util.paulis[2], [3, 4, 5], 'Y'],
             [Z, [5, 6, 7], 'Z']],
            [[util.paulis[3], [1, 2, 3], 'Z'],
             [Y, [3, 4, 5], 'Y'],
             [util.paulis[1], [5, 6, 7], 'X']],
            [1, 3, 5]
        )
        pulse_2 = ff.PulseSequence(
            [[Y, [3, 4, 5], 'Y'],
             [util.paulis[3], [5, 6, 7], 'Z'],
             [util.paulis[1], [1, 2, 3], 'X']],
            [[X, [5, 6, 7], 'X'],
             [Z, [1, 2, 3], 'Z'],
             [util.paulis[2], [3, 4, 5], 'Y']],
            [1, 3, 5]
        )
        self.assertEqual(pulse_1, pulse_2)

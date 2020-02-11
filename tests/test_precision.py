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
This module tests if the package produces the correct results numerically.
"""

import numpy as np
import qutip as qt
from numpy.random import randint, randn
from opt_einsum import contract

import filter_functions as ff
from filter_functions import analytic
from filter_functions.numeric import (
    calculate_error_vector_correlation_functions,
    liouville_representation
)
from tests import testutil


class PrecisionTest(testutil.TestCase):

    def test_FID(self):
        """FID"""
        tau = abs(randn())
        FID_pulse = ff.PulseSequence([[ff.util.P_np[1]/2, [0]]],
                                     [[ff.util.P_np[3]/2, [1]]],
                                     [tau])

        omega = ff.util.get_sample_frequencies(FID_pulse, 50, spacing='linear')
        # Comparison to filter function defined with omega**2
        F = FID_pulse.get_filter_function(omega).squeeze()*omega**2

        self.assertArrayAlmostEqual(F, analytic.FID(omega*tau), atol=1e-10)

    def test_SE(self):
        """Spin echo"""
        tau = np.pi
        tau_pi = 1e-8
        n = 1

        H_c, dt = testutil.generate_dd_hamiltonian(n, tau=tau, tau_pi=tau_pi,
                                                   dd_type='cpmg')

        H_n = [[ff.util.P_np[3]/2, np.ones_like(dt)]]

        SE_pulse = ff.PulseSequence(H_c, H_n, dt)
        omega = ff.util.get_sample_frequencies(SE_pulse, 100, spacing='linear')
        # Comparison to filter function defined with omega**2
        F = SE_pulse.get_filter_function(omega)[0, 0]*omega**2

        self.assertArrayAlmostEqual(F, analytic.SE(omega*tau), atol=1e-10)

        # Test again with a factor of one between the noise operators and
        # coefficients
        r = randn()
        H_n = [[ff.util.P_np[3]/2*r, np.ones_like(dt)/r]]

        SE_pulse = ff.PulseSequence(H_c, H_n, dt)
        # Comparison to filter function defined with omega**2
        F = SE_pulse.get_filter_function(omega)[0, 0]*omega**2

        self.assertArrayAlmostEqual(F, analytic.SE(omega*tau), atol=1e-10)

    def test_6_pulse_CPMG(self):
        """6-pulse CPMG"""
        tau = np.pi
        tau_pi = 1e-9
        n = 6

        H_c, dt = testutil.generate_dd_hamiltonian(n, tau=tau, tau_pi=tau_pi,
                                                   dd_type='cpmg')

        H_n = [[ff.util.P_np[3]/2, np.ones_like(dt)]]

        CPMG_pulse = ff.PulseSequence(H_c, H_n, dt)
        omega = ff.util.get_sample_frequencies(CPMG_pulse, 100, spacing='log')
        # Comparison to filter function defined with omega**2
        F = CPMG_pulse.get_filter_function(omega)[0, 0]*omega**2

        self.assertArrayAlmostEqual(F, analytic.CPMG(omega*tau, n), atol=1e-10)

    def test_6_pulse_UDD(self):
        """6-pulse UDD"""
        tau = np.pi
        tau_pi = 1e-9
        omega = np.logspace(0, 3, 100)
        omega = np.concatenate([-omega[::-1], omega])
        n = 6

        H_c, dt = testutil.generate_dd_hamiltonian(n, tau=tau, tau_pi=tau_pi,
                                                   dd_type='udd')

        H_n = [[ff.util.P_np[3]/2, np.ones_like(dt)]]

        UDD_pulse = ff.PulseSequence(H_c, H_n, dt)
        # Comparison to filter function defined with omega**2
        F = UDD_pulse.get_filter_function(omega)[0, 0]*omega**2

        self.assertArrayAlmostEqual(F, analytic.UDD(omega*tau, n), atol=1e-10)

    def test_6_pulse_PDD(self):
        """6-pulse PDD"""
        tau = np.pi
        tau_pi = 1e-9
        omega = np.logspace(0, 3, 100)
        omega = np.concatenate([-omega[::-1], omega])
        n = 6

        H_c, dt = testutil.generate_dd_hamiltonian(n, tau=tau, tau_pi=tau_pi,
                                                   dd_type='pdd')

        H_n = [[ff.util.P_np[3]/2, np.ones_like(dt)]]

        PDD_pulse = ff.PulseSequence(H_c, H_n, dt)
        # Comparison to filter function defined with omega**2
        F = PDD_pulse.get_filter_function(omega)[0, 0]*omega**2

        self.assertArrayAlmostEqual(F, analytic.PDD(omega*tau, n), atol=1e-10)

    def test_5_pulse_CDD(self):
        """5-pulse CDD"""
        tau = np.pi
        tau_pi = 1e-9
        omega = np.logspace(0, 3, 100)
        omega = np.concatenate([-omega[::-1], omega])
        n = 3

        H_c, dt = testutil.generate_dd_hamiltonian(n, tau=tau, tau_pi=tau_pi,
                                                   dd_type='cdd')

        H_n = [[ff.util.P_np[3]/2, np.ones_like(dt)]]

        CDD_pulse = ff.PulseSequence(H_c, H_n, dt)
        # Comparison to filter function defined with omega**2
        F = CDD_pulse.get_filter_function(omega)[0, 0]*omega**2

        self.assertArrayAlmostEqual(F, analytic.CDD(omega*tau, n), atol=1e-10)

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
            U_liouville = liouville_representation(U[0], basis)
            U_liouville = liouville_representation(U, basis)

            # should have dimension d^2 x d^2
            self.assertEqual(U_liouville.shape, (U.shape[0], d**2, d**2))
            # Real
            self.assertTrue(np.isreal(U_liouville).all())
            # Hermitian for unitary input
            self.assertArrayAlmostEqual(
                U_liouville.swapaxes(-1, -2) @ U_liouville,
                np.tile(np.eye(d**2), (U.shape[0], 1, 1)),
                atol=np.finfo(float).eps*d**2
            )

            if d == 2:
                U_liouville = liouville_representation(ff.util.P_np[1:],
                                                       basis)
                self.assertArrayAlmostEqual(U_liouville[0],
                                            np.diag([1, 1, -1, -1]),
                                            atol=np.finfo(float).eps)
                self.assertArrayAlmostEqual(U_liouville[1],
                                            np.diag([1, -1, 1, -1]),
                                            atol=np.finfo(float).eps)
                self.assertArrayAlmostEqual(U_liouville[2],
                                            np.diag([1, -1, -1, 1]),
                                            atol=np.finfo(float).eps)

    def test_diagonalization_cnot(self):
        """CNOT"""
        subspace_c_opers = testutil.subspace_opers
        subspace_n_opers = subspace_c_opers
        c_opers = testutil.opers
        n_opers = c_opers
        c_coeffs, n_coeffs = testutil.c_coeffs, testutil.n_coeffs
        dt = testutil.dt
        subspace = testutil.subspace

        cnot_subspace = ff.PulseSequence(list(zip(subspace_c_opers, c_coeffs)),
                                         list(zip(subspace_n_opers, n_coeffs)),
                                         dt)

        cnot = ff.PulseSequence(list(zip(c_opers, c_coeffs)),
                                list(zip(n_opers, n_coeffs)),
                                dt)

        cnot.diagonalize()
        cnot_subspace.diagonalize()

        phase_eq = ff.util.oper_equiv(cnot_subspace.total_Q[1:5, 1:5],
                                      qt.cnot(), eps=1e-9)

        self.assertTrue(phase_eq[0])

        phase_eq = ff.util.oper_equiv(
            cnot.total_Q[np.ix_(*subspace)][1:5, 1:5], qt.cnot(), eps=1e-9)

        self.assertTrue(phase_eq[0])

    def test_infidelity_cnot(self):
        """Compare infidelity to monte carlo results"""
        c_opers = testutil.subspace_opers
        n_opers = c_opers
        c_coeffs, n_coeffs = testutil.c_coeffs, testutil.n_coeffs
        dt = testutil.dt
        infid_MC = testutil.cnot_infid_fast
        A = testutil.A

        subspace_projector = np.diag([0, 1, 1, 1, 1, 0])

        # Basis for qubit subspace
        qubit_subspace_basis = ff.Basis(
            [np.pad(b, 1, 'constant') for b in ff.Basis.pauli(2)],
            skip_check=True,
            btype='Pauli'
        )

        identifiers = ['eps_12', 'eps_23', 'eps_34', 'b_12', 'b_23', 'b_34']
        H_c = list(zip(c_opers, c_coeffs, identifiers))
        H_n = list(zip(n_opers, n_coeffs, identifiers))
        cnot = ff.PulseSequence(H_c, H_n, dt, basis=qubit_subspace_basis)
        cnot_full = ff.PulseSequence(H_c, H_n, dt)

        T = dt.sum()
        omega = np.logspace(np.log10(1/T), 2, 125)
        for A, alpha, MC, rtol in zip(A, (0.0, 0.7), infid_MC, (0.1, 0.02)):
            S_t, omega_t = ff.util.symmetrize_spectrum(A/omega**alpha, omega)
            infid, xi = ff.infidelity(cnot, S_t, omega_t, identifiers[:3],
                                      return_smallness=True)
            # infid scaled with d = 6, but we actually have d = 4
            infid *= 1.5
            U = ff.error_transfer_matrix(cnot_full, S_t, omega_t,
                                         identifiers[:3])
            PI = ff.numeric.liouville_representation(subspace_projector,
                                                     cnot_full.basis)
            infid_P = np.trace(PI @ U, axis1=1, axis2=2)/4**2

            self.assertLessEqual(np.abs(1 - (infid.sum()/MC)), rtol)
            self.assertLessEqual(np.abs(1 - (infid_P.sum()/MC)), rtol)
            self.assertLessEqual(infid.sum(), xi**2/4)

    def test_infidelity(self):
        """Benchmark infidelity results against previous version's results"""
        np.random.seed(123456789)

        spectra = [
            lambda S0, omega: S0*omega**0,
            lambda S0, omega: S0/omega**0.7,
            lambda S0, omega: S0*np.exp(-omega),
            # different spectra for different n_opers
            lambda S0, omega: np.array([S0*omega**0, S0/omega**0.7]),
            # cross-correlated spectra
            lambda S0, omega: np.array([[S0/omega**0.7, S0*np.exp(-omega)],
                                        [S0*np.exp(-omega), S0/omega**0.7]])
        ]

        ref_infids = (
            [0.4755805, 0.51188265],
            [0.31035242, 0.39776846],
            [0.06177699, 0.07792038],
            [0.4755805, 0.39776846],
            [[0.31035242, -0.00324699],
             [-0.00324699, 0.39776846]],
            [12.21164461, 7.83251637],
            [7.37577981, 4.39094918],
            [1.34470448, 0.77622854],
            [12.21164461, 4.39094918],
            [[7.37577981, -0.21663542],
             [-0.21663542, 4.39094918]],
            [10.83864437, 3.7166346],
            [6.01031287, 3.21358216],
            [1.09434513, 0.74036871],
            [10.83864437, 3.21358216],
            [[6.01031287, -0.1146554],
             [-0.1146554, 3.21358216]]
        )

        count = 0
        for d in (2, 3, 4):
            c_opers = testutil.rand_herm(d, 2)
            n_opers = testutil.rand_herm(d, 3)
            pulse = ff.PulseSequence(
                list(zip(c_opers, randn(2, 10))),
                list(zip(n_opers, np.abs(randn(3, 10)))),
                np.abs(randn(10))
            )

            omega = np.geomspace(0.1, 10, 51)
            S0 = np.abs(randn())
            for spec in spectra:
                S, omega_t = ff.util.symmetrize_spectrum(spec(S0, omega),
                                                         omega)
                infids = ff.infidelity(pulse, S, omega_t,
                                       n_oper_identifiers=['B_0', 'B_2'])
                self.assertArrayAlmostEqual(infids, ref_infids[count],
                                            atol=1e-8)
                if S.ndim == 3:
                    # Diagonal of the infidelity matrix should correspond to
                    # uncorrelated terms
                    uncorrelated_infids = ff.infidelity(
                        pulse, S[range(2), range(2)], omega_t,
                        n_oper_identifiers=['B_0', 'B_2']
                    )
                    self.assertArrayAlmostEqual(np.diag(infids),
                                                uncorrelated_infids)

                    # Infidelity matrix should be hermitian
                    self.assertArrayEqual(infids, infids.conj().T)

                count += 1

        # Check raises
        with self.assertRaises(TypeError):
            # spectrum not callable
            ff.infidelity(pulse, 2, omega_t, test_convergence=True)

        with self.assertRaises(TypeError):
            # omega not dict
            ff.infidelity(pulse, lambda x: x, 2, test_convergence=True)

        with self.assertRaises(ValueError):
            # omega['spacing'] not in ('linear', 'log')
            ff.infidelity(pulse, lambda x: x, {'spacing': 2},
                          test_convergence=True)

        with self.assertRaises(ValueError):
            # which not total or correlation
            ff.infidelity(pulse, spectra[0](S0, omega_t), omega, which=2)

        with self.assertRaises(ValueError):
            # S wrong dimensions
            ff.infidelity(pulse, spectra[0](S0, omega_t)[:10], omega)

        with self.assertRaises(ValueError):
            # S wrong dimensions
            ff.infidelity(pulse, spectra[3](S0, omega), omega,
                          n_oper_identifiers=['B_0', 'B_1', 'B_2'])

        with self.assertRaises(ValueError):
            # S wrong dimensions
            ff.infidelity(pulse, spectra[4](S0, omega)[:, [0]], omega,
                          n_oper_identifiers=['B_0'])

        with self.assertRaises(NotImplementedError):
            # smallness parameter for correlated noise source
            ff.infidelity(pulse, spectra[4](S0, omega), omega,
                          n_oper_identifiers=['B_0', 'B_2'],
                          return_smallness=True)

    def test_single_qubit_error_transfer_matrix(self):
        """Test the calculation of the single-qubit transfer matrix"""
        d = 2
        for n_dt in randint(1, 11, 10):
            pulse = ff.PulseSequence(
                list(zip(ff.util.P_np[1:], randn(3, n_dt))),
                list(zip(ff.util.P_np[2:], np.abs(randn(3, n_dt)))),
                np.abs(randn(n_dt)),
                basis=ff.Basis.pauli(1)
            )
            omega = ff.util.get_sample_frequencies(pulse, n_samples=51)
            n_oper_identifiers = ['B_0', 'B_1']
            traces = pulse.basis.four_element_traces.todense()

            # Single spectrum
            # Assert fidelity is same as computed by infidelity()
            S = 1e-2/omega**2
            U = ff.error_transfer_matrix(pulse, S, omega)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

            # Check that _single_qubit_error_transfer_matrix and
            # _multi_qubit_... # give the same
            u_kl = calculate_error_vector_correlation_functions(
                pulse, S, omega, n_oper_identifiers
            )
            U_multi = np.zeros_like(U)
            U_multi[..., 1:, 1:] = (
                contract('...kl,klij->...ij', u_kl, traces)/2 +
                contract('...kl,klji->...ij', u_kl, traces)/2 -
                contract('...kl,kilj->...ij', u_kl, traces)
            ).real
            self.assertArrayAlmostEqual(U, U_multi)

            S = np.outer(1e-2*np.arange(1, 3), 400/(omega**2 + 400))
            U = ff.error_transfer_matrix(pulse, S, omega)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

            # Check that _single_qubit_error_transfer_matrix and
            # _multi_qubit_... # give the same
            u_kl = calculate_error_vector_correlation_functions(
                pulse, S, omega, n_oper_identifiers
            )
            U_multi = np.zeros_like(U)
            U_multi[..., 1:, 1:] = (
                contract('...kl,klij->...ij', u_kl, traces)/2 +
                contract('...kl,klji->...ij', u_kl, traces)/2 -
                contract('...kl,kilj->...ij', u_kl, traces)
            ).real
            self.assertArrayAlmostEqual(U, U_multi)

            S = np.einsum('i,j,o->ijo',
                          1e-2*np.arange(1, 3), 1e-2*np.arange(1, 3),
                          400/(omega**2 + 400))
            U = ff.error_transfer_matrix(pulse, S, omega)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

            # Check that _single_qubit_error_transfer_matrix and
            # _multi_qubit_... # give the same
            u_kl = calculate_error_vector_correlation_functions(
                pulse, S, omega, n_oper_identifiers
            )
            U_multi = np.zeros_like(U)
            U_multi[..., 1:, 1:] = (
                contract('...kl,klij->...ij', u_kl, traces)/2 +
                contract('...kl,klji->...ij', u_kl, traces)/2 -
                contract('...kl,kilj->...ij', u_kl, traces)
            ).real
            self.assertArrayAlmostEqual(U, U_multi)

    def test_multi_qubit_error_transfer_matrix(self):
        """Test the calculation of the multi-qubit transfer matrix"""
        for d, n_dt in zip(randint(3, 13, 10), randint(1, 11, 10)):
            f, n = np.modf(np.log2(d))
            if f == 0.0:
                basis = ff.Basis.pauli(int(n))
            else:
                basis = ff.Basis.ggm(d)

            c_opers = testutil.rand_herm(d, randint(2, 4))
            n_opers = testutil.rand_herm(d, randint(2, 4))
            pulse = ff.PulseSequence(
                list(zip(c_opers, randn(len(c_opers), n_dt))),
                list(zip(n_opers, np.abs(randn(len(n_opers), n_dt)))),
                np.abs(randn(n_dt)),
                basis
            )
            omega = ff.util.get_sample_frequencies(pulse, n_samples=51)

            # Assert fidelity is same as computed by infidelity()
            S = 1e-2/omega**2
            U = ff.error_transfer_matrix(pulse, S, omega)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

            S = np.outer(1e-2*(np.arange(len(n_opers)) + 1),
                         400/(omega**2 + 400))
            U = ff.error_transfer_matrix(pulse, S, omega)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

            S = np.einsum('i,j,o->ijo',
                          1e-2*(np.arange(len(n_opers)) + 1),
                          1e-2*(np.arange(len(n_opers)) + 1),
                          400/(omega**2 + 400))
            U = ff.error_transfer_matrix(pulse, S, omega)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

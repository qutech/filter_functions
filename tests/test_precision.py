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
This module tests if the package produces the correct results numerically.
"""

import numpy as np
import qutip as qt

import filter_functions as ff
from filter_functions import analytic, numeric
from tests import testutil


class PrecisionTest(testutil.TestCase):

    def test_FID(self):
        """FID"""
        tau = abs(testutil.rng.randn())
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
        r = testutil.rng.randn()
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
            U_liouville = numeric.liouville_representation(U[0], basis)
            U_liouville = numeric.liouville_representation(U, basis)

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
                U_liouville = numeric.liouville_representation(
                    ff.util.P_np[1:], basis)
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

        # Basis for qubit subspace
        qubit_subspace_basis = ff.Basis(
            [np.pad(b, 1, 'constant') for b in ff.Basis.pauli(2)],
            skip_check=True,
            btype='Pauli'
        )
        complete_basis = ff.Basis(qubit_subspace_basis, traceless=False,
                                  btype='Pauli')

        identifiers = ['eps_12', 'eps_23', 'eps_34', 'b_12', 'b_23', 'b_34']
        H_c = list(zip(c_opers, c_coeffs, identifiers))
        H_n = list(zip(n_opers, n_coeffs, identifiers))
        cnot = ff.PulseSequence(H_c, H_n, dt, basis=qubit_subspace_basis)
        cnot_full = ff.PulseSequence(H_c, H_n, dt, basis=complete_basis)

        # Manually set dimension of pulse as the dimension of the computational
        # subspace
        cnot.d = 4
        T = dt.sum()

        for f_min, A, alpha, MC, rtol in zip((1/T, 1e-2/T), A, (0.0, 0.7),
                                             infid_MC, (0.04, 0.02)):

            omega = np.geomspace(f_min, 1e2, 250)*2*np.pi
            S_t, omega_t = ff.util.symmetrize_spectrum(A/omega**alpha, omega)

            infid, xi = ff.infidelity(cnot, S_t, omega_t, identifiers[:3],
                                      return_smallness=True)

            U = ff.error_transfer_matrix(cnot_full, S_t, omega_t,
                                         identifiers[:3])
            infid_P = np.trace(U[:, :16, :16], axis1=1, axis2=2).real/4**2

            print(np.abs(1 - (infid.sum()/MC)))
            print(np.abs(1 - (infid_P.sum()/MC)))
            self.assertLessEqual(np.abs(1 - (infid.sum()/MC)), rtol)
            self.assertLessEqual(np.abs(1 - (infid_P.sum()/MC)), rtol)
            self.assertLessEqual(infid.sum(), xi**2/4)

    def test_infidelity(self):
        """Benchmark infidelity results against previous version's results"""
        testutil.rng.seed(123456789)

        spectra = [
            lambda S0, omega: S0*omega**0,
            lambda S0, omega: S0/omega**0.7,
            lambda S0, omega: S0*np.exp(-omega),
            # different spectra for different n_opers
            lambda S0, omega: np.array([S0*omega**0, S0/omega**0.7]),
            # cross-correlated spectra
            lambda S0, omega: np.array(
                [[S0/omega**0.7, (1 + 1j)*S0*np.exp(-omega)],
                 [(1 - 1j)*S0*np.exp(-omega), S0/omega**0.7]]
            )
        ]

        ref_infids = (
            [0.448468950307, 0.941871479562],
            [0.65826575772, 1.042914346335],
            [0.163303005479, 0.239032549377],
            [0.448468950307, 1.042914346335],
            [[0.65826575772, 0.069510589685+0.069510589685j],
             [0.069510589685-0.069510589685j, 1.042914346335]],
            [3.687399348243, 3.034914820757],
            [2.590545568435, 3.10093804628],
            [0.55880380219, 0.782544974968],
            [3.687399348243, 3.10093804628],
            [[2.590545568435, -0.114514760108-0.114514760108j],
             [-0.114514760108+0.114514760108j, 3.10093804628]],
            [2.864567451344, 1.270260393902],
            [1.847740998731, 1.559401345443],
            [0.362116177417, 0.388022992097],
            [2.864567451344, 1.559401345443],
            [[1.847740998731, 0.088373663409+0.088373663409j],
             [0.088373663409-0.088373663409j, 1.559401345443]]
        )

        count = 0
        for d in (2, 3, 4):
            pulse = testutil.rand_pulse_sequence(d, 10, 2, 3)
            pulse.n_oper_identifiers = np.array(['B_0', 'B_2'])

            omega = np.geomspace(0.1, 10, 51)
            S0 = np.abs(testutil.rng.randn())
            for spec in spectra:
                S, omega_t = ff.util.symmetrize_spectrum(spec(S0, omega),
                                                         omega)
                infids = ff.infidelity(pulse, S, omega_t,
                                       n_oper_identifiers=['B_0', 'B_2'])
                self.assertArrayAlmostEqual(infids, ref_infids[count],
                                            atol=1e-12)
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

        with self.assertRaises(ValueError):
            # S wrong dimensions
            ff.infidelity(pulse, testutil.rng.randn(2, 3, 4, len(omega)),
                          omega)

        with self.assertRaises(NotImplementedError):
            # smallness parameter for correlated noise source
            ff.infidelity(pulse, spectra[4](S0, omega), omega,
                          n_oper_identifiers=['B_0', 'B_2'],
                          return_smallness=True)

    def test_single_qubit_error_transfer_matrix(self):
        """Test the calculation of the single-qubit transfer matrix"""
        d = 2
        for n_dt in testutil.rng.randint(1, 11, 10):
            pulse = testutil.rand_pulse_sequence(d, n_dt, 3, 2, btype='Pauli')
            omega = ff.util.get_sample_frequencies(pulse, n_samples=51)
            n_oper_identifiers = pulse.n_oper_identifiers
            traces = pulse.basis.four_element_traces.todense()

            # Single spectrum
            # Assert fidelity is same as computed by infidelity()
            S = 1e-2/omega**2
            U = ff.error_transfer_matrix(pulse, S, omega)
            # Calculate U in loop
            Up = ff.error_transfer_matrix(pulse, S, omega,
                                          memory_parsimonious=True)
            self.assertArrayAlmostEqual(Up, U)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

            # Check that _single_qubit_error_transfer_matrix and
            # _multi_qubit_... # give the same
            u_kl = numeric.calculate_error_vector_correlation_functions(
                pulse, S, omega, n_oper_identifiers
            )
            U_multi = (np.einsum('...kl,klij->...ij', u_kl, traces)/2 +
                       np.einsum('...kl,klji->...ij', u_kl, traces)/2 -
                       np.einsum('...kl,kilj->...ij', u_kl, traces))
            self.assertArrayAlmostEqual(U, U_multi, atol=1e-14)

            # Different spectra for each noise oper
            S = np.outer(1e-2*np.arange(1, 3), 400/(omega**2 + 400))
            U = ff.error_transfer_matrix(pulse, S, omega)
            # Calculate U in loop
            Up = ff.error_transfer_matrix(pulse, S, omega,
                                          memory_parsimonious=True)
            self.assertArrayAlmostEqual(Up, U)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

            # Check that _single_qubit_error_transfer_matrix and
            # _multi_qubit_... # give the same
            u_kl = numeric.calculate_error_vector_correlation_functions(
                pulse, S, omega, n_oper_identifiers
            )
            U_multi = (np.einsum('...kl,klij->...ij', u_kl, traces)/2 +
                       np.einsum('...kl,klji->...ij', u_kl, traces)/2 -
                       np.einsum('...kl,kilj->...ij', u_kl, traces))
            self.assertArrayAlmostEqual(U, U_multi, atol=1e-14)

            # Cross-correlated spectra
            S = np.einsum('i,j,o->ijo',
                          1e-2*np.arange(1, 3), 1e-2*np.arange(1, 3),
                          400/(omega**2 + 400), dtype=complex)
            # Cross spectra are complex
            S[0, 1] *= 1 + 1j
            S[1, 0] *= 1 - 1j
            U = ff.error_transfer_matrix(pulse, S, omega)
            # Calculate U in loop
            Up = ff.error_transfer_matrix(pulse, S, omega,
                                          memory_parsimonious=True)
            self.assertArrayAlmostEqual(Up, U)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

            # Check that _single_qubit_error_transfer_matrix and
            # _multi_qubit_... # give the same
            u_kl = numeric.calculate_error_vector_correlation_functions(
                pulse, S, omega, n_oper_identifiers
            )
            U_multi = np.zeros_like(U)
            U_multi = (np.einsum('...kl,klij->...ij', u_kl, traces)/2 +
                       np.einsum('...kl,klji->...ij', u_kl, traces)/2 -
                       np.einsum('...kl,kilj->...ij', u_kl, traces))
            self.assertArrayAlmostEqual(U, U_multi, atol=1e-16)

    def test_multi_qubit_error_transfer_matrix(self):
        """Test the calculation of the multi-qubit transfer matrix"""
        n_cops = 4
        n_nops = 2
        for d, n_dt in zip(testutil.rng.randint(3, 9, 10),
                           testutil.rng.randint(1, 11, 10)):
            f, n = np.modf(np.log2(d))
            btype = 'Pauli' if f == 0.0 else 'GGM'
            pulse = testutil.rand_pulse_sequence(d, n_dt, n_cops, n_nops,
                                                 btype)
            omega = ff.util.get_sample_frequencies(pulse, n_samples=51)

            # Assert fidelity is same as computed by infidelity()
            S = 1e-2/omega**2
            U = ff.error_transfer_matrix(pulse, S, omega)
            # Calculate U in loop
            Up = ff.error_transfer_matrix(pulse, S, omega,
                                          memory_parsimonious=True)
            self.assertArrayAlmostEqual(Up, U)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

            S = np.outer(1e-2*(np.arange(n_nops) + 1),
                         400/(omega**2 + 400))
            U = ff.error_transfer_matrix(pulse, S, omega)
            # Calculate U in loop
            Up = ff.error_transfer_matrix(pulse, S, omega,
                                          memory_parsimonious=True)
            self.assertArrayAlmostEqual(Up, U)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

            S = np.einsum('i,j,o->ijo',
                          1e-2*(np.arange(n_nops) + 1),
                          1e-2*(np.arange(n_nops) + 1),
                          400/(omega**2 + 400))
            U = ff.error_transfer_matrix(pulse, S, omega)
            # Calculate U in loop
            Up = ff.error_transfer_matrix(pulse, S, omega,
                                          memory_parsimonious=True)
            self.assertArrayAlmostEqual(Up, U)
            I_fidelity = ff.infidelity(pulse, S, omega)
            I_transfer = np.einsum('...ii', U)/d**2
            self.assertArrayAlmostEqual(I_transfer, I_fidelity)

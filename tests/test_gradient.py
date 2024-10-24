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
#     Written by Julian Teske.
#
#     Contact email: j.teske@fz-juelich.de
# =============================================================================
import numpy as np

import filter_functions as ff
from tests import gradient_testutil, testutil


class GradientTest(testutil.TestCase):

    def test_gradient_calculation_variable_noise_coefficients(self):

        initial_pulse = testutil.rng.uniform(size=(1, gradient_testutil.n_time_steps))
        u_drift = np.full(gradient_testutil.n_time_steps, testutil.rng.standard_normal())
        # dJ/dJ = 1
        n_coeffs_deriv = np.ones((1, 1, gradient_testutil.n_time_steps))

        fin_diff_grad = gradient_testutil.finite_diff_infid(
            u_ctrl_central=initial_pulse, u_drift=u_drift, d=2,
            pulse_sequence_builder=gradient_testutil.create_sing_trip_pulse_seq,
            spectral_noise_density=gradient_testutil.one_over_f_noise, n_freq_samples=200,
            c_id=['control1'], delta_u=1e-4
        )
        ana_grad = gradient_testutil.analytic_gradient(
            u_ctrl=initial_pulse, u_drift=u_drift, d=2,
            pulse_sequence_builder=gradient_testutil.create_sing_trip_pulse_seq,
            spectral_noise_density=gradient_testutil.one_over_f_noise,
            c_id=['control1'], n_coeffs_deriv=n_coeffs_deriv,
            ctrl_amp_deriv=gradient_testutil.deriv_exchange_interaction
        )
        self.assertArrayAlmostEqual(ana_grad, fin_diff_grad, rtol=1e-6, atol=1e-10)

    def test_n_coeffs_deriv_sorting(self):
        for _ in range(5):
            pulse = testutil.rand_pulse_sequence(testutil.rng.integers(2, 5),
                                                 testutil.rng.integers(2, 11))
            omega = ff.util.get_sample_frequencies(pulse, n_samples=37)

            # Not the correct derivative, but irrelevant for comparison between analytics
            n_coeffs_deriv = testutil.rng.normal(size=(len(pulse.n_opers),
                                                       len(pulse.c_opers),
                                                       len(pulse)))

            # indices to sort sorted opers into a hypothetical unsorted original order.
            n_oper_unsort_idx = np.random.permutation(np.arange(len(pulse.n_opers)))
            c_oper_unsort_idx = np.random.permutation(np.arange(len(pulse.c_opers)))

            # subset of c_opers and n_opers to compute the derivative for
            n_choice = np.random.choice(np.arange(len(pulse.n_opers)),
                                        testutil.rng.integers(1, len(pulse.n_opers) + 1),
                                        replace=False)
            c_choice = np.random.choice(np.arange(len(pulse.c_opers)),
                                        testutil.rng.integers(1, len(pulse.c_opers) + 1),
                                        replace=False)

            grad = pulse.get_filter_function_derivative(
                omega,
                n_coeffs_deriv=n_coeffs_deriv
            )
            grad_as_given = pulse.get_filter_function_derivative(
                omega,
                n_oper_identifiers=pulse.n_oper_identifiers[n_oper_unsort_idx],
                control_identifiers=pulse.c_oper_identifiers[c_oper_unsort_idx],
                n_coeffs_deriv=n_coeffs_deriv[n_oper_unsort_idx[:, None], c_oper_unsort_idx]
            )
            grad_n_choice = pulse.get_filter_function_derivative(
                omega,
                n_oper_identifiers=pulse.n_oper_identifiers[n_choice],
                n_coeffs_deriv=n_coeffs_deriv[n_choice]
            )
            grad_c_choice = pulse.get_filter_function_derivative(
                omega,
                control_identifiers=pulse.c_oper_identifiers[c_choice],
                n_coeffs_deriv=n_coeffs_deriv[:, c_choice]
            )
            grad_nc_choice = pulse.get_filter_function_derivative(
                omega,
                control_identifiers=pulse.c_oper_identifiers[c_choice],
                n_oper_identifiers=pulse.n_oper_identifiers[n_choice],
                n_coeffs_deriv=n_coeffs_deriv[n_choice[:, None], c_choice]
            )
            self.assertArrayAlmostEqual(
                grad[np.ix_(n_oper_unsort_idx, np.arange(len(pulse)), c_oper_unsort_idx)],
                grad_as_given
            )
            self.assertArrayAlmostEqual(
                grad[np.ix_(n_choice, np.arange(len(pulse)))],
                grad_n_choice
            )
            self.assertArrayAlmostEqual(
                grad[np.ix_(np.arange(len(pulse.n_opers)), np.arange(len(pulse)), c_choice)],
                grad_c_choice
            )
            self.assertArrayAlmostEqual(
                grad[np.ix_(n_choice, np.arange(len(pulse)), c_choice)],
                grad_nc_choice
            )

    def test_gradient_calculation_random_pulse(self):

        for d, n_dt in zip(testutil.rng.integers(2, 5, 5), testutil.rng.integers(2, 8, 5)):
            u_ctrl = testutil.rng.normal(0, 1, (testutil.rng.integers(1, 4), n_dt))
            u_drift = testutil.rng.normal(0, 1, (d**2-1-len(u_ctrl), n_dt))

            fin_diff_grad = gradient_testutil.finite_diff_infid(
                u_ctrl_central=u_ctrl, u_drift=u_drift, d=d,
                pulse_sequence_builder=gradient_testutil.create_pulse_sequence,
                spectral_noise_density=gradient_testutil.one_over_f_noise, n_freq_samples=200,
                c_id=[f'c{i}' for i in range(len(u_ctrl))], delta_u=1e-4
            )
            ana_grad = gradient_testutil.analytic_gradient(
                u_ctrl=u_ctrl, u_drift=u_drift, d=d,
                pulse_sequence_builder=gradient_testutil.create_pulse_sequence,
                spectral_noise_density=gradient_testutil.one_over_f_noise,
                c_id=[f'c{i}' for i in range(len(u_ctrl))], n_coeffs_deriv=None
            )
            self.assertArrayAlmostEqual(ana_grad, fin_diff_grad, rtol=1e-5, atol=1e-7)

    def test_caching(self):
        """Make sure calculation works with or without cached intermediates."""

        for d, n_dt in zip(testutil.rng.integers(2, 5, 5), testutil.rng.integers(2, 8, 5)):
            pulse = testutil.rand_pulse_sequence(d, n_dt)
            omega = ff.util.get_sample_frequencies(pulse, n_samples=27)
            spect = 1/omega

            # Cache control matrix but not intermediates
            pulse.cache_control_matrix(omega, cache_intermediates=False)
            infid_nocache = ff.infidelity(pulse, spect, omega, cache_intermediates=False)
            infid_cache = ff.infidelity(pulse, spect, omega, cache_intermediates=True)

            self.assertArrayAlmostEqual(infid_nocache, infid_cache)

            cm_nocache = ff.gradient.calculate_derivative_of_control_matrix_from_scratch(
                omega, pulse.propagators, pulse.eigvals, pulse.eigvecs, pulse.basis, pulse.t,
                pulse.dt, pulse.n_opers, pulse.n_coeffs, pulse.c_opers, intermediates=dict()
            )

            pulse.cleanup('frequency dependent')
            pulse.cache_control_matrix(omega, cache_intermediates=True)
            cm_cache = ff.gradient.calculate_derivative_of_control_matrix_from_scratch(
                omega, pulse.propagators, pulse.eigvals, pulse.eigvecs, pulse.basis, pulse.t,
                pulse.dt, pulse.n_opers, pulse.n_coeffs, pulse.c_opers,
                intermediates=pulse._intermediates
            )

            self.assertArrayAlmostEqual(cm_nocache, cm_cache)

    def test_raises(self):
        pulse = testutil.rand_pulse_sequence(2, 3)
        omega = ff.util.get_sample_frequencies(pulse, n_samples=13)
        with self.assertRaises(ValueError):
            ff.infidelity_derivative(pulse, 1/omega, omega, control_identifiers=['long string'])

        with self.assertRaises(ValueError):
            pulse.get_filter_function_derivative(
                omega,
                n_coeffs_deriv=testutil.rng.normal(size=(2, 5, 10))
            )

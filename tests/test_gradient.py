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

        initial_pulse = np.random.rand(gradient_testutil.n_time_steps)
        initial_pulse = np.expand_dims(initial_pulse, 0)
        u_drift = 1. * np.ones(gradient_testutil.n_time_steps)

        n_coeffs_deriv = gradient_testutil.deriv_2_exchange_interaction(eps=initial_pulse)
        n_coeffs_deriv = np.expand_dims(n_coeffs_deriv, 0)

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
            c_id=['control1'], n_coeffs_deriv=np.ones_like(n_coeffs_deriv),
            ctrl_amp_deriv=gradient_testutil.deriv_exchange_interaction
        )
        self.assertArrayAlmostEqual(ana_grad, fin_diff_grad, rtol=1e-6, atol=1e-10)

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
            self.assertArrayAlmostEqual(ana_grad, fin_diff_grad, rtol=1e-6, atol=1e-8)

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
                pulse.dt, pulse.n_opers, pulse.n_coeffs, pulse.c_opers, pulse.c_oper_identifiers,
                intermediates=dict()
            )

            pulse.cleanup('frequency dependent')
            pulse.cache_control_matrix(omega, cache_intermediates=True)
            cm_cache = ff.gradient.calculate_derivative_of_control_matrix_from_scratch(
                omega, pulse.propagators, pulse.eigvals, pulse.eigvecs, pulse.basis, pulse.t,
                pulse.dt, pulse.n_opers, pulse.n_coeffs, pulse.c_opers, pulse.c_oper_identifiers,
                intermediates=pulse._intermediates
            )

            self.assertArrayAlmostEqual(cm_nocache, cm_cache)

    def test_raises(self):
        pulse = testutil.rand_pulse_sequence(2, 3)
        omega = ff.util.get_sample_frequencies(pulse, n_samples=13)
        with self.assertRaises(ValueError):
            ff.infidelity_derivative(pulse, 1/omega, omega, control_identifiers=['long string'])

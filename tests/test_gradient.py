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

import tests.gradient_testutil as grad_util
from tests import testutil

np.random.seed(0)
initial_pulse = np.random.rand(grad_util.n_time_steps)
initial_pulse = np.expand_dims(initial_pulse, 0)
u_drift = 1. * np.ones(grad_util.n_time_steps)

s_derivs = grad_util.deriv_2_exchange_interaction(eps=initial_pulse)
s_derivs = np.expand_dims(s_derivs, 0)


fin_diff_grad = grad_util.finite_diff_infid(
    u_ctrl_central=initial_pulse, u_drift=u_drift,
    pulse_sequence_builder=grad_util.create_sing_trip_pulse_seq,
    spectral_noise_density=grad_util.one_over_f_noise, n_freq_samples=200,
    c_id=['control1'], delta_u=1e-4
)


ana_grad = grad_util.analytic_gradient(
    u_ctrl=initial_pulse, u_drift=u_drift,
    pulse_sequence_builder=grad_util.create_sing_trip_pulse_seq,
    spectral_noise_density=grad_util.one_over_f_noise,
    c_id=['control1'], s_derivs=np.ones_like(s_derivs),
    ctrl_amp_deriv=grad_util.deriv_exchange_interaction
)


class GradientTest(testutil.TestCase):

    def test_gradient_calculation_variable_noise_coefficients(self):
        relativ_diff = grad_util.relative_norm_difference(
            fin_diff_grad, ana_grad)
        self.assertLess(relativ_diff, 5e-9)

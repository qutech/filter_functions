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
This module tests if optional extras are handled correctly.
"""
import os
from unittest import mock

import pytest
from numpy import ndarray

from tests import testutil

from . import matplotlib

all_extras = ['plotting', 'bloch_sphere_visualization']


class MissingExtrasTest(testutil.TestCase):

    @pytest.mark.skipif(
        any(extra in os.environ.get('INSTALL_EXTRAS', all_extras)
            for extra in ['plotting', 'bloch_sphere_visualization']),
        reason='Skipping tests for missing plotting extra in build with matplotlib')
    def test_plotting_not_available(self):
        with self.assertRaises(ModuleNotFoundError):
            from filter_functions import plotting  # noqa

    @pytest.mark.skipif(
        ('bloch_sphere_visualization' in os.environ.get('INSTALL_EXTRAS', all_extras)
         or matplotlib is None),
        reason='Skipping tests for missing bloch sphere visualization tests in build with qutip')
    def test_bloch_sphere_visualization_not_available(self):

        if matplotlib is not None:
            from filter_functions import plotting
        else:
            plotting = mock.Mock()

        with self.assertRaises(RuntimeError):
            plotting.get_bloch_vector(testutil.rng.standard_normal((10, 2)))

        with self.assertRaises(RuntimeError):
            plotting.init_bloch_sphere()

        with self.assertRaises(RuntimeError):
            plotting.plot_bloch_vector_evolution(
                testutil.rand_pulse_sequence(2, 1))

        from filter_functions import types
        self.assertIs(types.State, ndarray)
        self.assertIs(types.Operator, ndarray)

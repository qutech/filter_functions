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
This module tests the plotting functionality of the package.
"""
import string
from copy import copy
from random import sample

import numpy as np
import pytest

import filter_functions as ff
from filter_functions import numeric
from tests import testutil
from tests.testutil import rng

from . import qutip

plotting = pytest.importorskip('filter_functions.plotting',
                               reason='Skipping plotting tests for build without matplotlib')
if plotting is not None:
    import matplotlib.pyplot as plt
    from cycler import cycler

simple_pulse = testutil.rand_pulse_sequence(2, 1, 1, 1, btype='Pauli')
complicated_pulse = testutil.rand_pulse_sequence(2, 100, 3, 3)
two_qubit_pulse = testutil.rand_pulse_sequence(4, 1, 1, 6, btype='Pauli')


class PlottingTest(testutil.TestCase):

    def test_plot_pulse_train(self):
        # Call with default args
        fig, ax, leg = plotting.plot_pulse_train(simple_pulse)

        # Call with no axes but figure
        fig = plt.figure()
        fig, ax, leg = plotting.plot_pulse_train(simple_pulse, fig=fig)

        # Call with axes but no figure
        fig, ax, leg = plotting.plot_pulse_train(simple_pulse, axes=ax)

        # Call with custom args
        c_oper_identifiers = sample(
            complicated_pulse.c_oper_identifiers.tolist(), rng.integers(2, 4)
        )

        fig, ax = plt.subplots()
        fig, ax, leg = plotting.plot_pulse_train(complicated_pulse,
                                                 c_oper_identifiers,
                                                 fig=fig, axes=ax)

        # Test cycler arg
        cycle = cycler(color=['r', 'g', 'b'])
        fig, ax, leg = plotting.plot_pulse_train(simple_pulse, cycler=cycle)

        # invalid identifier
        with self.assertRaises(ValueError):
            plotting.plot_pulse_train(complicated_pulse,
                                      c_oper_identifiers=['foo'],
                                      fig=fig, axes=ax)

        # Test various keyword args for matplotlib
        plot_kw = {'linewidth': 1}
        subplot_kw = {'facecolor': 'r'}
        gridspec_kw = {'hspace': 0.2,
                       'wspace': 0.1}
        figure_kw = {'num': rng.integers(1, 10000)}

        fig, ax, leg = plotting.plot_pulse_train(simple_pulse, plot_kw=plot_kw,
                                                 subplot_kw=subplot_kw,
                                                 gridspec_kw=gridspec_kw,
                                                 **figure_kw)

        plt.close('all')

    def test_plot_filter_function(self):
        # Call with default args
        simple_pulse.cleanup('all')
        fig, ax, leg = plotting.plot_filter_function(simple_pulse)

        # Call with no axes but figure
        fig = plt.figure()
        fig, ax, leg = plotting.plot_filter_function(simple_pulse, fig=fig)

        # Call with axes but no figure
        fig, ax, leg = plotting.plot_filter_function(simple_pulse, axes=ax)

        # Non-default args
        n_oper_identifiers = sample(
            complicated_pulse.n_oper_identifiers.tolist(), rng.integers(2, 4)
        )

        fig, ax = plt.subplots()
        omega = ff.util.get_sample_frequencies(complicated_pulse, n_samples=50,
                                               spacing='log')
        fig, ax, leg = plotting.plot_filter_function(
            complicated_pulse, omega=omega,
            n_oper_identifiers=n_oper_identifiers,
            fig=fig, axes=ax, omega_in_units_of_tau=False
        )

        # Test cycler arg
        cycle = cycler(color=['r', 'g', 'b'])
        fig, ax, leg = plotting.plot_filter_function(simple_pulse, cycler=cycle)

        # invalid identifier
        with self.assertRaises(ValueError):
            plotting.plot_filter_function(complicated_pulse,
                                          n_oper_identifiers=['foo'],
                                          fig=fig, axes=ax)

        # Test different axis scales
        scales = ('linear', 'log')
        for xscale in scales:
            for yscale in scales:
                fig, ax, leg = plotting.plot_filter_function(simple_pulse,
                                                             xscale=xscale,
                                                             yscale=yscale)

        # Test various keyword args for matplotlib
        plot_kw = {'linewidth': 1}
        subplot_kw = {'facecolor': 'r'}
        gridspec_kw = {'hspace': 0.2,
                       'wspace': 0.1}
        figure_kw = {'num': rng.integers(1, 10000)}
        fig, ax, leg = plotting.plot_filter_function(simple_pulse,
                                                     plot_kw=plot_kw,
                                                     subplot_kw=subplot_kw,
                                                     gridspec_kw=gridspec_kw,
                                                     **figure_kw)

        plt.close('all')

    @pytest.mark.filterwarnings('ignore::UserWarning')  # tight_layout warning
    def test_plot_pulse_correlation_filter_function(self):
        omega = np.linspace(-1, 1, 50)
        concatenated_simple_pulse = ff.concatenate(
            (simple_pulse, simple_pulse),
            calc_pulse_correlation_FF=True,
            omega=omega
        )
        concatenated_complicated_pulse = ff.concatenate(
            (complicated_pulse, complicated_pulse),
            calc_pulse_correlation_FF=True,
            omega=omega
        )

        # Exception if not pulse correl. FF is cached
        with self.assertRaises(ff.util.CalculationError):
            plotting.plot_pulse_correlation_filter_function(simple_pulse)

        # Call with default args
        fig, ax, leg = plotting.plot_pulse_correlation_filter_function(
            concatenated_simple_pulse
        )

        # Non-default args
        n_oper_identifiers = sample(
            complicated_pulse.n_oper_identifiers.tolist(), rng.integers(2, 4)
        )

        fig, ax = plt.subplots()
        omega = np.linspace(-10, 10, 50)
        fig, ax, leg = plotting.plot_pulse_correlation_filter_function(
            concatenated_complicated_pulse, omega=omega,
            n_oper_identifiers=n_oper_identifiers, fig=fig,
            omega_in_units_of_tau=False
        )

        # Test cycler arg
        cycle = cycler(color=['r', 'g', 'b'])
        fig, ax, leg = plotting.plot_pulse_correlation_filter_function(concatenated_simple_pulse,
                                                                       cycler=cycle)

        # invalid identifiers
        with self.assertRaises(ValueError):
            plotting.plot_pulse_correlation_filter_function(
                concatenated_complicated_pulse,
                n_oper_identifiers=['foo'], fig=fig, axes=ax
            )

        # Test different axis scales
        scales = ('linear', 'log')
        for xscale in scales:
            for yscale in scales:
                fig, ax, leg = plotting.plot_pulse_correlation_filter_function(
                    concatenated_simple_pulse, xscale=xscale, yscale=yscale
                )

        # Test various keyword args for matplotlib
        plot_kw = {'linewidth': 1}
        subplot_kw = {'facecolor': 'r'}
        gridspec_kw = {'hspace': 0.2,
                       'wspace': 0.1}
        figure_kw = {'num': rng.integers(1, 10000)}
        fig, ax, leg = plotting.plot_pulse_correlation_filter_function(
            concatenated_simple_pulse, plot_kw=plot_kw, subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw, **figure_kw
        )

        plt.close('all')

    # Ignore deprecation warning caused by qutip
    @pytest.mark.filterwarnings('ignore::PendingDeprecationWarning')
    def test_plot_cumulant_function(self):
        omega = ff.util.get_sample_frequencies(simple_pulse)
        spectrum = 1e-4*np.sin(omega)/omega

        # Test calling with pulse, spectrum, omega
        fig, grid = plotting.plot_cumulant_function(simple_pulse, spectrum, omega,
                                                    colorscale='linear')
        fig, grid = plotting.plot_cumulant_function(simple_pulse, spectrum, omega, fig=fig)
        fig, grid = plotting.plot_cumulant_function(simple_pulse, spectrum, omega, grid=grid)

        # Test calling with precomputed transfer matrix
        K = numeric.calculate_cumulant_function(simple_pulse, spectrum, omega)
        fig, grid = plotting.plot_cumulant_function(cumulant_function=K)

        # Test calling with precomputed transfer matrix and pulse
        K = numeric.calculate_cumulant_function(simple_pulse, spectrum, omega)
        fig, grid = plotting.plot_cumulant_function(simple_pulse, cumulant_function=K)

        # Test calling with precomputed transfer matrix of ndim == 2
        K = numeric.calculate_cumulant_function(simple_pulse, spectrum, omega)
        fig, grid = plotting.plot_cumulant_function(cumulant_function=K[0])

        # Log colorscale
        fig, grid = plotting.plot_cumulant_function(cumulant_function=K, colorscale='log')

        # Non-default args
        n_oper_inds = sample(range(len(complicated_pulse.n_opers)), rng.integers(2, 4))
        n_oper_identifiers = complicated_pulse.n_oper_identifiers[n_oper_inds]

        basis_labels = []
        for i in range(4):
            basis_labels.append(string.ascii_uppercase[rng.integers(0, 26)])

        omega = ff.util.get_sample_frequencies(complicated_pulse, n_samples=50, spacing='log')
        spectrum = np.exp(-omega**2)
        K = numeric.calculate_cumulant_function(complicated_pulse, spectrum, omega)
        fig, grid = plotting.plot_cumulant_function(
            complicated_pulse, spectrum=spectrum, omega=omega,
            n_oper_identifiers=n_oper_identifiers, basis_labels=np.array(basis_labels),
            basis_labelsize=4, linthresh=1e-4, cmap=plt.cm.jet
        )
        fig, grid = plotting.plot_cumulant_function(
            cumulant_function=K[n_oper_inds], n_oper_identifiers=n_oper_identifiers,
            basis_labels=basis_labels, basis_labelsize=4, linthresh=1e-4,
            cmap=plt.cm.jet
        )

        # neither K nor all of pulse, spectrum, omega given
        with self.assertRaises(ValueError):
            plotting.plot_cumulant_function(complicated_pulse, spectrum)

        # invalid identifiers
        with self.assertRaises(ValueError):
            plotting.plot_cumulant_function(complicated_pulse, spectrum, omega,
                                            n_oper_identifiers=['foo'])

        # K and identifiers given but unequal length
        with self.assertRaises(ValueError):
            plotting.plot_cumulant_function(cumulant_function=K,
                                            n_oper_identifiers=n_oper_identifiers[0])

        # number of basis_labels not correct
        with self.assertRaises(ValueError):
            plotting.plot_cumulant_function(complicated_pulse, spectrum, omega,
                                            basis_labels=basis_labels[:2])

        # grid too small
        with self.assertRaises(ValueError):
            plotting.plot_cumulant_function(complicated_pulse, spectrum, omega,
                                            grid=grid[:1])

        # Test various keyword args for matplotlib for the two-qubit pulse
        spectrum = np.tile(spectrum, (6, 6, 1))
        grid_kw = {'axes_pad': 0.1}
        cbar_kw = {'orientation': 'horizontal'}
        imshow_kw = {'interpolation': 'bilinear'}
        figure_kw = {'num': rng.integers(1, 10000)}
        fig, ax = plotting.plot_cumulant_function(two_qubit_pulse, spectrum, omega,
                                                  imshow_kw=imshow_kw, grid_kw=grid_kw,
                                                  cbar_kw=cbar_kw, **figure_kw)

        plt.close('all')

    def test_plot_infidelity_convergence(self):
        n, infids = ff.infidelity(simple_pulse, lambda x: x**0, {}, test_convergence=True)
        fig, ax = plotting.plot_infidelity_convergence(n, infids)

        fig, ax = plt.subplots(1, 2)
        cfig, ax = plotting.plot_infidelity_convergence(n, infids, ax)
        self.assertIs(cfig, fig)


class LaTeXRenderingTest(testutil.TestCase):

    def test_plot_filter_function(self):
        pulse = copy(simple_pulse)
        pulse.c_oper_identifiers = np.array([f'B_{i}' for i in range(len(pulse.c_opers))])
        pulse.n_oper_identifiers = np.array([f'B_{i}' for i in range(len(pulse.n_opers))])
        with plt.rc_context(rc={'text.usetex': True}):
            _ = plotting.plot_pulse_train(pulse)
            _ = plotting.plot_filter_function(pulse)


@pytest.mark.skipif(
    qutip is None,
    reason='Skipping bloch sphere visualization tests for build without qutip')
class BlochSphereVisualizationTest(testutil.TestCase):

    def test_get_bloch_vector(self):
        states = [qutip.rand_ket(2) for _ in range(10)]
        bloch_vectors_qutip = plotting.get_bloch_vector(states)
        bloch_vectors_np = plotting.get_bloch_vector([state.full()
                                                      for state in states])

        for bv_qutip, bv_np in zip(bloch_vectors_qutip, bloch_vectors_np):
            self.assertArrayAlmostEqual(bv_qutip, bv_np)

    def test_get_states_from_prop(self):
        P = testutil.rand_unit(2, 10)
        Q = np.empty((11, 2, 2), dtype=complex)
        Q[0] = np.identity(2)
        for i in range(10):
            Q[i+1] = P[i] @ Q[i]

        psi0 = qutip.rand_ket(2)
        psi0_np = psi0.full()
        states = plotting.get_states_from_prop(Q, psi0)
        states_np = plotting.get_states_from_prop(Q, psi0_np)
        states_0 = plotting.get_states_from_prop(Q)
        self.assertArrayAlmostEqual(Q @ psi0_np, states)
        self.assertArrayAlmostEqual(Q @ psi0_np, states_np)
        self.assertArrayAlmostEqual(Q @ qutip.basis(2, 0).full(), states_0)

        with self.assertRaises(ValueError):
            plotting.get_states_from_prop(Q, rng.standard_normal((3, 1, 2)))
        with self.assertRaises(ValueError):
            plotting.get_states_from_prop(Q, rng.standard_normal((3, 4)))

    def test_plot_bloch_vector_evolution(self):
        # Call with default args
        b = plotting.plot_bloch_vector_evolution(simple_pulse)
        # Call with custom args
        b = plotting.init_bloch_sphere(background=True)
        b = plotting.plot_bloch_vector_evolution(simple_pulse,
                                                 psi0=qutip.basis(2, 1), b=b,
                                                 n_samples=50, show=False,
                                                 return_Bloch=True)

        b = plotting.plot_bloch_vector_evolution(complicated_pulse)

        # Test add_cbar kwarg
        b = plotting.plot_bloch_vector_evolution(simple_pulse, cmap='viridis', add_cbar=True)

        # Check exceptions being raised
        with self.assertRaises(ValueError):
            plotting.plot_bloch_vector_evolution(two_qubit_pulse)

        plt.close('all')

    def test_box_aspect(self):
        """Fix https://github.com/qutech/filter_functions/issues/41"""
        b = plotting.plot_bloch_vector_evolution(simple_pulse, return_Bloch=True)
        aspect = b.axes.get_box_aspect()
        self.assertAlmostEqual(aspect.std(), 0)

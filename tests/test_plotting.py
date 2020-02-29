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
import matplotlib
# Needs to be executed before the pyplot import
matplotlib.use('Agg')

import string
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt
from numpy.random import randint, randn
from tests import testutil

import filter_functions as ff
from filter_functions.plotting import (get_bloch_vector, get_states_from_prop,
                                       init_bloch_sphere,
                                       plot_bloch_vector_evolution,
                                       plot_error_transfer_matrix,
                                       plot_filter_function,
                                       plot_pulse_correlation_filter_function,
                                       plot_pulse_train)

simple_pulse = ff.PulseSequence(
    [[qt.sigmax(), [np.pi/2]]],
    [[qt.sigmax(), [1]]],
    [1],
    basis=ff.Basis.pauli(1)
)
complicated_pulse = ff.PulseSequence(
    list(zip(ff.util.P_qt[1:], randn(3, 100))),
    list(zip(ff.util.P_qt[1:], np.abs(randn(3, 100)))),
    np.abs(randn(100))
)
two_qubit_pulse = ff.PulseSequence(
    [[qt.tensor(qt.sigmaz(), qt.qeye(2)), [np.pi/2]]],
    [[qt.tensor(qt.sigmax(), qt.qeye(2)), [1]],
     [qt.tensor(qt.sigmay(), qt.qeye(2)), [1]],
     [qt.tensor(qt.sigmaz(), qt.qeye(2)), [1]],
     [qt.tensor(qt.qeye(2), qt.sigmax()), [1]],
     [qt.tensor(qt.qeye(2), qt.sigmay()), [1]],
     [qt.tensor(qt.qeye(2), qt.sigmaz()), [1]]],
    [1],
    ff.Basis.pauli(2)
)


class PlottingTest(testutil.TestCase):

    def test_get_bloch_vector(self):
        states = [qt.rand_ket(2) for _ in range(10)]
        bloch_vectors_qt = get_bloch_vector(states)
        bloch_vectors_np = get_bloch_vector([state.full() for state in states])

        for bv_qt, bv_np in zip(bloch_vectors_qt, bloch_vectors_np):
            self.assertArrayAlmostEqual(bv_qt, bv_np)

    def test_get_states_from_prop(self):
        P = testutil.rand_unit(2, 10)
        Q = np.empty((11, 2, 2), dtype=complex)
        Q[0] = np.identity(2)
        for i in range(10):
            Q[i+1] = P[i] @ Q[i]

        psi0 = qt.rand_ket(2)
        states_piecewise = get_states_from_prop(P, psi0, 'piecewise')
        states_total = get_states_from_prop(Q[1:], psi0, 'total')
        self.assertArrayAlmostEqual(states_piecewise, states_total)

    def test_plot_bloch_vector_evolution(self):
        two_qubit_pulse = ff.PulseSequence(
            [[qt.tensor(qt.sigmax(), qt.sigmax()), [np.pi/2]]],
            [[qt.tensor(qt.sigmax(), qt.sigmax()), [1]]],
            [1]
        )

        # Call with default args
        b = plot_bloch_vector_evolution(simple_pulse)
        # Call with custom args
        b = init_bloch_sphere(background=True)
        b = plot_bloch_vector_evolution(simple_pulse, psi0=qt.basis(2, 1), b=b,
                                        n_samples=50, show=False,
                                        return_Bloch=True)

        b = plot_bloch_vector_evolution(complicated_pulse)

        # Check exceptions being raised
        with self.assertRaises(ValueError):
            plot_bloch_vector_evolution(two_qubit_pulse)

        plt.close('all')

    def test_plot_pulse_train(self):
        # Call with default args
        fig, ax, leg = plot_pulse_train(simple_pulse)

        # Call with no axes but figure
        fig = plt.figure()
        fig, ax, leg = plot_pulse_train(simple_pulse, fig=fig)

        # Call with axes but no figure
        fig, ax, leg = plot_pulse_train(simple_pulse, axes=ax)

        # Call with custom args
        c_oper_identifiers = sample(
            complicated_pulse.c_oper_identifiers.tolist(), randint(2, 4)
        )

        fig, ax = plt.subplots()
        fig, ax, leg = plot_pulse_train(complicated_pulse,
                                        c_oper_identifiers=c_oper_identifiers,
                                        fig=fig, axes=ax)

        # invalid identifier
        with self.assertRaises(ValueError):
            plot_pulse_train(complicated_pulse, c_oper_identifiers=['foo'],
                             fig=fig, axes=ax)

        # Test various keyword args for matplotlib
        plot_kw = {'linewidth': 1}
        subplot_kw = {'facecolor': 'r'}
        gridspec_kw = {'hspace': 0.2,
                       'wspace': 0.1}
        figure_kw = {'num': 1}

        fig, ax, leg = plot_pulse_train(simple_pulse, plot_kw=plot_kw,
                                        subplot_kw=subplot_kw,
                                        gridspec_kw=gridspec_kw, **figure_kw)

        plt.close('all')

    def test_plot_filter_function(self):
        # Call with default args
        simple_pulse.cleanup('all')
        fig, ax, leg = plot_filter_function(simple_pulse)

        # Call with no axes but figure
        fig = plt.figure()
        fig, ax, leg = plot_filter_function(simple_pulse, fig=fig)

        # Call with axes but no figure
        fig, ax, leg = plot_filter_function(simple_pulse, axes=ax)

        # Non-default args
        n_oper_identifiers = sample(
            complicated_pulse.n_oper_identifiers.tolist(), randint(2, 4)
        )

        fig, ax = plt.subplots()
        omega = ff.util.get_sample_frequencies(complicated_pulse, n_samples=50,
                                               spacing='log')
        fig, ax, leg = plot_filter_function(
            complicated_pulse, omega=omega,
            n_oper_identifiers=n_oper_identifiers,
            fig=fig, axes=ax, omega_in_units_of_tau=False
        )

        # invalid identifier
        with self.assertRaises(ValueError):
            plot_filter_function(complicated_pulse, n_oper_identifiers=['foo'],
                                 fig=fig, axes=ax)

        # Test different axis scales
        scales = ('linear', 'log')
        for xscale in scales:
            for yscale in scales:
                fig, ax, leg = plot_filter_function(simple_pulse,
                                                    xscale=xscale,
                                                    yscale=yscale)

        # Test various keyword args for matplotlib
        plot_kw = {'linewidth': 1}
        subplot_kw = {'facecolor': 'r'}
        gridspec_kw = {'hspace': 0.2,
                       'wspace': 0.1}
        figure_kw = {'num': 1}
        fig, ax, leg = plot_filter_function(simple_pulse, plot_kw=plot_kw,
                                            subplot_kw=subplot_kw,
                                            gridspec_kw=gridspec_kw,
                                            **figure_kw)

        plt.close('all')

    @pytest.mark.filterwarnings('ignore::UserWarning')  # tight_layout warning
    def test_plot_pulse_correlation_filter_function(self):
        omega = np.linspace(-1, 1, 50)
        concatenated_simple_pulse = ff.concatenate(
            (simple_pulse, simple_pulse),
            calc_pulse_correlation_ff=True,
            omega=omega
        )
        concatenated_complicated_pulse = ff.concatenate(
            (complicated_pulse, complicated_pulse),
            calc_pulse_correlation_ff=True,
            omega=omega
        )

        # Exception if not pulse correl. FF is cached
        with self.assertRaises(ff.util.CalculationError):
            plot_pulse_correlation_filter_function(simple_pulse)

        # Call with default args
        fig, ax, leg = plot_pulse_correlation_filter_function(
            concatenated_simple_pulse
        )

        # Non-default args
        n_oper_identifiers = sample(
            complicated_pulse.n_oper_identifiers.tolist(), randint(2, 4)
        )

        fig, ax = plt.subplots()
        omega = np.linspace(-10, 10, 50)
        fig, ax, leg = plot_pulse_correlation_filter_function(
            concatenated_complicated_pulse, omega=omega,
            n_oper_identifiers=n_oper_identifiers, fig=fig,
            omega_in_units_of_tau=False
        )

        # invalid identifiers
        with self.assertRaises(ValueError):
            plot_pulse_correlation_filter_function(
                concatenated_complicated_pulse,
                n_oper_identifiers=['foo'], fig=fig, axes=ax
            )

        # Test different axis scales
        scales = ('linear', 'log')
        for xscale in scales:
            for yscale in scales:
                fig, ax, leg = plot_pulse_correlation_filter_function(
                    concatenated_simple_pulse, xscale=xscale, yscale=yscale
                )

        # Test various keyword args for matplotlib
        plot_kw = {'linewidth': 1}
        subplot_kw = {'facecolor': 'r'}
        gridspec_kw = {'hspace': 0.2,
                       'wspace': 0.1}
        figure_kw = {'num': 1}
        fig, ax, leg = plot_pulse_correlation_filter_function(
            concatenated_simple_pulse, plot_kw=plot_kw, subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw, **figure_kw
        )

        plt.close('all')

    # Ignore deprecation warning caused by qutip
    @pytest.mark.filterwarnings('ignore::PendingDeprecationWarning')
    def test_plot_error_transfer_matrix(self):
        omega = ff.util.get_sample_frequencies(simple_pulse)
        S = 1e-4*np.sin(omega)/omega

        # Test calling with pulse, spectrum, omega
        fig, grid = plot_error_transfer_matrix(simple_pulse, S, omega,
                                               colorscale='linear')
        fig, grid = plot_error_transfer_matrix(simple_pulse, S, omega, fig=fig)
        fig, grid = plot_error_transfer_matrix(simple_pulse, S, omega,
                                               grid=grid)

        # Test calling with precomputed transfer matrix
        U = ff.error_transfer_matrix(simple_pulse, S, omega)
        fig, grid = plot_error_transfer_matrix(U=U)

        # Test calling with precomputed transfer matrix and pulse
        U = ff.error_transfer_matrix(simple_pulse, S, omega)
        fig, grid = plot_error_transfer_matrix(simple_pulse, U=U)

        # Test calling with precomputed transfer matrix of ndim == 2
        U = ff.error_transfer_matrix(simple_pulse, S, omega)
        fig, grid = plot_error_transfer_matrix(U=U[0])

        # Log colorscale
        fig, grid = plot_error_transfer_matrix(U=U, colorscale='log')

        # Non-default args
        n_oper_inds = sample(range(len(complicated_pulse.n_opers)),
                             randint(2, 4))
        n_oper_identifiers = complicated_pulse.n_oper_identifiers[n_oper_inds]

        basis_labels = []
        for i in range(4):
            basis_labels.append(string.ascii_uppercase[randint(0, 26)])

        omega = ff.util.get_sample_frequencies(complicated_pulse, n_samples=50,
                                               spacing='log')
        S = np.exp(-omega**2)
        U = ff.error_transfer_matrix(complicated_pulse, S, omega)
        fig, grid = plot_error_transfer_matrix(
            complicated_pulse, S=S, omega=omega,
            n_oper_identifiers=n_oper_identifiers, basis_labels=basis_labels,
            basis_labelsize=4, linthresh=1e-4, cmap=matplotlib.cm.jet
        )
        fig, grid = plot_error_transfer_matrix(
            U=U[n_oper_inds], n_oper_identifiers=n_oper_identifiers,
            basis_labels=basis_labels, basis_labelsize=4, linthresh=1e-4,
            cmap=matplotlib.cm.jet
        )

        # neither U nor all of pulse, S, omega given
        with self.assertRaises(ValueError):
            plot_error_transfer_matrix(complicated_pulse, S)

        # invalid identifiers
        with self.assertRaises(ValueError):
            plot_error_transfer_matrix(complicated_pulse, S, omega,
                                       n_oper_identifiers=['foo'])

        # number of basis_labels not correct
        with self.assertRaises(ValueError):
            plot_error_transfer_matrix(complicated_pulse, S, omega,
                                       basis_labels=basis_labels[:2])

        # grid too small
        with self.assertRaises(ValueError):
            plot_error_transfer_matrix(complicated_pulse, S, omega,
                                       grid=grid[:1])

        # Test various keyword args for matplotlib for the two-qubit pulse
        S = np.tile(S, (6, 6, 1))
        grid_kw = {'axes_pad': 0.1}
        imshow_kw = {'interpolation': 'bilinear'}
        figure_kw = {'num': 1}
        fig, ax = plot_error_transfer_matrix(two_qubit_pulse, S, omega,
                                             imshow_kw=imshow_kw,
                                             grid_kw=grid_kw,
                                             **figure_kw)

        plt.close('all')

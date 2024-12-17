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
This module provides various plotting functions.

Functions
---------
:func:`plot_bloch_vector_evolution`
    Plot the evolution of the Bloch vector on a QuTiP-generated Bloch
    sphere
:func:`plot_filter_function`
    Plot the filter function of a given ``PulseSequence``
:func:`plot_infidelity_convergence`
    Helper function called by
    :func:`~filter_functions.pulse_sequence.infidelity` to plot the
    convergence of the infidelity
:func:`plot_pulse_correlation_filter_function`
    Plot the pulse correlation filter function of a given
    ``PulseSequence``
:func:`plot_pulse_train`
    Plot the pulse train of a given ``PulseSequence``
:func:`plot_cumulant_function`
    Plot the cumulant function of a ``PulseSequence`` for a given
    spectrum as an image.

"""
from packaging import version
from typing import Optional, Sequence, Union
from unittest import mock
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, collections, colors, lines
from mpl_toolkits import axes_grid1
from numpy import ndarray

from . import numeric, util
from .types import (Axes, Coefficients, Colormap, Cycler, Figure, FigureAxes, FigureAxesLegend,
                    FigureGrid, Grid, Operator, State)

__all__ = ['plot_cumulant_function', 'plot_infidelity_convergence', 'plot_filter_function',
           'plot_pulse_correlation_filter_function', 'plot_pulse_train']

try:
    import qutip as qt
    __all__.append('plot_bloch_vector_evolution')
except ImportError:
    warn('Qutip not installed. plot_bloch_vector_evolution() is not available')
    qt = mock.Mock()


def _make_str_tex_compatible(s: str) -> str:
    """Escape incompatible characters in strings passed to TeX."""
    if not plt.rcParams['text.usetex']:
        return s

    s = str(s)
    incompatible = ('_',)
    for char in incompatible:
        locs = [i for i, c in enumerate(s) if c == char]
        # Loop backwards so as not to change locs when modifying s
        for loc in locs[::-1]:
            # Check if already escaped or math environment, if not add escape character
            if not s[loc-1:].startswith('\\') and not s.count('$', loc) % 2:
                s = s[:loc] + '\\' + s[loc:]

    return s


def _import_qutip_or_raise():
    try:
        import qutip as qt
    except ImportError as err:
        raise RuntimeError('Requirements not fulfilled. Please install Qutip') from err
    return qt


def get_bloch_vector(states: Sequence[State]) -> ndarray:
    """Get the Bloch vector from quantum states."""
    qt = _import_qutip_or_raise()

    if isinstance(states[0], qt.Qobj):
        a = np.empty((3, len(states)))
        X, Y, Z = qt.sigmax(), qt.sigmay(), qt.sigmaz()
        for i, state in enumerate(states):
            a[:, i] = [qt.expect(X, state),
                       qt.expect(Y, state),
                       qt.expect(Z, state)]
    else:
        a = np.einsum('...ij,kil,...lm->k...', np.conj(states), util.paulis[1:], states)

    return a.real


def init_bloch_sphere(**bloch_kwargs) -> qt.Bloch:
    """A helper function to create a Bloch instance with a default viewing
    angle and axis labels."""
    qt = _import_qutip_or_raise()

    bloch_kwargs.setdefault('view', [-150, 30])
    b = qt.Bloch(**bloch_kwargs)

    # https://github.com/qutip/qutip/issues/1385
    if hasattr(b.axes, 'set_box_aspect'):
        b.axes.set_box_aspect([1, 1, 1])

    b.xlabel = [r'$|+\rangle$', '']
    b.ylabel = [r'$|+_i\rangle$', '']
    return b


def get_states_from_prop(U: Sequence[Operator], psi0: Optional[State] = None) -> ndarray:
    r"""
    Get the the quantum state at time t from the propagator and the
    inital state:

    .. math::

        |\psi(t)\rangle = U(t, 0)|\psi(0)\rangle

    """
    if psi0 is None:
        # default to |0>
        psi0 = np.c_[1:-1:-1]
    elif hasattr(psi0, 'full'):
        # qutip.Qobj
        psi0 = psi0.full()

    if psi0.shape[-2:] != (2, 1):
        raise ValueError('Initial state should be shape (..., 2, 1)')

    return U @ psi0


def plot_bloch_vector_evolution(
        pulse: 'PulseSequence',
        psi0: Optional[State] = None,
        b: Optional[qt.Bloch] = None,
        n_samples: Optional[int] = None,
        cmap: Colormap = 'winter',
        add_cbar: bool = False,
        show: bool = True,
        return_Bloch: bool = False,
        cbar_kwargs: Optional[dict] = None,
        **bloch_kwargs
) -> Union[None, qt.Bloch]:
    r"""
    Plot the evolution of the Bloch vector under the given pulse
    sequence.

    Parameters
    ----------
    pulse: PulseSequence
        The PulseSequence instance whose control Hamiltonian determines
        the time evolution of the Bloch vector.
    psi0: Qobj or array_like, optional
        The initial state before the pulse is applied. Defaults to
        :math:`|0\rangle`.
    b: qutip.Bloch, optional
        If given, the QuTiP Bloch instance on which to plot the time
        evolution.
    n_samples: int, optional
        The number of time points to be sampled.
    cmap: matplotlib colormap, optional
        The colormap for the trajectory. Requires ``matplotlib >= 3.3.0``.
    add_cbar: bool, optional
        Add a colorbar encoding the time evolution to the figure.
        Default is false.
    show: bool, optional
        Whether to show the sphere (by calling :code:`b.make_sphere()`).
    return_Bloch: bool, optional
        Whether to return the :class:`qutip.bloch.Bloch` instance
    cbar_kwargs: dict, optional
        A dictionary with keyword arguments to be fed into the
        colorbar constructor (if ``add_cbar == True``).
    **bloch_kwargs: optional
        Keyword arguments to be fed into the qutip.Bloch constructor
        (if *b* not given).

    Returns
    -------
    b: qutip.Bloch
        The qutip.Bloch instance

    Raises
    ------
    ValueError
        If the pulse is for more than one qubit

    See Also
    --------
    qutip.bloch.Bloch: Qutip's Bloch sphere implementation.
    """
    # Raise an exception if not a one-qubit pulse
    if not pulse.d == 2:
        raise ValueError('Plotting Bloch sphere evolution only implemented for one-qubit case!')

    # Parse default arguments
    figsize = bloch_kwargs.pop('figsize', [5, 5])
    view = bloch_kwargs.pop('view', [-60, 30])
    if b is None:
        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(projection='3d', azim=view[0], elev=view[1])
        b = init_bloch_sphere(fig=fig, axes=axes, **bloch_kwargs)
    else:
        if b.fig is None:
            b.fig = plt.figure(figsize=figsize)
        if b.axes is None:
            b.axes = b.fig.add_subplot(projection='3d', azim=view[0], elev=view[1])

    if show:
        b.make_sphere()

    if n_samples is None:
        # At least 100, at most 5000 points, default 10 points per smallest
        # time interval
        n_samples = min(5000, max(10*int(pulse.tau/pulse.dt.min()), 100))

    times = np.linspace(pulse.t[0], pulse.tau, n_samples)
    propagators = pulse.propagator_at_arb_t(times)
    points = get_bloch_vector(get_states_from_prop(propagators, psi0))
    # Qutip convention: -x at +y, +y at +x
    copy = points.copy()
    points[0] = copy[1]
    points[1] = -copy[0]

    # Check the matplotlib version to see if we can draw a color gradient line. If not, draw sphere
    # after adding the points using the Bloch method. If yes, we apparently need to draw the sphere
    # before manually adding the line collection, otherwise there is some strange thing going on in
    # notebooks and the lines are not rendered.
    if version.parse(matplotlib.__version__) < version.parse('3.3.0'):
        # Colored trajectory not available.
        b.axes.plot(*points, color='b', alpha=0.75)
    else:
        points = points.T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        cmap = plt.get_cmap(cmap)
        segment_colors = cmap(np.linspace(0, 1, n_samples - 1))
        lc = collections.LineCollection(segments[:, :, :2], colors=segment_colors, alpha=0.75)
        b.axes.add_collection3d(lc, zdir='z', zs=segments[:, :, 2])

        if add_cbar:
            default_cbar_kwargs = dict(shrink=2/3, pad=0.05, label=r'$t$ ($\tau$)', ticks=[0, 1],
                                       ax=b.axes)
            cbar_kwargs = {**default_cbar_kwargs, **(cbar_kwargs or {})}
            b.fig.colorbar(cm.ScalarMappable(cmap=cmap), **cbar_kwargs)

    if return_Bloch:
        return b


def plot_pulse_train(
        pulse: 'PulseSequence',
        c_oper_identifiers: Optional[Sequence[int]] = None,
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        cycler: Optional[Cycler] = None,
        plot_kw: Optional[dict] = {},
        subplot_kw: Optional[dict] = None,
        gridspec_kw: Optional[dict] = None,
        **figure_kw
) -> FigureAxesLegend:
    """
    Plot the pulsetrain of the ``PulseSequence`` *pulse*.

    Parameters
    ----------
    pulse: PulseSequence
        The pulse sequence whose pulse train to plot.
    c_oper_identifiers: array_like, optional
        The identifiers of the control operators for which the pulse
        train should be plotted. All identifiers can be accessed via
        ``pulse.c_oper_identifiers``. Defaults to all.
    fig: matplotlib figure, optional
        A matplotlib figure instance to plot in
    axes: matplotlib axes, optional
        A matplotlib axes instance to use for plotting.
    cycler: cycler.Cycler, optional
        A Cycler instance used to set the style cycle if multiple lines
        are to be drawn
    plot_kw: dict, optional
        Dictionary with keyword arguments passed to the plot function
    subplot_kw: dict, optional
        Dictionary with keyword arguments passed to the subplots
        constructor
    gridspec_kw: dict, optional
        Dictionary with keyword arguments passed to the gridspec
        constructor
    figure_kw: optional
        Keyword argument dictionaries that are fed into the
        :func:`matplotlib.pyplot.subplots` function if no *fig* instance
        is specified.

    Returns
    -------
    fig: matplotlib figure
        The matplotlib figure instance used for plotting.
    axes: matplotlib axes
        The matplotlib axes instance used for plotting.
    legend: matplotlib legend
        The matplotlib legend instance in the plot.

    Raises
    ------
    ValueError
        If an invalid number of c_oper_labels were given
    """
    c_oper_inds = util.get_indices_from_identifiers(pulse.c_oper_identifiers, c_oper_identifiers)
    c_oper_identifiers = pulse.c_oper_identifiers[c_oper_inds]

    if fig is None and axes is None:
        fig, axes = plt.subplots(subplot_kw=subplot_kw,
                                 gridspec_kw=gridspec_kw,
                                 **figure_kw)
    elif axes is None and fig is not None:
        subplot_kw = subplot_kw or {}
        axes = fig.add_subplot(111, **subplot_kw)
    elif fig is None and axes is not None:
        fig = axes.figure

    if cycler is not None:
        axes.set_prop_cycle(cycler)

    handles = []
    for i, c_coeffs in enumerate(pulse.c_coeffs[tuple(c_oper_inds), ...]):
        coeffs = np.insert(c_coeffs, 0, c_coeffs[0])
        handles += axes.step(pulse.t, coeffs,
                             label=_make_str_tex_compatible(c_oper_identifiers[i]), **plot_kw)

    axes.set_xlim(pulse.t[0], pulse.tau)
    axes.set_xlabel(r'$t$ / a.u.')
    axes.set_ylabel(r'Control parameter / a.u.')
    axes.grid(True)
    legend = axes.legend(framealpha=1)

    return fig, axes, legend


def plot_filter_function(
        pulse: 'PulseSequence',
        omega: Optional[Coefficients] = None,
        n_oper_identifiers: Optional[Sequence[int]] = None,
        fig: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        xscale: str = 'log',
        yscale: str = 'linear',
        omega_in_units_of_tau: bool = False,
        cycler: Optional[Cycler] = None,
        plot_kw: dict = {},
        subplot_kw: Optional[dict] = None,
        gridspec_kw: Optional[dict] = None,
        **figure_kw
) -> FigureAxesLegend:
    r"""
    Plot the fidelity filter function(s) of the given PulseSequence for
    positive frequencies. As of now only the diagonal elements of
    :math:`F_{\alpha\beta}` are implemented, i.e. the filter functions
    corresponding to uncorrelated noise sources.

    Parameters
    ----------
    pulse: PulseSequence
        The pulse sequence whose filter function to plot.
    omega: array_like, optional
        The frequencies at which to evaluate the filter function. If not
        given, the pulse sequence's omega attribute is used (if set) or
        sensible values are chosen automatically (if ``None``)
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which the filter
        function should be plotted. All identifiers can be accessed via
        ``pulse.n_oper_identifiers``. Defaults to all.
    fig: matplotlib figure, optional
        A matplotlib figure instance to plot in
    axes: matplotlib axes, optional
        A matplotlib axes instance to use for plotting.
    xscale: str, optional
        x-axis scaling. One of ('linear', 'log').
    yscale: str, optional
        y-axis scaling. One of ('linear', 'log').
    omega_in_units_of_tau: bool, optional
        Plot :math:`\omega\tau` or just :math:`\omega` on x-axis.
    cycler: cycler.Cycler, optional
        A Cycler instance used to set the style cycle if multiple lines
        are to be drawn
    plot_kw: dict, optional
        Dictionary with keyword arguments passed to the plot function
    subplot_kw: dict, optional
        Dictionary with keyword arguments passed to the subplots
        constructor
    gridspec_kw: dict, optional
        Dictionary with keyword arguments passed to the gridspec
        constructor
    figure_kw: optional
        Keyword argument dictionaries that are fed into the
        :func:`matplotlib.pyplot.subplots` function if no *fig* instance
        is specified.

    Returns
    -------
    fig: matplotlib figure
        The matplotlib figure instance used for plotting.
    axes: matplotlib axes
        The matplotlib axes instance used for plotting.
    legend: matplotlib legend
        The matplotlib legend instance in the plot.

    Raises
    ------
    ValueError
        If an invalid number of n_oper_labels were given
    """
    if omega is None:
        if pulse.omega is None:
            omega = util.get_sample_frequencies(pulse, spacing=xscale)
        else:
            omega = pulse.omega

    n_oper_inds = util.get_indices_from_identifiers(pulse.n_oper_identifiers, n_oper_identifiers)
    n_oper_identifiers = pulse.n_oper_identifiers[n_oper_inds]

    if fig is None and axes is None:
        fig, axes = plt.subplots(subplot_kw=subplot_kw,
                                 gridspec_kw=gridspec_kw,
                                 **figure_kw)
    elif axes is None and fig is not None:
        subplot_kw = subplot_kw or {}
        axes = fig.add_subplot(111, **subplot_kw)
    elif fig is None and axes is not None:
        fig = axes.figure

    if cycler is not None:
        axes.set_prop_cycle(cycler)

    if omega_in_units_of_tau:
        tau = np.ptp(pulse.t)
        z = omega*tau
        xlabel = r'$\omega\tau$'
    else:
        z = omega
        xlabel = r'$\omega$'

    diag_idx = np.arange(len(pulse.n_opers))
    filter_function = pulse.get_filter_function(omega)[diag_idx, diag_idx].real

    handles = []
    for i, ind in enumerate(n_oper_inds):
        handles += axes.plot(z, filter_function[ind],
                             label=_make_str_tex_compatible(n_oper_identifiers[i]),
                             **plot_kw)

    # Set the axis scales
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    if xscale != 'linear':
        z_min_idx = (z > 0).nonzero()[0][0]
    else:
        z_min_idx = (z >= 0).nonzero()[0][0]

    if yscale == 'linear':
        axes.set_ylim(bottom=0)

    axes.set_xlim(z[z_min_idx], max(z))
    axes.set_xlabel(xlabel)
    axes.set_ylabel(r'$F(\omega)$')
    axes.grid(True)
    legend = axes.legend()

    return fig, axes, legend


def plot_pulse_correlation_filter_function(
        pulse: 'PulseSequence',
        n_oper_identifiers: Optional[Sequence[int]] = None,
        fig: Optional[Figure] = None,
        xscale: str = 'log',
        yscale: str = 'linear',
        omega_in_units_of_tau: bool = True,
        cycler: Optional[Cycler] = None,
        plot_kw: dict = {},
        subplot_kw: Optional[dict] = None,
        gridspec_kw: Optional[dict] = None,
        **figure_kw
) -> FigureAxesLegend:
    r"""
    Plot the fidelity pulse correlation filter functions of the given
    PulseSequence if they were computed during concatenation for
    positive frequencies.

    Returns a figure with *n* by *n* subplots where *n* is the number of
    pulses that were concatenated. As of now only the diagonal elements
    of :math:`F_{\alpha\beta}` are implemented, i.e. the filter
    functions corresponding to uncorrelated noise sources.

    Parameters
    ----------
    pulse: PulseSequence
        The pulse sequence whose filter function to plot.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which the filter
        function should be plotted. All identifiers can be accessed via
        ``pulse.n_oper_identifiers``. Defaults to all.
    fig: matplotlib figure, optional
        A matplotlib figure instance to plot in
    xscale: str, optional
        x-axis scaling. One of ('linear', 'log').
    yscale: str, optional
        y-axis scaling. One of ('linear', 'log').
    omega_in_units_of_tau: bool, optional
        Plot :math:`\omega\tau` or just :math:`\omega` on x-axis.
    cycler: cycler.Cycler, optional
        A Cycler instance used to set the style cycle if multiple lines
        are to be drawn in one subplot. Used for all subplots.
    plot_kw: dict, optional
        Dictionary with keyword arguments passed to the plot function
    subplot_kw: dict, optional
        Dictionary with keyword arguments passed to the subplots
        constructor
    gridspec_kw: dict, optional
        Dictionary with keyword arguments passed to the gridspec
        constructor
    figure_kw: optional
        Keyword argument dictionaries that are fed into the
        :func:`matplotlib.pyplot.subplots` function if no *fig* instance
        if specified.

    Returns
    -------
    fig: matplotlib figure
        The matplotlib figure instance used for plotting.
    axes: matplotlib axes
        The matplotlib axes instances used for plotting.
    legend: matplotlib legend
        The matplotlib legend instance in the plot.

    Raises
    ------
    CalculationError
        If the pulse correlation filter function was not computed during
        concatenation.
    """
    n_oper_inds = util.get_indices_from_identifiers(pulse.n_oper_identifiers, n_oper_identifiers)
    n_oper_identifiers = pulse.n_oper_identifiers[n_oper_inds]
    diag_idx = np.arange(len(pulse.n_opers))
    F_pc = pulse.get_pulse_correlation_filter_function()
    F_pc = F_pc[:, :, diag_idx, diag_idx]
    n = F_pc.shape[0]

    if fig is None:
        fig, axes = plt.subplots(n, n, sharex=True, subplot_kw=subplot_kw,
                                 gridspec_kw=gridspec_kw, **figure_kw)

    else:
        subplot_kw = subplot_kw or {}
        axes = np.empty((n, n), dtype='O')
        axes[0, 0] = fig.add_subplot(n, n, 1, **subplot_kw)
        for row in range(n):
            for col in range(n):
                if not (row == 0 and col == 0):
                    index = np.ravel_multi_index([row, col], dims=(n, n)) + 1
                    axes[row, col] = fig.add_subplot(n, n, index, sharex=axes[0, 0], **subplot_kw)

    omega = pulse.omega
    if omega_in_units_of_tau:
        tau = np.ptp(pulse.t)
        z = omega*tau
        xlabel = r'$\omega\tau$'
    else:
        z = omega
        xlabel = r'$\omega$'

    transparent_line = lines.Line2D([], [], alpha=0)
    solid_line = lines.Line2D([], [], color='gray', linestyle='-')
    dashed_line = lines.Line2D([], [], color='gray', linestyle='--')
    for i in range(n):
        for j in range(n):
            if cycler is not None:
                axes[i, j].set_prop_cycle(cycler)

            handles = []
            for k, ind in enumerate(n_oper_inds):
                handles += axes[i, j].plot(z, F_pc[i, j, ind].real,
                                           label=_make_str_tex_compatible(n_oper_identifiers[k]),
                                           **plot_kw)
                if i != j:
                    axes[i, j].plot(z, F_pc[i, j, ind].imag, linestyle='--',
                                    color=handles[-1].get_color(), **plot_kw)

            # Set the axis scales
            axes[i, j].set_yscale(yscale)
            axes[i, j].set_title(rf'$F^{{({i}{j})}}(\omega)$')
            if i != n-1:
                # Hide the ticklabels on all but the lowest row
                axes[i, j].axes.xaxis.set_ticklabels([])
            else:
                axes[i, j].set_xlabel(xlabel)

            if i == 0 and j == n-1:
                handles += [transparent_line, solid_line, dashed_line]
                labels = ([_make_str_tex_compatible(n) for n in n_oper_identifiers]
                          + ['', r'$Re$', r'$Im$'])
                legend = axes[i, j].legend(handles=handles, labels=labels,
                                           bbox_to_anchor=(1.05, 1), loc=2,
                                           borderaxespad=0., frameon=False)

    axes[i, j].set_xscale(xscale)
    if xscale == 'log':
        z_min_idx = (z > 0).nonzero()[0][0]
    else:
        z_min_idx = (z >= 0).nonzero()[0][0]

    axes[i, j].set_xlim(z[z_min_idx], max(z))
    fig.tight_layout(h_pad=0.1)

    return fig, axes, legend


def plot_infidelity_convergence(n_samples: Sequence[int], infids: Sequence[float],
                                axes: Optional[Axes] = None) -> FigureAxes:
    """
    Plot the convergence of the infidelity integral. The function
    arguments are those returned by
    :func:`~filter_functions.numeric.infidelity` with the
    *test_convergence* flag set to ``True``.

    Parameters
    ----------
    n_samples: array_like
        Array with the number of samples at which the integral was
        evaluated
    infids: array_like, shape (n_samples, [n_oper_inds, optional])
        Array with the calculated infidelities for each noise operator
        on the second axis or the second axis already traced out.
    axes: sequence of two matplotlib axes, optional
        Two axes that the result is plotted in.

    Returns
    -------
    fig: matplotlib figure
        The matplotlib figure instance used for plotting.
    axes: matplotlib axes
        The matplotlib axes instances used for plotting.

    """
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True)
    else:
        fig = axes[0].get_figure()

    axes[1].set_xlabel(r'$n_\omega$')
    axes[0].set_ylabel(r'$\mathcal{I}$')
    axes[1].set_ylabel(r'$|\Delta\mathcal{I}|/\mathcal{I}$ (%)')
    axes[0].set_xlim(min(n_samples), max(n_samples))
    axes[0].grid()
    axes[1].grid()

    axes[0].plot(n_samples, infids, 'o-')
    axes[1].semilogy(n_samples, np.abs(np.gradient(infids, axis=0))/infids*100, 'o-')

    return fig, axes


@util.parse_optional_parameters(colorscale=('linear', 'log'))
def plot_cumulant_function(
        pulse: Optional['PulseSequence'] = None,
        spectrum: Optional[ndarray] = None,
        omega: Optional[Coefficients] = None,
        cumulant_function: Optional[ndarray] = None,
        n_oper_identifiers: Optional[Sequence[int]] = None,
        second_order: bool = False,
        colorscale: str = 'linear',
        linthresh: Optional[float] = None,
        basis_labels: Optional[Sequence[str]] = None,
        basis_labelsize: Optional[int] = None,
        cbar_label: str = 'Cumulant Function',
        cbar_labelsize: Optional[int] = None,
        fig: Optional[Figure] = None,
        grid: Optional[Grid] = None,
        cmap: Optional[Colormap] = None,
        grid_kw: Optional[dict] = None,
        cbar_kw: Optional[dict] = None,
        imshow_kw: Optional[dict] = None,
        **figure_kw
) -> FigureGrid:
    r"""Plot the cumulant function for a given noise spectrum as an image.

    The cumulant function generates the error transfer matrix
    :math:`\tilde{\mathcal{U}}` exactly for Gaussian noise and to second
    order for non-Gaussian noise.

    The function may be called with either a ``PulseSequence``, a
    spectrum, and a list of frequencies in which case the cumulant
    function is calculated for those parameters, or with a precomputed
    cumulant function.

    As of now, only auto-correlated spectra are implemented.

    Parameters
    ----------
    pulse: PulseSequence
        The pulse sequence.
    spectrum: ndarray
        The two-sided noise spectrum.
    omega: array_like
        The frequencies for which to evaluate the error transfer matrix.
    cumulant_function: ndarray, shape (n_nops, d**2, d**2)
        A precomputed cumulant function. If given, *pulse*, *spectrum*,
        *omega* are not required.
    n_oper_identifiers: array_like, optional
        The identifiers of the noise operators for which the cumulant
        function should be plotted. All identifiers can be accessed via
        ``pulse.n_oper_identifiers``. Defaults to all.
    second_order: bool, optional
        Also take into account the frequency shifts :math:`\Delta` that
        correspond to second order Magnus expansion and constitute unitary
        terms. Default ``False``.
    colorscale: str, optional
        The scale of the color code ('linear' or 'log' (default))
    linthresh: float, optional
        The threshold below which the colorscale will be linear (only
        for 'log') colorscale
    basis_labels: array_like (str), optional
        Custom labels for the elements of the cumulant function (the
        basis elements). Grabbed from the basis by default.
    basis_labelsize: int, optional
        The size in points for the basis labels.
    cbar_label: str, optional
        The label for the colorbar. Default: 'Cumulant Function'.
    cbar_labelsize: int, optional
        The size in points for the colorbar label.
    fig: matplotlib figure, optional
        A matplotlib figure instance to plot in
    grid: matplotlib ImageGrid, optional
        An ImageGrid instance to use for plotting.
    cmap: matplotlib colormap, optional
        The colormap for the matrix plot.
    grid_kw: dict, optional
        Dictionary with keyword arguments passed to the ImageGrid
        constructor.
    cbar_kw: dict, optional
        Dictionary with keyword arguments passed to the colorbar
        constructor.
    imshow_kw: dict, optional
        Dictionary with keyword arguments passed to imshow.
    figure_kw: optional
        Keyword argument dictionaries that are fed into the
        :func:`matplotlib.pyplot.figure` function if no *fig* instance
        is specified.

    Returns
    -------
    fig: matplotlib figure
        The matplotlib figure instance used for plotting.
    grid: matplotlib ImageGrid
        The ImageGrid instance used for plotting.
    """
    K = cumulant_function
    if K is not None:
        if K.ndim == 2:
            K = np.array([K])

        n_oper_inds = np.arange(len(K))
        if n_oper_identifiers is None:
            if pulse is not None and len(pulse.n_oper_identifiers) == len(K):
                n_oper_identifiers = pulse.n_oper_identifiers
            else:
                n_oper_identifiers = [f'$B_{{{i}}}$' for i in range(len(n_oper_inds))]
        else:
            if len(n_oper_identifiers) != len(K):
                raise ValueError(
                    'Both precomputed cumulant function and n_oper_identifiers '
                    + f'given but not same len: {len(K)} != {len(n_oper_identifiers)}'
                )

    else:
        if pulse is None or spectrum is None or omega is None:
            raise ValueError('Require either precomputed cumulant function '
                             + 'or pulse, spectrum, and omega as arguments.')

        n_oper_inds = util.get_indices_from_identifiers(pulse.n_oper_identifiers,
                                                        n_oper_identifiers)
        n_oper_identifiers = pulse.n_oper_identifiers[n_oper_inds]
        K = numeric.calculate_cumulant_function(pulse, spectrum, omega, n_oper_identifiers,
                                                'total', second_order)
        if K.ndim == 4:
            # Only autocorrelated noise supported
            K = K[tuple(n_oper_inds), tuple(n_oper_inds)]

    # Only autocorrelated noise implemented for now, ie K is real
    K = K.real

    if basis_labels is None:
        if pulse is not None:
            basis_labels = pulse.basis.labels
        else:
            basis_labels = [f'$C_{{{i}}}$' for i in range(K.shape[-1])]
    else:
        if basis_labels is not None and len(basis_labels) != K.shape[-1]:
            raise ValueError('Invalid number of basis_labels given')

        basis_labels = [_make_str_tex_compatible(bl) for bl in basis_labels]

    if grid is None:
        aspect_ratio = 2/3
        n_rows = int(np.round(np.sqrt(aspect_ratio*len(n_oper_inds))))
        n_cols = int(np.ceil(len(n_oper_inds)/n_rows))
        grid_kw = grid_kw or {}
        grid_kw.setdefault('rect', 111)
        grid_kw.setdefault('nrows_ncols', (n_rows, n_cols))
        grid_kw.setdefault('axes_pad', 0.3)
        grid_kw.setdefault('label_mode', 'L')
        grid_kw.setdefault('share_all', True)
        grid_kw.setdefault('direction', 'row')
        grid_kw.setdefault('cbar_mode', 'single')
        grid_kw.setdefault('cbar_pad', 0.3)
        if fig is None:
            figsize = figure_kw.pop('figsize', (8*n_cols, 6*n_rows))
            fig = plt.figure(figsize=figsize, **figure_kw)

        grid = axes_grid1.ImageGrid(fig, **grid_kw)
    else:
        if len(grid) != len(n_oper_inds):
            raise ValueError('Size of supplied ImageGrid instance does not '
                             + 'match the number of n_oper_identifiers given!')

        fig = grid[0].get_figure()

    # Parse default arguments
    if cmap is not None:
        plt.get_cmap(cmap)
    else:
        cmap = plt.get_cmap('RdBu')

    Kmax = np.abs(K).max()
    Kmin = -Kmax
    if colorscale == 'log':
        linthresh = np.abs(K).mean()/10 if linthresh is None else linthresh
        norm = colors.SymLogNorm(linthresh=linthresh, vmin=Kmin, vmax=Kmax)
    else:
        # colorscale == 'linear'
        norm = colors.Normalize(vmin=Kmin, vmax=Kmax)

    imshow_kw = imshow_kw or {}
    imshow_kw.setdefault('origin', 'upper')
    imshow_kw.setdefault('interpolation', 'nearest')
    imshow_kw.setdefault('cmap', cmap)
    imshow_kw.setdefault('norm', norm)

    basis_labelsize = basis_labelsize or 8
    cbar_labelsize = cbar_labelsize or plt.rcParams['axes.labelsize']

    # Draw the images
    for i, n_oper_identifier in enumerate(n_oper_identifiers):
        ax = grid[i]
        im = ax.imshow(K[i], **imshow_kw)
        ax.set_title(n_oper_identifier, loc='left')
        ax.set_xticks(np.arange(K.shape[-1]))
        ax.set_yticks(np.arange(K.shape[-1]))
        ax.set_xticklabels(basis_labels, fontsize=basis_labelsize, rotation='vertical')
        ax.set_yticklabels(basis_labels, fontsize=basis_labelsize)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # Set up the colorbar
    cbar_kw = cbar_kw or {}
    cbar_kw.setdefault('orientation', 'vertical')
    cbar = fig.colorbar(im, cax=grid.cbar_axes[0], **cbar_kw)
    cbar.set_label(_make_str_tex_compatible(cbar_label), fontsize=cbar_labelsize)

    return fig, grid

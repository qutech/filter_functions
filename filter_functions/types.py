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
Defines custom types for the package.
"""
from typing import Mapping, Optional, Sequence, Tuple, Union

from numpy import ndarray

try:
    import cycler
    from matplotlib import axes, colors, figure, legend
    from mpl_toolkits import axes_grid1

    Axes = axes.Axes
    Colormap = Union[colors.Colormap, str]
    Figure = figure.Figure
    Grid = axes_grid1.ImageGrid
    Legend = legend.Legend
    Cycler = cycler.Cycler
    FigureAxes = Tuple[Figure, Axes]
    FigureAxesLegend = Tuple[Figure, Axes, Legend]
    FigureGrid = Tuple[Figure, Grid]
except ImportError:
    pass

try:
    from qutip import Qobj

    State = Union[ndarray, Qobj]
    Operator = Union[ndarray, Qobj]
except ImportError:
    State = ndarray
    Operator = ndarray

Coefficients = Sequence[float]
Hamiltonian = Sequence[Sequence[Union[Operator, Coefficients]]]
PulseMapping = Sequence[Sequence[Union['PulseSequence',
                                       Union[Sequence[int], int],
                                       Optional[Mapping[str, str]]]]]

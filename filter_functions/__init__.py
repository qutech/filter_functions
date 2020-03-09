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
"""Package for efficient calculation of generalized filter functions"""

from . import analytic, basis, numeric, plotting, pulse_sequence, util
from .basis import Basis
from .numeric import (error_transfer_matrix, infidelity,
                      liouville_representation)
from .plotting import (
    plot_bloch_vector_evolution, plot_error_transfer_matrix,
    plot_filter_function, plot_pulse_correlation_filter_function,
    plot_pulse_train)
from .pulse_sequence import (PulseSequence, concatenate, concatenate_periodic,
                             extend, remap)

__all__ = ['Basis', 'PulseSequence', 'analytic', 'basis', 'concatenate',
           'concatenate_periodic', 'error_transfer_matrix', 'extend',
           'infidelity', 'liouville_representation', 'numeric',
           'plot_bloch_vector_evolution', 'plot_error_transfer_matrix',
           'plot_filter_function', 'plot_pulse_correlation_filter_function',
           'plot_pulse_train', 'plotting', 'pulse_sequence', 'remap', 'util']

__version__ = '0.2.4'
__license__ = 'GNU GPLv3+'
__author__ = 'Quantum Technology Group, RWTH Aachen University'

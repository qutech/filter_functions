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

from . import analytic, basis, gradient, numeric, pulse_sequence, superoperator, util
from .basis import Basis
from .gradient import infidelity_derivative
from .numeric import error_transfer_matrix, infidelity
from .pulse_sequence import PulseSequence, concatenate, concatenate_periodic, extend, remap
from .superoperator import liouville_representation

__all__ = ['Basis', 'PulseSequence', 'analytic', 'basis', 'concatenate', 'concatenate_periodic',
           'error_transfer_matrix', 'extend', 'infidelity', 'liouville_representation', 'numeric',
           'gradient', 'pulse_sequence', 'remap', 'util', 'superoperator', 'infidelity_derivative']


__version__ = '1.1.3'
__license__ = 'GNU GPLv3+'
__author__ = 'Quantum Technology Group, RWTH Aachen University'

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
This example implements the quantum Fourier transform (QFT) with Ising-type
Hamiltonians presented in [1]_.

References
----------
.. [1]:
    Ivanov, S. S., Johanning, M., & Wunderlich, C. (2015). Simplified
    implementation of the quantum Fourier transform with Ising-type
    Hamiltonians: Example with ion traps, 1(1), 1–7.
    Retrieved from http://arxiv.org/abs/1503.08806
"""
import filter_functions as ff
import numpy as np
import qutip as qt
from filter_functions import plotting
from qutip import qip
from qutip.qip.algorithms.qft import qft as qt_qft

# %% Define some functions


def R_k_pulse(k, theta, phi, N: int = 4, tau: float = 1):
    Id = qt.qeye(2)
    if N == 1:
        X = qt.sigmax()
        Y = qt.sigmay()
    else:
        X = [Id]*(N - 1)
        Y = [Id]*(N - 1)
        X.insert(k, qt.sigmax())
        Y.insert(k, qt.sigmay())
        X = qt.tensor(X)
        Y = qt.tensor(Y)

    H_c = [[X, [theta/2/tau*np.cos(phi)], 'I'*k + 'X' + 'I'*(N - k - 1)],
           [Y, [theta/2/tau*np.sin(phi)], 'I'*k + 'Y' + 'I'*(N - k - 1)]]
    H_n = [[X/np.sqrt(X.shape[0]), [1], 'I'*k + 'X' + 'I'*(N - k - 1)],
           [Y/np.sqrt(Y.shape[0]), [1], 'I'*k + 'Y' + 'I'*(N - k - 1)]]
    dt = [tau]

    return ff.PulseSequence(H_c, H_n, dt)


def T_I_pulse(N: int = 4, tau: float = 1):
    Id = qt.qeye(2)
    if N == 1:
        T = Id
        H_c = [[Id, [0], 'I']]
        H_n = [[T/np.sqrt(T.shape[0]), [1], 'I']]
    else:
        H_c = []
        H_n = []
        T = [Id]*(N - 1)
        T.insert(0, qt.sigmaz())
        for k in range(1, N+1):
            H_c.append([qt.tensor(T[-k+1:] + T[:-k+1]),
                        [np.pi/4*(1 - 2**(1 - k))/tau],
                        'I'*(k - 1) + 'Z' + 'I'*(N - k)])
            H_n.append([qt.tensor(T[-k+1:] + T[:-k+1])/np.sqrt(2**len(T)),
                        [1], 'I'*(k - 1) + 'Z' + 'I'*(N - k)])

    return ff.PulseSequence(H_c, H_n, [tau])


def T_F_pulse(N: int = 4, tau: float = 1):
    Id = qt.qeye(2)
    if N == 1:
        T = Id
        H_c = [[Id, [0], 'I']]
        H_n = [[T/np.sqrt(T.shape[0]), [1], 'I']]
    else:
        H_c = []
        H_n = []
        T = [Id]*(N - 1)
        T.insert(0, qt.sigmaz())
        for k in range(1, N+1):
            H_c.append([qt.tensor(T[-k+1:] + T[:-k+1]),
                        [np.pi/4*(1 - 2**(k - N))/tau],
                        'I'*(k - 1) + 'Z' + 'I'*(N - k)])
            H_n.append([qt.tensor(T[-k+1:] + T[:-k+1])/np.sqrt(2**len(T)),
                        [1], 'I'*(k - 1) + 'Z' + 'I'*(N - k)])

    return ff.PulseSequence(H_c, H_n, [tau])


def P_n_pulse(n, N: int = 4, tau: float = 1):
    Id = qt.qeye(2)
    H_c = []
    H_n = []
    for l in range(n+1, N+1):
        Z = [Id]*(N - 2)
        Z.insert(n-1, qt.sigmaz())
        Z.insert(l-1, qt.sigmaz())
        Z = qt.tensor(Z)
        identifier = ('I'*(n-1) + 'Z' + 'I'*(l - n - 1) + 'Z' + 'I'*(N - l))
        H_c.append([Z, [-np.pi/4*2**(n - l)/tau], identifier])
        H_n.append([Z/np.sqrt(Z.shape[0]), [1], identifier])

    return ff.PulseSequence(H_c, H_n, [tau])


def H_k_pulse(k, N: int = 4, tau: float = 1):
    return ff.concatenate([R_k_pulse(k, np.pi, 0, N, tau),
                           R_k_pulse(k, np.pi/2, -np.pi/2, N, tau)])


def QFT_pulse(N: int = 4, tau: float = 1):
    pulses = [T_I_pulse(N, tau)]
    for n in range(N-1):
        pulses.append(H_k_pulse(n, N, tau))
        pulses.append(P_n_pulse(n+1, N, tau))

    pulses.append(H_k_pulse(N-1, N, tau))
    pulses.append(T_F_pulse(N, tau))

    return ff.concatenate(pulses, calc_pulse_correlation_FF=False)


# %% Get a 4-qubit QFT and plot the filter function
omega = np.logspace(-2, 2, 500)

N = 4
QFT = QFT_pulse(N)

# Check the pulse produces the correct propagator after swapping the qubits
swaps = [qip.operations.swap(N, [i, j]).full()
         for i, j in zip(range(N//2), range(N-1, N//2-1, -1))]
prop = ff.util.mdot(swaps) @ QFT.total_propagator
qt.matrix_histogram(prop, bar_style='abs', color_style='phase')
print('Correct action: ', ff.util.oper_equiv(prop, qt_qft(N), eps=1e-13))

fig, ax, _ = plotting.plot_filter_function(QFT, omega)
# Move the legend to the side because of many entries
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

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
This example implements a basic Randomized Benchmarking simulation with two
different types of noise - white and correlated - to highlight correlations
between gates captured in the filter functions.
"""
import time

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from numpy import ndarray
from scipy import optimize

import filter_functions as ff
from filter_functions import PulseSequence, util
from numpy.random import permutation

# %%


def fitfun(m, A, B):
    return A*m + B


def state_infidelity(pulse: PulseSequence, S: ndarray, omega: ndarray,
                     ind: int = 2) -> float:
    """Compute state infidelity for input state eigenstate of pauli *ind*"""
    R = pulse.get_control_matrix(omega)
    F = np.einsum('jko->jo', util.abs2(R[:, np.delete([0, 1, 2], ind)]))
    return np.trapz(F*S, omega)/(2*np.pi*pulse.d)


def find_inverse(U: ndarray) -> ndarray:
    """
    Function to find the inverting gate to take the input state back to itself.
    """
    eye = np.identity(U.shape[0])
    if util.oper_equiv(U, eye, eps=1e-8)[0]:
        return Id

    for i, gate in enumerate(permutation(cliffords)):
        if util.oper_equiv(gate.total_Q @ U, eye, eps=1e-8)[0]:
            return gate

    # Shouldn't reach this point because the major axis pi and pi/2 rotations
    # are in the Clifford group, the state is always an eigenstate of a Pauli
    # operator during the pulse sequence.
    raise Exception


def run_randomized_benchmarking(N_G: int, N_l: int, min_l: int, max_l: int,
                                omega):
    infidelities = np.empty((N_l, N_G), dtype=float)
    lengths = np.round(np.linspace(min_l, max_l, N_l)).astype(int)
    delta_t = []
    t_now = [time.perf_counter()]
    print(f'Start simulation with {len(lengths)} sequence lengths')
    print('---------------------------------------------')
    for l, length in enumerate(lengths):
        t_now.append(time.perf_counter())
        delta_t.append(t_now[-1] - t_now[-2])
        print('Sequence length', length,
              f'Elapsed time: {t_now[-1] - t_now[0]:.2f} s', sep='\t')
        for j in range(N_G):
            randints = np.random.randint(0, len(cliffords), lengths[l])
            U = ff.concatenate(cliffords[randints])
            U_inv = find_inverse(U.total_Q)
            pulse_sequence = U @ U_inv
            infidelities[l, j] = state_infidelity(
                pulse_sequence, S, omega
            ).sum()

    return infidelities, delta_t


# %% Set up Hamiltonians
T = 20
H_n = {}
H_c = {}
dt = {}
H_c['Id'] = [[qt.sigmax().full(), [0], 'X']]
H_n['Id'] = [[qt.sigmax().full(), [1], 'X'],
             [qt.sigmay().full(), [1], 'Y']]
dt['Id'] = [T]
H_c['X2'] = [[qt.sigmax().full(), [np.pi/4/T], 'X']]
H_n['X2'] = [[qt.sigmax().full(), [1], 'X'],
             [qt.sigmay().full(), [1], 'Y']]
dt['X2'] = [T]
H_c['Y2'] = [[qt.sigmay().full(), [np.pi/4/T], 'Y']]
H_n['Y2'] = [[qt.sigmax().full(), [1], 'X'],
             [qt.sigmay().full(), [1], 'Y']]
dt['Y2'] = [T]

# %% Set up PulseSequences
Id = ff.PulseSequence(H_c['Id'], H_n['Id'], dt['Id'])
X2 = ff.PulseSequence(H_c['X2'], H_n['X2'], dt['X2'])
Y2 = ff.PulseSequence(H_c['Y2'], H_n['Y2'], dt['Y2'])

# %% Define some parameters
m_min = 1
m_max = 151
# sequence lengths
N_l = 21
lengths = np.round(np.linspace(m_min, m_max, N_l)).astype(int)
# no. of random sequences per length
N_G = 50

omega = np.geomspace(1e-2/(7*m_max*T), 1e2/T, 301)*2*np.pi

# %% Cache filter functions for primitive gates
Id.cache_control_matrix(omega)
X2.cache_control_matrix(omega)
Y2.cache_control_matrix(omega)

# %% Construct Clifford group
tic = time.perf_counter()
cliffords = np.array([
    Id,                                 # Id
    X2 @ X2,                            # X
    Y2 @ Y2,                            # Y
    Y2 @ Y2 @ X2 @ X2,                  # Z
    X2 @ Y2,                            # Y/2 ○ X/2
    X2 @ Y2 @ Y2 @ Y2,                  # -Y/2 ○ X/2
    X2 @ X2 @ X2 @ Y2,                  # Y/2 ○ -X/2
    X2 @ X2 @ X2 @ Y2 @ Y2 @ Y2,        # -Y/2 ○ -X/2
    Y2 @ X2,                            # X/2 ○ Y/2
    Y2 @ X2 @ X2 @ X2,                  # -X/2 ○ Y/2
    Y2 @ Y2 @ Y2 @ X2,                  # X/2 ○ -Y/2
    Y2 @ Y2 @ Y2 @ X2 @ X2 @ X2,        # -X/2 ○ -Y/2
    X2,                                 # X/2
    X2 @ X2 @ X2,                       # -X/2
    Y2,                                 # Y/2
    Y2 @ Y2 @ Y2,                       # -Y/2
    X2 @ Y2 @ Y2 @ Y2 @ X2 @ X2 @ X2,   # Z/2
    X2 @ X2 @ X2 @ Y2 @ Y2 @ Y2 @ X2,   # -Z/2
    X2 @ X2 @ Y2,                       # Y/2 ○ X
    X2 @ X2 @ Y2 @ Y2 @ Y2,             # -Y/2 ○ X
    Y2 @ Y2 @ X2,                       # X/2 ○ Y
    Y2 @ Y2 @ X2 @ X2 @ X2,             # -X/2 ○ Y
    X2 @ Y2 @ X2,                       # X/2 ○ Y/2 ○ X/2
    X2 @ Y2 @ Y2 @ Y2 @ X2              # X/2 ○ -Y/2 ○ X/2
])

toc = time.perf_counter()
print(f'Construction of Clifford group: {toc - tic:.2f} s')
print()
# %% Run simulation

eps0 = 2.7241e-4
# Scaling factor for the noise so that alpha = 0 and alpha = 0.7 have the same
# power
noise_scaling_factor = {
    0.0: 0.4415924985735799,
    0.7: 1
}

state_infidelities = {}
clifford_infidelities = {}

for i, alpha in enumerate((0.0, 0.7)):
    S0 = 1e-13*(2*np.pi*1e-3)**alpha/eps0**2*noise_scaling_factor[alpha]
    S = S0/omega**alpha

    # Need to calculate with two-sided spectra
    clifford_infidelities[alpha] = [
        ff.infidelity(C, *util.symmetrize_spectrum(S, omega)).sum()
        for C in cliffords
    ]

    print('=============================================')
    print(f'\t\talpha = {alpha}')
    print('=============================================')
    state_infidelities[alpha], exec_times = run_randomized_benchmarking(
        N_G, N_l, m_min, m_max, omega
    )

# %% Plot results
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 3))

fidelities = {alpha: 1 - infid for alpha, infid in state_infidelities.items()}
for i, alpha in enumerate((0.0, 0.7)):

    means = np.mean(fidelities[alpha], axis=1)
    stds = np.std(fidelities[alpha], axis=1)

    popt, pcov = optimize.curve_fit(fitfun, lengths, means, [0, 1], stds,
                                    absolute_sigma=True)

    for j in range(N_G):
        fid = ax[i].plot(lengths, fidelities[alpha][:, j], 'k.', alpha=0.1,
                         zorder=2)

    mean = ax[i].errorbar(lengths, means, stds, fmt='.', zorder=3,
                          color='tab:red')
    fit = ax[i].plot(lengths, fitfun(lengths, *popt), '--', zorder=4,
                     color='tab:red')
    # The expectation for uncorrelated pulses is F = 1 - r*m with m the
    # sequence length and r = 1 - F_avg = d/(d + 1)*(1 - F_ent) the average
    # error per gate
    exp = ax[i].plot(lengths,
                     1 - np.mean(clifford_infidelities[alpha])*lengths*2/3,
                     '--', zorder=4, color='tab:blue')
    ax[i].set_title(rf'$\alpha = {alpha}$')
    ax[i].set_xlabel(r'Sequence length $m$')

handles = [fid[0], mean[0], fit[0], exp[0]]
labels = ['State Fidelity', 'Fidelity mean', 'Fit',
          'RB theory w/o pulse correlations']
ax[0].set_xlim(0, max(lengths))
ax[0].set_ylim(.993, 1)
ax[0].set_ylabel(r'Surival Probability')
ax[0].legend(frameon=False, handles=handles, labels=labels)

fig.tight_layout()

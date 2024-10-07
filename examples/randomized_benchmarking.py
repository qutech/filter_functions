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
import pathlib
import time
from typing import Dict, Sequence

import filter_functions as ff
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from numpy import ndarray
from numpy.random import permutation
from scipy import io, optimize, integrate

# %%


def fitfun(m, A):
    return 1 - A*m


def state_infidelity(pulse: ff.PulseSequence, S: ndarray, omega: ndarray,
                     ind: int = 3) -> float:
    """Compute state infidelity for input state eigenstate of pauli *ind*"""
    R = pulse.get_control_matrix(omega)
    F = np.einsum('jko->jo', ff.util.abs2(R[:, np.delete([0, 1, 2, 3], ind)]))
    return integrate.trapezoid(F*S, omega)/(2*np.pi*pulse.d)


def find_inverse(U: ndarray, cliffords: Sequence[ff.PulseSequence]) -> ndarray:
    """
    Function to find the inverting gate to take the input state back to itself.
    """
    eye = np.identity(U.shape[0])
    if ff.util.oper_equiv(U, eye, eps=1e-8)[0]:
        return cliffords[0]

    for i, gate in enumerate(permutation(cliffords)):
        if ff.util.oper_equiv(gate.total_propagator @ U, eye, eps=1e-8)[0]:
            return gate

    # Shouldn't reach this point because the major axis pi and pi/2 rotations
    # are in the Clifford group, the state is always an eigenstate of a Pauli
    # operator during the pulse sequence.
    raise Exception


def run_randomized_benchmarking(N_G: int, N_l: int, min_l: int, max_l: int, alpha: Sequence[float],
                                spectra: Dict[float, Sequence[float]], omega: Sequence[float],
                                cliffords: Sequence[ff.PulseSequence]):
    infidelities = {a: np.empty((N_l, N_G), dtype=float) for a in alpha}
    lengths = np.round(np.linspace(min_l, max_l, N_l)).astype(int)
    delta_t = []
    t_now = [time.perf_counter()]
    print(f'Start simulation with {len(lengths)} sequence lengths')
    print('---------------------------------------------')
    for l, length in enumerate(lengths):
        t_now.append(time.perf_counter())
        delta_t.append(t_now[-1] - t_now[-2])
        print('Sequence length', length, f'Elapsed time: {t_now[-1] - t_now[0]:.2f} s', sep='\t')
        for j in range(N_G):
            randints = np.random.randint(0, len(cliffords), lengths[l])
            U = ff.concatenate(cliffords[randints])
            U_inv = find_inverse(U.total_propagator, cliffords)
            pulse_sequence = U @ U_inv
            for k, a in enumerate(alpha):
                infidelities[a][l, j] = state_infidelity(pulse_sequence, spectra[a], omega).sum()

    return infidelities, delta_t


# %% Set up Hamiltonians
T = 20
pulse_types = ('naive', 'optimized')
gates = ('X2', 'Y2')

H_n = {pt: {} for pt in pulse_types}
H_c = {pt: {} for pt in pulse_types}
dt = {pt: {} for pt in pulse_types}
# %%% naive gates (assume noise just on X)
H_c['naive']['Id'] = [[qt.sigmax().full()/2, [0], 'X']]
H_n['naive']['Id'] = [[qt.sigmax().full()/2, [1], 'X']]
dt['naive']['Id'] = [T]
H_c['naive']['X2'] = [[qt.sigmax().full()/2, [np.pi/2/T], 'X']]
H_n['naive']['X2'] = [[qt.sigmax().full()/2, [1], 'X']]
dt['naive']['X2'] = [T]
H_c['naive']['Y2'] = [[qt.sigmay().full()/2, [np.pi/2/T], 'Y']]
H_n['naive']['Y2'] = [[qt.sigmax().full()/2, [1], 'X']]
dt['naive']['Y2'] = [T]

# %%% optimized gates
fpath = pathlib.Path(ff.__file__).parent.parent / 'examples/data'

# Set up Hamiltonian for X2, Y2 gate
struct = {gate: io.loadmat(fpath / f'{gate}ID.mat') for gate in gates}
eps = {gate: np.asarray(struct[gate]['eps'], order='C') for gate in gates}
B = {gate: np.asarray(struct[gate]['B'].ravel(), order='C') for gate in gates}
dt['optimized'] = {gate: np.asarray(struct[gate]['t'].ravel(), order='C') for gate in gates}

J = {gate: np.exp(eps[gate]) for gate in gates}
n_dt = {gate: len(dt['optimized'][gate]) for gate in gates}

c_coeffs = {gate: [J[gate][0], B[gate][0]*np.ones(n_dt[gate])] for gate in gates}
n_coeffs = {gate: np.ones((3, n_dt[gate])) for gate in gates}

# Add identity gate, choosing the X/2 operation on the right qubit
for gate in gates:
    H_c['optimized'][gate] = list(zip(ff.util.paulis[[1, 3]]/2, c_coeffs[gate], ('X', 'Z')))
    H_n['optimized'][gate] = list(zip(ff.util.paulis[1:2]/2, n_coeffs[gate], ('X',)))
# %% Set up PulseSequences
pulses = {p: {g: ff.PulseSequence(H_c[p][g], H_n[p][g], dt[p][g]) for g in gates}
          for p in pulse_types}

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
for p in pulse_types:
    for g in gates:
        pulses[p][g].cache_control_matrix(omega)

# %% Construct Clifford group
cliffords = {}

tic = time.perf_counter()
for p in pulse_types:
    X2, Y2 = pulses[p].values()
    cliffords[p] = np.array([
        Y2 @ Y2 @ Y2 @ Y2,                  # Id
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
    ], dtype=object)

toc = time.perf_counter()
print(f'Construction of Clifford group: {toc - tic:.2f} s\n')
# %% Run simulation
# We use the 1/f^0.7 spectrum from Dial et al (2013) and a white spectrum
# leading to the same average clifford infidelity


def spectrum(omega, alpha):
    eps0 = 2.7241e-4
    return 4e-11*(2*np.pi*1e-3/omega)**alpha/eps0**2


alpha = (0.0, 0.7)
# Scale noise such that average clifford infidelity is the same for all pulse types and alpha
clifford_infids = {p: {a: np.array([
    ff.infidelity(c, spectrum(omega, a), omega) for c in cliffords[p]
]) for a in alpha} for p in pulse_types}

noise_scaling_factor = {p: {
    a: clifford_infids['optimized'][0.7].sum(1).mean(0) / clifford_infids[p][a].sum(1).mean(0)
    for a in alpha
} for p in pulse_types}

state_infidelities = {}
clifford_infidelities = {}

spectra = {}
for p in pulse_types:
    spectra[p] = {}
    clifford_infidelities[p] = {}
    for a in alpha:
        spectra[p][a] = noise_scaling_factor[p][a] * spectrum(omega, a)
        clifford_infidelities[p][a] = np.array([ff.infidelity(c, spectra[p][a], omega).sum()
                                                for c in cliffords[p]])

    print(f'\nRunning RB simulation for {p} gates')
    state_infidelities[p], exec_times = run_randomized_benchmarking(N_G, N_l, m_min, m_max,
                                                                    alpha, spectra[p], omega,
                                                                    cliffords[p])

# %% Plot results
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 5))

fidelities = {p: {a: 1 - infid for a, infid in state_infidelities[p].items()} for p in pulse_types}

for i, p in enumerate(pulse_types):
    for j, a in enumerate(alpha):
        means = np.mean(fidelities[p][a], axis=1)
        stds = np.std(fidelities[p][a], axis=1)

        popt, pcov = optimize.curve_fit(fitfun, lengths, means, [0], stds, absolute_sigma=True)

        for k in range(N_G):
            fid = ax[i, j].plot(lengths, fidelities[p][a][:, k], 'k.', alpha=0.1, zorder=2)

        mean = ax[i, j].errorbar(lengths, means, yerr=stds, fmt='+', zorder=3, color='tab:red')
        fit = ax[i, j].plot(lengths, fitfun(lengths, *popt), '--', zorder=4, color='tab:red')
        # The expectation for uncorrelated pulses is F = 1 - r*m with m the
        # sequence length and r = 1 - F_avg = d/(d + 1)*(1 - F_ent) the average
        # error per gate
        exp = ax[i, j].plot(lengths, 1 - np.mean(clifford_infidelities[p][a])*lengths*2/3,
                            '--', zorder=4, color='tab:blue')
        ax[i, j].set_title(rf'{p} pulses, $\alpha = {a}$')
        if i == 1:
            ax[i, j].set_xlabel(r'Sequence length $m$')
        if j == 0:
            ax[i, j].set_ylabel(r'Surival Probability')


handles = [fid[0], mean, fit[0], exp[0]]
labels = ['State Fidelity', 'Fidelity mean+std', 'Fit', 'RB theory w/o pulse correlations']
ax[i, j].legend(frameon=False, handles=handles, labels=labels)
ax[i, j].set_xlim(0, max(lengths))

fig.tight_layout()

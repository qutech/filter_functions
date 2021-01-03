import numpy as np
import filter_functions as ff

sigma_x = np.asarray([[0, 1], [1, 0]]) / 2
sigma_z = np.diag([1, -1]) / 2

j_0 = 1.
eps_0 = 1.
S_0 = 1.

n_time_steps = 10
time_step = 1e-1


def exchange_interaction(eps):
    return j_0 * np.exp(eps / eps_0)


def deriv_exchange_interaction(eps):
    return j_0 / eps_0 * np.exp(eps / eps_0)


def deriv_2_exchange_interaction(eps):
    return j_0 / eps_0 ** 2 * np.exp(eps / eps_0)


def one_over_f_noise(f):
    return S_0 / f


def create_sing_trip_pulse_seq(eps, dbz):

    H_c = [
        [sigma_z, exchange_interaction(eps[0]), 'control1'],
        [sigma_x, dbz * np.ones(eps.shape[1]), 'drift']
    ]

    H_n = [
        [sigma_z, deriv_exchange_interaction(eps[0])]
        # [sigma_z, np.ones_like(eps[0])]

    ]

    dt = time_step * np.ones(n_time_steps)

    pulse = ff.PulseSequence(H_c, H_n, dt)
    return pulse


def finite_diff_infid(u_ctrl_central, u_drift, pulse_sequence_builder,
                      spectral_noise_density, n_freq_samples=200,
                      c_id=None, delta_u=1e-6):
    """
    Parameters
    ----------
    u_ctrl_central: shape (n_ctrl, n_dt)
    u_drift: shape (n_dt)
    pulse_sequence_builder: function handle
    spectral_noise_density: function handle
    n_freq_samples: int
    c_id: List of string
    delta_u: float

    Returns
    -------
    gradient: shape (n_nop, n_dt, n_ctrl)

    """
    pulse = pulse_sequence_builder(u_ctrl_central, u_drift)
    all_id = pulse.c_oper_identifiers
    if c_id is None:
        c_id = all_id[:len(u_ctrl_central)]
    omega = ff.util.get_sample_frequencies(
        pulse=pulse, n_samples=n_freq_samples, spacing='log')
    S = spectral_noise_density(omega)

    gradient = np.empty((len(pulse.n_coeffs), len(pulse.dt), len(c_id)))

    for g in range(len(pulse.dt)):
        for k in range(len(c_id)):
            u_plus = u_ctrl_central.copy()
            u_plus[k, g] += delta_u
            pulse_plus = pulse_sequence_builder(u_plus, u_drift)
            infid_plus = ff.numeric.infidelity(pulse=pulse_plus, spectrum=S,
                                               omega=omega)

            u_minus = u_ctrl_central.copy()
            u_minus[k, g] -= delta_u
            pulse_minus = pulse_sequence_builder(u_minus, u_drift)
            infid_minus = ff.numeric.infidelity(pulse=pulse_minus, spectrum=S,
                                                omega=omega)

            gradient[:, g, k] = (infid_plus - infid_minus) / 2 / delta_u
    return gradient


def analytic_gradient(u_ctrl, u_drift, pulse_sequence_builder,
                      spectral_noise_density, s_derivs=None, n_freq_samples=200,
                      c_id=None, ctrl_amp_deriv=None):
    """
    Parameters
    ----------
    u_ctrl: shape (n_ctrl, n_dt)
    u_drift: shape (n_dt)
    pulse_sequence_builder: function handle
    spectral_noise_density: function handle
    n_freq_samples: int
    c_id: List of string
    s_derivs: shape (n_nops, n_ctrl, n_dt)
    ctrl_amp_deriv: function handle

    Returns
    -------
    gradient: shape (n_nop, n_dt, n_ctrl)

    """

    pulse = pulse_sequence_builder(u_ctrl, u_drift)
    omega = ff.util.get_sample_frequencies(
        pulse=pulse, n_samples=n_freq_samples, spacing='log')
    S = spectral_noise_density(omega)
    gradient = ff.gradient.infidelity_derivative(
        pulse=pulse, S=S, omega=omega, control_identifiers=c_id,
        s_derivs=s_derivs)
    if ctrl_amp_deriv is not None:
        # gradient shape (n_nops, n_dt, n_ctrl)
        # deriv_exchange_interaction shape (n_ctrl, n_dt)
        gradient = np.einsum("agk,kg -> agk", gradient,
                             ctrl_amp_deriv(u_ctrl))
    return gradient


def relative_norm_difference(m, n):
    return np.linalg.norm(m - n) / (np.linalg.norm(m) + np.linalg.norm(n)) * 2

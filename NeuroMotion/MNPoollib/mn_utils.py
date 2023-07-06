import torch
import numpy as np
import matplotlib.pyplot as plt

from BioMime.utils.params import coeff_r_a, coeff_r_b, coeff_fb_a, coeff_fb_b, coeff_a_a, coeff_a_b, coeff_iz_a, coeff_iz_b, coeff_cv_a, coeff_cv_b, coeff_len_a, coeff_len_b, w_amp


def generate_emg_mu(muaps, spikes, time_samples):
    """
    Args:
        muaps (np.array): [time_steps, nrow, ncol, duration]
        spikes (list): indices of spikes
        time_samples (int): fs * movement_time

    Return:
        EMG (np.array): [nrow, ncol, time_samples]
    """

    muap_steps, nrow, ncol, time_length = muaps.shape
    emg = np.zeros((nrow, ncol, time_samples + time_length))
    for t in spikes:
        muap_time_id = get_cur_muap(muap_steps, t, time_samples)
        emg[:, :, t:t + time_length] = muaps[muap_time_id]

    return emg


def normalise_properties(db, num_mus, steps=1):

    num = torch.from_numpy((db['num_fibre_log'] + coeff_fb_a) * coeff_fb_b).reshape(num_mus, 1).repeat(1, steps)
    depth = torch.from_numpy((db['mu_depth'] + coeff_r_a) * coeff_r_b).reshape(num_mus, 1).repeat(1, steps)
    angle = torch.from_numpy((db['mu_angle'] + coeff_a_a) * coeff_a_b).reshape(num_mus, 1).repeat(1, steps)
    iz = torch.from_numpy((db['iz'] + coeff_iz_a) * coeff_iz_b).reshape(num_mus, 1).repeat(1, steps)
    cv = torch.from_numpy((db['velocity'] + coeff_cv_a) * coeff_cv_b).reshape(num_mus, 1).repeat(1, steps)
    length = torch.from_numpy((db['len'] + coeff_len_a) * coeff_len_b).reshape(num_mus, 1).repeat(1, steps)

    base_muap = db['muap'].transpose(0, 3, 1, 2) * w_amp
    base_muap = torch.from_numpy(base_muap).unsqueeze(1).float()

    return num, depth, angle, iz, cv, length, base_muap


def get_cur_muap(muap_steps, cur_step, time_samples):
    return int(muap_steps * cur_step / time_samples)


def plot_spike_trains(spikes, pth):
    """
    spikes  list (index of spikes) in list (MUs)
    pth     figure save path
    """

    fig = plt.figure()
    num_mu = len(spikes)
    for mu in range(num_mu):
        spike = spikes[mu]
        plt.vlines(spike, mu, mu + 0.5, linewidth=1.0)
    plt.xlabel('Time in sample')
    plt.ylabel('MU Index')
    plt.savefig(pth)

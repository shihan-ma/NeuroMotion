import numpy as np
import matplotlib.pyplot as plt


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

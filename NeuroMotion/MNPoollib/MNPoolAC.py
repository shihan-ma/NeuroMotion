""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
August 2023
Imperial College London
Department of Civil Engineering
Derived from the Github code provided with Caillet, A. H., Phillips, A. T., Farina, D., & Modenese, L. (2022). Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling. PLOS Computational Biology, 18(9), e1010556.
The github code was adapted for your needs.  
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from rc_lif_mod import RC_solve_func
from scipy import signal
from easydict import EasyDict as edict

import sys
sys.path.append('.')

from BioMime.utils.params import coeff_a, coeff_b
from NeuroMotion.MNPoollib.mn_params import NUM_FIBRES_MS, ANGLE, DEPTH


class MotoneuronPoolAC:
    def __init__(self, N, ms_name) -> None:
        """
        N       Number of motor units
        """
        self.N = N
        self.ms_name = ms_name
        self.MN_array = np.arange(0, N, 1) + 1
        self.I1 = 3.9 * 10**-9  # in A
        self.I2 = 35.0 * 10**-9

        self._init_pool()

    def _ARP_func(self, j):
        return 0.04 * j**0.05

    def _MN_size(self, j, N):
        return 1.49 * 10**-7 * 2.4**((j / N)**1.47)

    def _init_pool(self):
        self.MN_size_array = self._MN_size(self.MN_array, self.N)
        self.ARP_array = self._ARP_func(self.MN_array)

    # Properties
    def assign_properties(self, config, normalise=True, nrange=[0.5, 1.0]):
        """
        keys and vals in config:
        num     number of fibres or innervation ratio, for simplicity proportional to max twitch
        depth   uniformly distributed within a range
        angle   uniformly distributed within a range
        iz      innervation zone, mean and std
        len     fibre length, mean and std
        cv      conduction velocity, mean and std
        """

        def ftw_distrib(i, N):
            return 0.812 * (18.509 * i / N + 104.098**((i / N)**4.831))  # derived from experimental data, distirbution of normalised MU twitch forces

        MU_list = np.arange(self.N)
        ftw_list = ftw_distrib(MU_list ,self.N - 1)
        ftw_tot = np.sum(ftw_list)
        num = (ftw_list / ftw_tot * NUM_FIBRES_MS[self.ms_name]).astype(int)

        depth = np.random.rand(self.N) * (config.depth[1] - config.depth[0]) + config.depth[0]      # Uniform distribution
        angle = np.random.rand(self.N) * (config.angle[1] - config.angle[0]) + config.angle[0]      # Uniform distribution
        iz = np.random.randn(self.N) * config.iz[1] + config.iz[0]              # Normal distribution
        len = np.random.randn(self.N) * config.len[1] + config.len[0]           # Normal distribution
        cv = 1.5 * 10e4 * self.MN_size_array**0.69

        mn = {
            'num': self._normalise(normalise, np.log(num), *nrange, label='num'),
            'depth': self._normalise(normalise, depth, *nrange, label='depth'),
            'angle': self._normalise(normalise, angle, *nrange, label='angle'),
            'iz': self._normalise(normalise, iz, *nrange, label='iz'),
            'len': self._normalise(normalise, len, *nrange, label='len'),
            'cv': self._normalise(normalise, cv, *nrange, label='cv'),
        }

        self.properties = mn

        return mn

    def _normalise(self, normalise, vals, low, high, local=False, label=None):
        if not normalise:
            return vals
        if local:
            return (vals - vals.min()) / (vals.max() - vals.min()) * (high - low) + low
        else:
            assert label is not None
            return (vals + coeff_a[label]) * coeff_b[label]

    def get_spike_train(self, act, fs, t_stop):
        """
        act: normalised synaptic input [0, 1]
        """
        step_size = 1 / 2 / fs
        self.time_list = np.arange(0, t_stop, step_size)

        def I(t):
            return self.I1 + (self.I2 - self.I1) * act[int(t / step_size)]

        self.I_list = [I(self.time_list[i]) for i in range (len(self.time_list))]

        firing_times_sim = np.empty((self.N,), dtype=object)
        self.parameters_list = np.empty((self.N, 5))

        for mn in self.MN_array:
            V, firing_times_sim[mn-1], self.parameters_list[mn-1] = RC_solve_func(I, self.time_list, self.MN_size_array[mn-1], step_size, self.ARP_array[mn-1] )
            if mn % 20 == 0:
                print('Simulating mn n°', mn, len(firing_times_sim[mn-1]))
        return firing_times_sim, step_size

    def firing_times_to_spike_trains(self, firing_times_sim, step_size):
        firing_samples = np.empty((self.N,), dtype=object)
        n_sp = np.zeros((len(firing_times_sim), len(self.time_list)))
        for i in range(len(firing_times_sim)):
            cur_samples = (firing_times_sim[i] / step_size).astype(int)
            n_sp[i, cur_samples] = 1
            firing_samples[i] = cur_samples
        return n_sp, firing_samples

    def get_binary_spikes(self, act, fs, t_stop):
        firing_times_sim, step_size = self.get_spike_train(act, fs, t_stop)
        n_spike_trains, firing_samples = self.firing_times_to_spike_trains(firing_times_sim, step_size)
        cst = np.sum(n_spike_trains, 0)
        return n_spike_trains, firing_samples, cst, firing_times_sim

    def plot_current_input(self, pth):
        assert self.time_list is not None and self.I_list is not None
        plt.plot(self.time_list, self.I_list)
        plt.xlabel('Time (s)')
        plt.ylabel('Current Input (A)')
        plt.ylim(0, np.max(self.I_list) * 1.1)
        plt.title('Current input in A')
        plt.savefig(f'{pth}/current_input.jpg')
        plt.close()

    def plot_mn_size(self, pth):
        plt.scatter(self.MN_array, self.MN_size_array)
        plt.xlabel('MN population')
        plt.ylabel('MN Sizes (m2)')
        plt.ylim(0, np.max(self.MN_size_array) * 1.1)
        plt.title('Distribution of MN sizes in the MN pool')
        plt.savefig(f'{pth}/mn_sizes.jpg')
        plt.close()

    def plot_events(self, duration, firing_times_sim, pth):
        width = 0.9
        colors1, lineoffsets1, linelengths1 = ['C{}'.format(i) for i in range(self.N)], np.arange(1, self.N + 1, 1), np.ones((1, self.N))[0] * width
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        axs.set_xlabel('Time [s]', color='k', fontsize=20)
        ylabel='Motoneurons ' + r'$(N)$'
        axs.set_ylabel(ylabel, color='k', fontsize=20)
        axs.set_ylim(0, self.N + 5)
        axs.ax2= axs.twinx()
        f = lambda x, pos: str(x).rstrip('0').rstrip('.')
        axs.ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        axs.ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
        axs.ax2.set_ylabel('Current Input (A)',fontsize=15)
        axs.ax2.set_xlim(0, duration)
        axs.ax2.set_ylim(0, np.max(self.I_list))
        axs.eventplot(firing_times_sim, colors=colors1, lineoffsets=lineoffsets1,linelengths=linelengths1)
        axs.ax2.plot(self.time_list, self.I_list)
        plt.title('Simulated spike trains for the pool of N modelled MNs', fontsize=20)
        plt.savefig(f'{pth}/spikes.jpg')
        plt.close()

    def plot_parameter_dist(self, pth):
        # Parameter distribution
        plt.scatter(self.MN_array, self.parameters_list[:, 2])
        plt.xlabel('MN population')
        plt.ylabel('MN Input resistance (Ohm)')
        plt.ylim(0, np.max(self.parameters_list[:, 2]) * 1.1)
        plt.title('Distribution of membran Input Resistance R in the MN pool')
        plt.savefig(f'{pth}/membrane_input_res.jpg')
        plt.close()

    def plot_time_constant(self, pth):
        plt.scatter(self.MN_array, self.parameters_list[:, 4])
        plt.xlabel('MN population')
        plt.ylabel('MN Membrane time constant (s)')
        plt.ylim(0, np.max(self.parameters_list[:, 4]) * 1.1)
        plt.title('Distribution of MN membrane time constant in the MN pool')
        plt.savefig(f'{pth}/time_constant.jpg')
        plt.close()

    def plot_neural_drive(self, fs, act, cst, pth):
        # Neural drive (filtered CST) obtained from the virtual pool of N firing MNs
        hanning_window = signal.windows.hann(int(0.4 * fs))  # np.hanning(L)
        sum_Han = sum(hanning_window)    
        neural_drive= signal.convolve(cst, hanning_window, mode='same') / sum_Han
        plt.plot(self.time_list, neural_drive, label='Neural Drive')
        plt.plot(self.time_list, act, label='Normalized Input')
        plt.xlabel('Time (s)')
        plt.ylabel('Neural drive / Normalized Input')
        plt.legend()
        plt.savefig(f'{pth}/neural_drive.jpg')
        plt.close()

    def display_onion_skin_theory(self, firing_times_sim, n_spike_trains, t_stop, pth):
        # First compute the series of instantaneous discharge rates for the firing MNs
        idf = np.empty((self.N,), dtype=object) 
        for i in range(self.N):
            if len(firing_times_sim[i]) - 1 < 2:
                break
            else:
                idf[i] = 1 / np.diff(firing_times_sim[i])
        idf = idf[0 : i]

        smoothed_IDF_sim_sec = np.empty((len(idf),), dtype=object)
        for i in range(len(idf)):    
            smoothed_IDF_sim_sec[i] = np.poly1d(np.polyfit(self.time_list[n_spike_trains[i] > 0][0 : -1].astype(float), idf[i].astype(float), 6))(self.time_list[n_spike_trains[i] > 0][0 : -1].astype(float))
            if i % 5 == 0:
                plt.plot(self.time_list[n_spike_trains[i] > 0][0 : -1], smoothed_IDF_sim_sec[i], color=(i / len(idf), 0.4, (len(idf) - i) / len(idf)))  # c=colors[i])
        plt.xlim(0, t_stop)
        plt.ylim(0, np.max(1 / self.ARP_array) * 1.1)
        plt.xlabel('Time (s)')
        plt.ylabel('Filtered discharge frequencies (Hz)')
        plt.title('Onion Skin theory - Simulated N MNs')
        plt.savefig(f'{pth}/onion_skin.jpg')
        plt.close()


if __name__ == '__main__':

    # Test example
    num_mu = 186
    ms_label = 'ECRB'
    mn_pool = MotoneuronPoolAC(num_mu, ms_label)

    fs = 2048
    t_stop = 10
    act = np.arange(0, t_stop, 1 / 2 / fs) / t_stop * 0.6
    n_spike_trains, firing_samples, cst, firing_times_sim = mn_pool.get_binary_spikes(act, fs, t_stop)
    # firing_times_sim, step_size = mn_pool.get_spike_train(act, fs, t_stop)

    config = edict({
        'depth': DEPTH[ms_label],
        'angle': ANGLE[ms_label],
        'iz': [0.5, 0.1],
        'len': [1.0, 0.05],
        'cv': [4, 0.3]      # Recommend not setting std too large. cv range in training dataset is [3, 4.5]
    })
    properties = mn_pool.assign_properties(config)
    print(properties)
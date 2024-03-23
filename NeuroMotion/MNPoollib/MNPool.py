import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

import sys
sys.path.append('.')

from BioMime.utils.params import coeff_a, coeff_b
from NeuroMotion.MNPoollib.mn_params import DEPTH, ANGLE, MS_AREA, NUM_MUS, mn_default_settings


class MotoneuronPool:
    def __init__(self, N, ms_name, rr, rm, rp, pfr1, pfrd, mfr1, mfrd, gain, c_ipi, frs1, frsd, mode='exp', fibre_density=200, **kwargs):
        """
        N       Number of motor units
        ms_name name of muscle
        rr      Recruitment range: largest/smallest
        rm      Recruitment maximum when all MUs are active
        rp      Force fold: largest/smallest
        pfr1    Peak firing rate of first MU, Hz
        pfrd    Peak firing rate difference between first and last MU, Hz
        mfr1    Minimum firing rate of first MU, Hz
        mfrd    Minimum firing rate difference between first and last MU, Hz
        gain    Excitatory drive-firing rate relationship, 3 pps per 10% MVC => 0.3 Hz per % MVC
        c_ipi   Coefficient of std of inter-pulse intervals
        frs1    Slope of drive-firing rate of first MU
        frsd    Difference of slope of drive-firing rate between the first MU and the last MU
        mode    Linear or exp when generating min/max/slope of fr among MUs
        fibre density   density of muscle fibres

        Note that if frsd is not equal to 0, slope varies among MUs. 
        mode - linear uses gain v.s. mode - exp uses [frs1, frsd] are two ways to calculate fr given ext.
        """

        self.N = N
        self.ms_name = ms_name
        self.rm = rm
        self.rr = rr
        self.rp = rp
        self.pfr = [pfr1, pfrd]
        self.mfr = [mfr1, mfrd]
        self.gain = gain
        self.c_ipi = c_ipi
        self.frs = [frs1, frsd]
        self.fr_mode = mode
        self.fibre_density = fibre_density
        self.properties = None

        self._init_pool()

    def get_num_mu(self):
        return self.N

    def get_properties(self):
        return self.properties

    # Firing related
    def _init_pool(self, mode='ls2n'):
        self._init_recruitment_threshold(mode)
        self._init_frs(mode)
        self._init_phys_params()

    def _init_phys_params(self):
        num_fb = np.round(MS_AREA[self.ms_name] * self.fibre_density)
        self.phys_params = edict({
            'num_fb': num_fb,
            'depth': DEPTH[self.ms_name],
            'angle': ANGLE[self.ms_name],
            'iz': [0.5, 0.1],
            'len': [1.0, 0.05],
            'cv': [4, 0.3],
        })

    def _init_recruitment_threshold(self, mode):
        if mode == 'ls2n':
            rt = (self.rm / self.rr) * np.exp(np.arange(self.N) * np.log(self.rr) / (self.N - 1))
            self.rte = (self.rm / self.rr) * (np.exp(np.arange(self.N) * np.log(self.rr + 1) / self.N) - 1)
            self.rte = self.rte * np.max(rt) / np.max(self.rte)
        elif mode == 'fuglevand':
            self.rte = np.exp(np.arange(1, self.N + 1) * np.log(self.rr) / self.N)
        self.rte = self.rte.reshape(self.N, 1)

    def _init_frs(self, mode):
        if self.fr_mode == 'linear':
            self.minfr = np.linspace(self.mfr[0], self.mfr[0] - self.mfr[1], self.N)    # linear
            self.minfr = self.minfr.reshape(self.N, 1)
            self.maxfr = self.pfr[0] - self.pfr[1] * self.rte / self.rte[-1]    # PFRi = PFR1 - PFRD * RTEi / RTEn
            self.maxfr = self.maxfr.reshape(self.N, 1)
            self.slope_fr = np.ones((self.N, 1)) * self.gain
        elif self.fr_mode == 'exp':
            self.minfr = self.mfr[0] - self.mfr[1] * self.rte / np.max(self.rte)
            self.maxfr = self.pfr[0] - self.pfr[1] * self.rte / np.max(self.rte)
            self.slope_fr = self.frs[0] - self.frs[1] * self.rte / np.max(self.rte)
        if mode == 'fuglevand':
            self.rte2mvc = (1.0 - (self.pfr[-1] - self.mfr[-1]) / self.slope_fr[-1]) / self.rte[-1]     # Coefficient that converts RTE to percentage in MVC
            self.rte = self.rte * self.rte2mvc      # RTE now in % MVC and consistent with common drive units

    def _calculate_fr(self, E):
        """
        Args:
            E       excitation in percentage MVC, (1, T)

        Return:
            fr      (N, T)
        """

        if E.ndim == 1:
            E = E.reshape(1, E.shape[-1])

        fr = np.minimum(self.maxfr, self.minfr + (E - self.rte) * self.slope_fr)
        fr[E < self.rte] = 0
        return fr

    def generate_spike_trains(self, E, fit=False):
        """
        Args:
            E       excitation in percentage MVC, (1, T)

        Return:
            spikes  list(list), N, T
        """

        if fit:
            E = np.polyval(self.coeff_f2e, E)

        spikes = [[None]] * self.N
        for i in range(self.N):
            spikes[i] = []
        next_firing = np.ones(self.N) * -1
        cur_ipi = np.zeros(self.N)

        fr = self._calculate_fr(E)
        ipi_real = np.zeros_like(fr)

        time_samples = E.shape[-1]
        for mu in range(self.N):
            for t in range(time_samples):
                if E[t] > self.rte[mu]:
                    if next_firing[mu] < 0:
                        cur_ipi[mu] = self.fs / fr[mu, t]
                        cur_ipi[mu] = cur_ipi[mu] + np.random.randn() * cur_ipi[mu] * 1 / 6
                        next_firing[mu] = t + int(cur_ipi[mu])
                    if t == next_firing[mu]:
                        if len(spikes[mu]) == 0:
                            ipi_real[mu, :t] = t
                        else:
                            ipi_real[mu, spikes[mu][-1]: t] = (t - spikes[mu][-1])
                        spikes[mu].append(t)
                        cur_ipi[mu] = self.fs / fr[mu, t]
                        cur_ipi[mu] = cur_ipi[mu] + np.random.randn() * cur_ipi[mu] * 1 / 6
                        next_firing[mu] = t + int(cur_ipi[mu])
                else:
                    next_firing[mu] = -1
        for mu in range(self.N):
            if len(spikes[mu]) == 0:
                ipi_real[mu][:] = time_samples
            else:
                ipi_real[mu][spikes[mu][-1]: time_samples] = time_samples - spikes[mu][-1]
        return E, spikes, fr, ipi_real

    # Twitch related
    def _init_twitch_params(self, fs):
        """
        Initialise peak twitch forces, time to peak and peak height given Fuglevand et al. model.

        fs: modelling frequency in Hz
        """
        self.P = np.exp(np.log(self.rp) / self.N * np.arange(1, self.N + 1))   # Peak heights
        self.Tmax = 90 / 1000 * fs     # Maximum time-to-peak delay, in time sample, 90 ms
        self.Tr = 3     # Time to peak range
        self.Tcoeff = np.log(self.rp) / np.log(self.Tr)     # c in paper
        self.T = self.Tmax * ((1 / self.P)**(1 / self.Tcoeff))

    def _get_gain(self, ipi):
        """
        gain in motor-unit force as a function of the firing rate

        ipi: [N, T]
        gain: [N, T]
        """
        instant_fr = self.T[:, None] / (ipi + 1e-8)     # [N, T]
        S = 1 - np.exp(-2 * (instant_fr)**3)
        gain = np.ones_like(instant_fr)
        gain[instant_fr > 0.4] = (S[instant_fr > 0.4] / instant_fr[instant_fr > 0.4]) / ((1 - np.exp(-2 * 0.4**3)) / 0.4)
        return gain

    def generate_force_offline(self, spikes, ipi, gain=None):

        if gain is None:
            gain = self._get_gain(ipi)

        time_samples = gain.shape[-1]

        # Generate force
        force = np.zeros(time_samples + self.max_twitch_len)
        for mu, mu_spikes in enumerate(spikes):
            for t in mu_spikes:
                force[t: t + self.max_twitch_len] = force[t: t + self.max_twitch_len] + gain[mu, t] * self.twitch_mat[mu, :]
        force = force / self.fmax
        return force[:time_samples], gain

    def _normalise_mvc(self):
        """
        Calculate fmax by averaging the summed force during 10 s 100% MVC from all MUs
        """
        self.fmax = 1.0
        # MVC spikes, in default 10 s
        mvc_ext = np.ones(10 * self.fs)
        _, spikes, _, ipis = self.generate_spike_trains(mvc_ext)
        mvc_force = self.generate_force_offline(spikes, ipis)[0]
        time_samples = mvc_force.shape[-1]
        self.fmax = np.mean(mvc_force[time_samples // 2:])
        mvc_force = mvc_force / self.fmax

    def init_twitches(self, fs):
        """
        initialise twitch profiles in mat
        """
        self.fs = fs
        self._init_twitch_params(fs)

        max_twitch_len = int(5 * np.max(self.T))
        self.max_twitch_len = max_twitch_len
        twitch_time_line = np.arange(max_twitch_len).reshape(1, max_twitch_len)
        self.twitch_mat = self.P.reshape(self.N, 1) / self.T.reshape(self.N, 1) * twitch_time_line * np.exp(1 - twitch_time_line / self.T.reshape(self.N, 1))

        self._normalise_mvc()

    def init_quisistatic_ef_model(self):
        # Two ext patterns to compare accumulating effects, negligible difference
        qsi_T = 10 * self.fs
        # qsi_E = np.linspace(0, 1.0, qsi_T)        # linear from 0 to 1
        times = np.linspace(0, 10.0, qsi_T)
        qsi_E = (np.sin(times) + 1) * 0.5           # sine
        _, qsi_spikes, _, qsi_ipis = self.generate_spike_trains(qsi_E)
        qsi_force = self.generate_force_offline(qsi_spikes, qsi_ipis)[0]

        # Polynomial fitting, e2f and f2e, degrees = 5
        self.coeff_f2e = np.polyfit(qsi_force, qsi_E, 5)
        self.coeff_e2f = np.polyfit(qsi_E, qsi_force, 5)

    # Properties
    def assign_properties(self, config=None, normalise=True, nrange=[0.5, 1.0]):
        """
        keys and vals in config:
        num     number of fibres or innervation ratio, for simplicity proportional to max twitch
        depth   uniformly distributed within a range
        angle   uniformly distributed within a range
        iz      innervation zone, mean and std
        len     fibre length, mean and std
        cv      conduction velocity, mean and std
        """

        if config is None:
            config = self.phys_params

        num = config.num_fb
        P = np.exp(np.log(self.rp) / self.N * np.arange(1, self.N + 1))
        num = np.round(num * P / np.sum(P))
        depth = np.random.rand(self.N) * (config.depth[1] - config.depth[0]) + config.depth[0]      # Uniform distribution
        angle = np.random.rand(self.N) * (config.angle[1] - config.angle[0]) + config.angle[0]      # Uniform distribution
        iz = np.random.randn(self.N) * config.iz[1] + config.iz[0]              # Normal distribution
        len = np.random.randn(self.N) * config.len[1] + config.len[0]           # Normal distribution
        cv = np.sort(np.random.randn(self.N) * config.cv[1] + config.cv[0])     # Normal distribution

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

    def display_onion_skin_theory(self, spikes, duration, fs, pth):

        # from spikes to n_spike_trains
        n_spike_trains = np.zeros((self.N, int(duration * fs)))
        firing_times_sim = []
        for i in range(self.N):
            if len(spikes[i]) > 0:
                firing_times_sim.append(np.array(spikes[i]) / fs)
                for j in spikes[i]:
                    n_spike_trains[i, j] = 1

        idf = np.empty((self.N,), dtype=object) 
        for i in range(len(firing_times_sim)):
            if len(firing_times_sim[i]) - 1 < 2:
                break
            else:
                idf[i] = 1 / np.diff(firing_times_sim[i])
        idf = idf[0 : i]

        smoothed_IDF_sim_sec = np.empty((len(idf),), dtype=object)
        time_list = np.arange(0, duration, 1 / fs)
        for i in range(len(idf)):    
            smoothed_IDF_sim_sec[i] = np.poly1d(np.polyfit(time_list[n_spike_trains[i] > 0][0 : -1].astype(float), idf[i].astype(float), 6))(time_list[n_spike_trains[i] > 0][0 : -1].astype(float))
            if i % 5 == 0:
                plt.plot(time_list[n_spike_trains[i] > 0][0 : -1], smoothed_IDF_sim_sec[i], color=(i / len(idf), 0.4, (len(idf) - i) / len(idf)))  # c=colors[i])
        plt.xlim(0, duration)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Filtered discharge frequencies (Hz)', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.title('Onion Skin theory - Simulated N MNs')
        plt.savefig(pth)
        plt.close()

    def _normalise(self, normalise, vals, low, high, local=False, label=None):
        if not normalise:
            return vals
        if local:
            return (vals - vals.min()) / (vals.max() - vals.min()) * (high - low) + low
        else:
            assert label is not None
            return (vals + coeff_a[label]) * coeff_b[label]


if __name__ == '__main__':

    # Test example
    num_mu = 186
    muscle = "ECRL"
    mn_pool = MotoneuronPool(num_mu, muscle, **mn_default_settings)

    # properties
    config = edict({
        'num_fb': 25000,
        'depth': [20, 30],
        'angle': [20, 30],
        'iz': [0.5, 0.1],
        'len': [0.5, 0.1],
        'cv': [4, 0.5]
    })
    properties = mn_pool.assign_properties(config, True)

    # Excitation
    fs = 2048           # Hz
    duration = 6        # s
    times = np.linspace(0, duration, duration * fs)
    ext = np.linspace(0, 0.3, fs * duration)
    # ext = np.concatenate((np.linspace(0, 0.8, round(fs * duration / 2)), np.linspace(0.8, 0, round(fs * duration / 2))))
    # ext = (np.sin(times) + 1) * 0.4

    # start = 1
    # ramp = 1
    # ext = np.concatenate((np.zeros(start * fs), np.arange(0, ramp, 1 / fs) / ramp * 0.3, np.ones(int((duration // 2 - start - ramp) * fs))))
    # ext = np.concatenate((ext, ext[::-1]))

    # ext = np.ones(duration * fs) * 0.3

    # Force and Twitches
    mn_pool.init_twitches(fs)
    mn_pool.init_quisistatic_ef_model()

    _, spikes, fr, ipis = mn_pool.generate_spike_trains(ext)

    active_mu = 0
    for sp in spikes:
        if len(sp) > 0:
            active_mu += 1

    # # Visualisation
    # # plot spikes
    # fig = plt.figure()
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # num_mu = len(spikes)
    # for mu in range(num_mu):
    #     spike = spikes[mu]
    #     plt.vlines(spike, mu, mu + 0.5, linewidth=1.0)
    # ax.set_xticks(range(0, duration * fs + 1, 2*fs), labels=['0', '2', '4', '6'])
    # ax.set_ylabel('Discharge Patterns (MU index)', fontsize=14)
    # ax.set_xlabel('Time (s)', fontsize=14)
    # ax.xaxis.set_tick_params(labelsize=11)
    # ax.yaxis.set_tick_params(labelsize=11)

    # ax2 = ax.twinx()
    # ax2.plot(times * fs, ext, linewidth=4, c='#003366', alpha=0.3)
    # ax2.tick_params(axis='y')
    # ax2.set_ylabel('Neural input', fontsize=14)
    # ax2.set_yticks([0, 1], labels=['0.0', '1.0'])
    # ax2.xaxis.set_tick_params(labelsize=11)
    # ax2.yaxis.set_tick_params(labelsize=11)
    # plt.tight_layout()
    # plt.savefig('./figs/spikes_ramp.svg')
    # plt.close()

    # # plot fr
    # cm = matplotlib.colormaps['bone']
    # my_colors = []
    # for i in np.linspace(0, 1, num_mu):
    #     my_colors.append(cm(i))

    # fig = plt.figure()
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # for i in range(0, num_mu, 5):
    #     plt.plot(fr[i], color=(i / num_mu, 0.4, (num_mu - i) / num_mu))
    # ax.set_xticks(range(0, duration * fs + 1, 2*fs), labels=['0', '2', '4', '6'])
    # ax.set_ylabel('Firing Rate (Hz)')
    # ax.set_xlabel('Time (s)')
    # plt.savefig('./figs/fr_ramp.jpg')
    # plt.close()

    # mn_pool.display_onion_skin_theory(spikes, duration, fs, './figs/onion_skin_ramp.svg')

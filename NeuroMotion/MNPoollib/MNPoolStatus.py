import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

import sys
sys.path.append('.')

from BioMime.utils.params import coeff_a, coeff_b
from NeuroMotion.MNPoollib.mn_params import DEPTH, ANGLE, MS_AREA, NUM_MUS, mn_default_settings


class MotoneuronPoolStatus:
    def __init__(self, N, ms_name, rr, rm, rp, pfr1, pfrd, mfr1, mfrd, gain, c_ipi, frs1, frsd, mode='exp', fibre_density=200, **kwargs):
        """
        N       Number of motor units
        ms_name Name of muscle
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
        self.next_spiking = np.ones((self.N, 1)) * np.iinfo(np.int32).max
        self.fr = np.zeros((self.N, 1))

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

    def generate_current_spikes(self, activation, fs, dt):
        """
        Args:
            activation  excitation in percentage MVC, (1, 1)
            dt        observation interval, e.g., 0.1s
        Return:
            spikes  binary array, (N, 1)

        status of next firing will be updated as well
        """

        # check next spiking
        current_spikes = np.zeros((self.N, 1))
        current_spikes[self.next_spiking <= 0] = 1

        # update fr
        prev_fr = self.fr  # previous fr
        self.fr = self._calculate_fr(activation)  # current fr

        # update next spiking
        self.next_spiking[self.fr <= 0] = np.iinfo(np.int32).max
        for mu in range(self.N):
            if activation > self.rte[mu]:
                if self.next_spiking[mu] <= 0 or self.next_spiking[mu] == np.iinfo(np.int32).max:
                    # has already fired or not been activated
                    # print(f"mu {mu}, next spiking {self.next_spiking[mu]}")
                    ipi = 1 / dt / self.fr[mu]
                    ipi = ipi + np.random.randn() * ipi * 1 / 6
                    self.next_spiking[mu] = int(ipi)
                    # print(f"ipi {ipi}, dt {dt}, fr, {self.fr[mu]}, self.next_spiking {self.next_spiking[mu]}")
                    # assert 1 == 2
                else:
                    # activated but not fired yet
                    # if prev_fr[mu] <= 0:
                    #     coeff = 1
                    # else:
                    #     coeff = self.fr[mu] / prev_fr[mu]
                    coeff = 1.0
                    self.next_spiking[mu] = int(self.next_spiking[mu] - 1 / (1 / fs) * dt * coeff)
        return current_spikes

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
    mn_pool = MotoneuronPoolStatus(num_mu, muscle, **mn_default_settings)

    properties = mn_pool.assign_properties()

    # Excitation
    fs = 2048           # Hz
    duration = 1        # s
    ext = np.linspace(0.0, 0.6, fs * duration)

    for i, current_ext in enumerate(ext):
        current_spikes = mn_pool.generate_current_spikes(current_ext, 1 / fs)

        active_mu = 0
        for sp in current_spikes:
            if sp == 1:
                active_mu += 1

        if i % 100 == 0:
            print(f"{active_mu} MUs activated at step {i}.")

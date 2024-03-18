import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from easydict import EasyDict as edict

import sys

sys.path.append(".")

from BioMime.utils.params import coeff_a, coeff_b
from NeuroMotion.MNPoollib.mn_params import (
    DEPTH,
    ANGLE,
    MS_AREA,
    NUM_MUS,
    mn_default_settings,
)


class MotoneuronPoolStatus:
    def __init__(
        self,
        N,
        ms_name,
        rr,
        rm,
        rp,
        pfr1,
        pfrd,
        mfr1,
        mfrd,
        gain,
        c_ipi,
        frs1,
        frsd,
        mode="exp",
        fibre_density=200,
        device=torch.device("cuda"),
        dtype=torch.float32,
        **kwargs,
    ):
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

        self.dev_dtype = {
            "device": device,
            "dtype": dtype,
        }

        self._init_pool()

    def get_num_mu(self):
        return self.N

    def get_properties(self):
        return self.properties

    def _init_pool(self, mode="ls2n"):
        self._init_recruitment_threshold(mode)
        self._init_frs(mode)
        self._init_phys_params()
        self.next_spiking = (
            torch.ones((self.N, 1), **self.dev_dtype) * torch.iinfo(torch.int32).max
        )
        self.fr = torch.zeros((self.N, 1), **self.dev_dtype)

    def _init_phys_params(self):
        num_fb = np.round(MS_AREA[self.ms_name] * self.fibre_density)
        self.phys_params = edict(
            {
                "num_fb": torch.tensor(num_fb, **self.dev_dtype),
                "depth": torch.tensor(DEPTH[self.ms_name], **self.dev_dtype),
                "angle": torch.tensor(ANGLE[self.ms_name], **self.dev_dtype),
                "iz": torch.tensor([0.5, 0.1], **self.dev_dtype),
                "len": torch.tensor([1.0, 0.05], **self.dev_dtype),
                "cv": torch.tensor([4, 0.3], **self.dev_dtype),
            }
        )

    def _init_recruitment_threshold(self, mode):
        if mode == "ls2n":
            rt = (self.rm / self.rr) * torch.exp(
                torch.arange(self.N, **self.dev_dtype)
                * torch.log(torch.tensor(self.rr, **self.dev_dtype))
                / (self.N - 1)
            )
            self.rte = (self.rm / self.rr) * (
                torch.exp(
                    torch.arange(self.N, **self.dev_dtype)
                    * torch.log(torch.tensor(self.rr + 1, **self.dev_dtype))
                    / self.N
                )
                - 1
            )
            self.rte = self.rte * torch.max(rt) / torch.max(self.rte)
        elif mode == "fuglevand":
            self.rte = torch.exp(
                torch.arange(1, self.N + 1, **self.dev_dtype)
                * torch.log(torch.tensor(self.rr, **self.dev_dtype))
                / self.N
            )
        self.rte = self.rte.reshape(self.N, 1)

    def _init_frs(self, mode):
        if self.fr_mode == "linear":
            self.minfr = torch.linspace(
                self.mfr[0], self.mfr[0] - self.mfr[1], self.N, **self.dev_dtype
            )  # linear
            self.minfr = self.minfr.reshape(self.N, 1)
            self.maxfr = (
                self.pfr[0] - self.pfr[1] * self.rte / self.rte[-1]
            )  # PFRi = PFR1 - PFRD * RTEi / RTEn
            self.maxfr = self.maxfr.reshape(self.N, 1)
            self.slope_fr = torch.ones((self.N, 1), **self.dev_dtype) * self.gain
        elif self.fr_mode == "exp":
            self.minfr = self.mfr[0] - self.mfr[1] * self.rte / torch.max(self.rte)
            self.maxfr = self.pfr[0] - self.pfr[1] * self.rte / torch.max(self.rte)
            self.slope_fr = self.frs[0] - self.frs[1] * self.rte / torch.max(self.rte)
        if mode == "fuglevand":
            self.rte2mvc = (
                1.0 - (self.pfr[-1] - self.mfr[-1]) / self.slope_fr[-1]
            ) / self.rte[
                -1
            ]  # Coefficient that converts RTE to percentage in MVC
            self.rte = (
                self.rte * self.rte2mvc
            )  # RTE now in % MVC and consistent with common drive units

    def _calculate_fr(self, E):
        """
        Args:
            E       excitation in percentage MVC, (1, T)

        Return:
            fr      (N, T)
        """

        if E.ndim == 1:
            E = E.reshape(1, E.shape[-1])

        fr = torch.minimum(self.maxfr, self.minfr + (E - self.rte) * self.slope_fr)
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
        current_spikes = (self.next_spiking <= 0).float()

        # update fr
        self.fr = self._calculate_fr(activation)  # current fr

        # update next spiking
        self.next_spiking[self.fr <= 0] = torch.iinfo(torch.int32).max

        # calculate ipi for activated motor units

        activated_mu = activation > self.rte
        just_fired_or_activated = (self.next_spiking <= 0) | (
            self.next_spiking == torch.iinfo(torch.int32).max
        )
        update_mask = activated_mu & just_fired_or_activated

        tic = time.time()
        ipi = 1 / dt / self.fr[update_mask]
        ipi += torch.randn_like(ipi) * ipi * 1 / 6
        self.next_spiking[update_mask] = ipi.floor()
        toc = time.time()

        # update next spiking for activated but not fired motor units
        activated_not_fired = activated_mu & ~just_fired_or_activated
        coeff = torch.ones_like(self.next_spiking[activated_not_fired])
        self.next_spiking[activated_not_fired] = (
            self.next_spiking[activated_not_fired] - 1 / (1 / fs) * dt * coeff
        ).floor()

        print(toc - tic)

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
        P = torch.exp(
            torch.log(torch.tensor(self.rp, **self.dev_dtype))
            / self.N
            * torch.arange(1, self.N + 1, **self.dev_dtype)
        )
        num = torch.round(num * P / torch.sum(P))
        depth = (
            torch.rand(self.N, **self.dev_dtype) * (config.depth[1] - config.depth[0])
            + config.depth[0]
        )  # Uniform distribution
        angle = (
            torch.rand(self.N, **self.dev_dtype) * (config.angle[1] - config.angle[0])
            + config.angle[0]
        )  # Uniform distribution
        iz = (
            torch.randn(self.N, **self.dev_dtype) * config.iz[1] + config.iz[0]
        )  # Normal distribution
        len = (
            torch.randn(self.N, **self.dev_dtype) * config.len[1] + config.len[0]
        )  # Normal distribution
        cv = torch.sort(
            torch.randn(self.N, **self.dev_dtype) * config.cv[1] + config.cv[0]
        )[
            0
        ]  # Normal distribution

        mn = {
            "num": self._normalise(normalise, torch.log(num), *nrange, label="num"),
            "depth": self._normalise(normalise, depth, *nrange, label="depth"),
            "angle": self._normalise(normalise, angle, *nrange, label="angle"),
            "iz": self._normalise(normalise, iz, *nrange, label="iz"),
            "len": self._normalise(normalise, len, *nrange, label="len"),
            "cv": self._normalise(normalise, cv, *nrange, label="cv"),
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


if __name__ == "__main__":

    # Test example
    num_mu = 186
    muscle = "ECRL"
    mn_pool = MotoneuronPoolStatus(num_mu, muscle, **mn_default_settings)

    properties = mn_pool.assign_properties()

    # Excitation
    fs = 2048  # Hz
    duration = 1  # s
    ext = np.linspace(0.0, 0.6, fs * duration)

    for i, current_ext in enumerate(ext):
        current_spikes = mn_pool.generate_current_spikes(current_ext, 1 / fs, 0.1)

        active_mu = 0
        for sp in current_spikes:
            if sp == 1:
                active_mu += 1

        if i % 100 == 0:
            print(f"{active_mu} MUs activated at step {i}.")

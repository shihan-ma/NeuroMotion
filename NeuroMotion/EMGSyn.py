import argparse
import os
import sys
import torch
sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np

from easydict import EasyDict as edict
from BioMime.models.generator import Generator
from BioMime.utils.basics import update_config, load_generator
from BioMime.utils.plot_functions import plot_muaps
from NeuroMotion.MNPoollib.MNPoolStatus import MotoneuronPoolStatus
from NeuroMotion.MNPoollib.mn_params import mn_default_settings


def plot_emg(emg, path, fs=2048, time=None, figsize=(15, 6), percentile=100, svg=False, suffix="ms"):
    """
    emg: n_time, n_channels
    """
    n_time, n_channels = emg.shape
    if time is None:
        time = np.arange(n_time) / fs
    scale = np.abs(emg).max()
    ax = plt.subplots(1, 1, figsize=figsize)[1]
    ax.plot(time, emg / scale + np.arange(n_channels))

    ax.set(
        xlabel="Time [s]",
        ylabel="Channel",
        xlim=(time[0], time[-1]),
    )

    plt.savefig(os.path.join(path, "emg_{}.jpg".format(suffix)))
    if svg:
        plt.savefig(os.path.join(path, "emg_{}.jpg".format(suffix)))
    plt.close()


class EMGSynthesiser:
    def __init__(
            self,
            MNPool, 
            mnpool_kwargs, 
            generator,
            biomime_cfg,
            fs,
            win_len = 1.0,
            device = "cpu",
    ) -> None:
        """ EMG synthesiser

        Arguments
        ---------
        MNPool : class of motoneuron pool
            TODO: abstract class to include both Fuglevand"s model and LIF
        mnpool_kwargs : dict
            muscle name as the key and mnpool properties ad the values
            muscle name choices: ["ECRB", "ECRL", "PL", "FCU", "ECU", "EDCI", "FDSI"]
        generator : Generator of BioMime, instantiated
        """

        self._init_mn_pool(MNPool, mnpool_kwargs)
        self._init_muap_latent(mnpool_kwargs, biomime_cfg)

        self.generator = generator
        self.fs = fs
        self.win_len = int(self.fs * win_len)

        self.device = device
        if device == "cuda":
            self.generator.cuda()

        # states
        # keep one state as the next spiking time
        # one state to keep the previous spike trains
        # one state to keep the previous EMG (emg memory parameter as the length of the window we are going to keep) - dequeue
        # the EMG window should contain two parts, including a causal part and one that contains the tail of MUAP (96 samples)
        self.emg = np.zeros((10, 32, win_len + 96))
        self.full_emg = []

    def _init_mn_pool(self, MNPool, mnpool_kwargs):
        self.mn_pool = {}
        for k, v in mnpool_kwargs.items():
            mn_pool = MNPool(**v)
            num_mu = mn_pool.N

            properties = mn_pool.assign_properties()
            mn_pool.num = torch.from_numpy(properties["num"]).reshape(num_mu, 1)
            mn_pool.depth = torch.from_numpy(properties["depth"]).reshape(num_mu, 1)
            mn_pool.angle = torch.from_numpy(properties["angle"]).reshape(num_mu, 1)
            mn_pool.iz = torch.from_numpy(properties["iz"]).reshape(num_mu, 1)
            mn_pool.cv = torch.from_numpy(properties["cv"]).reshape(num_mu, 1)
            mn_pool.length = torch.from_numpy(properties["len"]).reshape(num_mu, 1)

            self.mn_pool[k] = mn_pool

    def _init_muap_latent(self, mnpool_kwargs, biomime_cfg):
        self.muap_latent = {}
        for k, v in mnpool_kwargs.items():
            self.muap_latent[k] = torch.randn(v["N"], biomime_cfg.Model.Generator.Latent)

    def update_emg(self, muscle_lengths, activations, dt=None):
        muaps = self.generate_muap(muscle_lengths)
        spikes = self.generate_spikes(activations, dt)
        emg = self.generate_emg(muaps, spikes)

        if dt is None:
            dt = 1 / self.fs
        n_steps = int(dt * self.fs)
        self.emg[:, :, :-n_steps] = self.emg[:, :, n_steps:]
        self.emg[:, :, -96:] = self.emg[:, :, -96:] + emg

        cur_emg = self.emg[:, :, -96].copy()
        self.full_emg.append(cur_emg)

        return cur_emg

    def generate_muap(self, muscle_lengths):
        muaps = {}
        for ms, pool in self.mn_pool.items():
            num_mu = pool.N
            ms_len = muscle_lengths[ms]
            zi = self.muap_latent[ms]

            # physiological parameters as input condition
            cond = torch.hstack((
                pool.num,
                pool.depth * (1 / np.sqrt(ms_len) + 1e-8),
                pool.angle,
                pool.iz,
                pool.cv * (1 / ms_len + 1e-8),
                pool.length * ms_len,
            ))

            if self.device == "cuda":
                cond = cond.cuda()
                zi = zi.cuda()

            sim = self.generator.sample(num_mu, cond.float(), cond.device, zi)
            if self.device == "cuda":
                sim = sim.permute(0, 2, 3, 1).cpu().detach().numpy()
            else:
                sim = sim.permute(0, 2, 3, 1).detach().numpy()
            muaps[ms] = sim

        self.current_muaps = muaps

        return muaps

    def generate_spikes(self, activations, dt=None):
        spikes = {}
        if dt is None:
            dt = 1 / self.fs
        for ms, pool in self.mn_pool.items():
            current_spikes = pool.generate_current_spikes(activations[ms], self.fs, dt)
            spikes[ms] = current_spikes

        self.current_spikes = spikes

        return spikes

    def generate_emg(self, muaps, spike_trains):
        emg = np.zeros((10, 32, 96))
        for muap, spike in zip(muaps.values(), spike_trains.values()):
            current_emg = spike[..., None, None] * muap
            emg = emg + current_emg.sum(0)
        return emg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EMG signals from movements")
    parser.add_argument("--cfg", type=str, default="config.yaml", help="Name of configuration file")
    parser.add_argument("--model_pth", default="./ckp/model_linear.pth", type=str, help="path of best pretrained BioMime model")
    parser.add_argument('--device', default='cuda', type=str, help='cuda|cpu')

    args = parser.parse_args()
    cfg = update_config("./ckp/" + args.cfg)

    mn_properties = {
        "ECRB": {
            "N": 186,
            "ms_name": "ECRB",
            **mn_default_settings,
        },
    }

    generator = Generator(cfg.Model.Generator)
    generator = load_generator(args.model_pth, generator, args.device)
    generator.eval()

    # Excitation
    fs = 2048           # Hz
    duration = 1        # s
    ext = np.linspace(0.0, 0.6, fs * duration)

    emg_synthesiser = EMGSynthesiser(
        MotoneuronPoolStatus,
        mn_properties,
        generator,
        cfg,
        fs=fs,
        win_len=1,
        device=args.device,
    )

    ms_lengths = {
        "ECRB": 1.0,
    }

    for i, current_ext in enumerate(ext):
        current_ext_dict = {"ECRB": current_ext}
        cur_emg = emg_synthesiser.update_emg(ms_lengths, current_ext_dict)

        current_spikes = emg_synthesiser.current_spikes
        active_mu = 0
        for sp in current_spikes["ECRB"]:
            if sp == 1:
                active_mu += 1

        if i % 100 == 0:
            print(f"{active_mu} MUs activated at step {i}.")

    # full_emg = np.stack(emg_synthesiser.full_emg)

    # res_pth = "./figs"
    # plot_emg(full_emg[:, 6, 10:30:2], res_pth, suffix="ECRB")

import argparse
import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from easydict import EasyDict as edict
from scipy.signal import butter, filtfilt

import sys
sys.path.append('.')

from NeuroMotion.MSKlib.MSKpose import MSKModel
from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
from NeuroMotion.MNPoollib.mn_utils import plot_spike_trains, generate_emg_mu
from NeuroMotion.MNPoollib.mn_params import DEPTH, ANGLE, MS_AREA, NUM_MUS
from BioMime.models.generator import Generator
from BioMime.utils.basics import update_config, load_generator
from BioMime.utils.plot_functions import plot_muaps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate EMG signals from movements')
    parser.add_argument('--cfg', type=str, default='config.yaml', help='Name of configuration file')
    parser.add_argument('--model_pth', default='./ckp/model_linear.pth', type=str, help='path of best pretrained BioMime model')
    parser.add_argument('--res_path', default='./res', type=str, help='path of result folder')
    parser.add_argument('--device', default='cuda', type=str, help='cuda|cpu')

    args = parser.parse_args()
    cfg = update_config('./ckp/' + args.cfg)

    # PART ONE: Define MSK model, movements, and extract param changes
    msk = MSKModel()
    poses = ['default', 'default+flex', 'default', 'default+ext', 'default']
    durations = [2] * 4
    duration = np.sum(durations)
    fs_mov = 5
    steps = fs_mov * duration
    ms_labels = ['ECRB', 'ECRL', 'PL', 'FCU', 'ECU', 'EDCI', 'FDSI']

    msk.sim_mov(fs_mov, poses, durations)
    ms_lens = msk.mov2len(ms_labels=ms_labels)
    changes = msk.len2params()

    # PART TWO: Define the MotoneuronPool of one muscle
    ms_label = 'FDSI'
    num_mus = NUM_MUS[ms_label]
    rr = 50
    rm = 0.75
    rp = 100
    pfr1 = 35
    pfrd = 12
    mfr1 = 8
    mfrd = 0
    ge = 30         # 0.3 per % MVC
    c_ipi = 0.1

    pfr1 = 40
    pfrd = 10
    mfr1 = 10
    mfrd = 5
    frs1 = 50
    frsd = 20

    mn_pool = MotoneuronPool(num_mus, rr, rm, rp, pfr1, pfrd, mfr1, mfrd, ge, c_ipi, frs1, frsd)
    # Assign physiological properties
    fibre_density = 200     # 200 fibres per mm^2
    num_fb = np.round(MS_AREA[ms_label] * fibre_density)    # total number within one muscle
    config = edict({
        'num_fb': num_fb,
        'depth': DEPTH[ms_label],
        'angle': ANGLE[ms_label],
        'iz': [0.5, 0.1],
        'len': [1.0, 0.05],
        'cv': [4, 0.3]      # Recommend not setting std too large. cv range in training dataset is [3, 4.5]
    })

    properties = mn_pool.assign_properties(config, normalise=True)
    num = torch.from_numpy(properties['num']).reshape(num_mus, 1).repeat(1, steps)
    depth = torch.from_numpy(properties['depth']).reshape(num_mus, 1).repeat(1, steps)
    angle = torch.from_numpy(properties['angle']).reshape(num_mus, 1).repeat(1, steps)
    iz = torch.from_numpy(properties['iz']).reshape(num_mus, 1).repeat(1, steps)
    cv = torch.from_numpy(properties['cv']).reshape(num_mus, 1).repeat(1, steps)
    length = torch.from_numpy(properties['len']).reshape(num_mus, 1).repeat(1, steps)

    fs = 2048
    mn_pool.init_twitches(fs)
    mn_pool.init_quisistatic_ef_model()
    time_samples = fs * duration
    ext = np.concatenate((np.linspace(0, 0.8, round(time_samples / 2)), np.linspace(0.8, 0, round(time_samples / 2))))      # percentage MVC
    ext_new, spikes, fr, ipis = mn_pool.generate_spike_trains(ext, fit=False)
    plot_spike_trains(spikes, './figs/spikes_{}.jpg'.format(ms_label))

    # PART THREE: Simulate MUAPs using BioMime during the movement
    if ms_label == 'FCU_u' or ms_label == 'FCU_h':
        tgt_ms_labels = ['FCU'] * num_mus
    else:
        tgt_ms_labels = [ms_label] * num_mus

    ch_depth = changes['depth'].loc[:, tgt_ms_labels]
    ch_cv = changes['cv'].loc[:, tgt_ms_labels]
    ch_len = changes['len'].loc[:, tgt_ms_labels]

    # Model
    generator = Generator(cfg.Model.Generator)
    generator = load_generator(args.model_pth, generator, args.device)
    generator.eval()

    # Device
    if args.device == 'cuda':
        assert torch.cuda.is_available()
        generator.cuda()

    if not os.path.exists(args.res_path):
        os.mkdir(args.res_path)

    # Filtering, not required
    # low-pass filtering for smoothing
    time_length = 96
    fs = 2048
    T = time_length / fs
    cutoff = 800
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    order = 4
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    start_time = time.time()

    muaps = []
    for sp in tqdm(range(steps), dynamic_ncols=True, desc='Simulating MUAPs during dynamic movement...'):
        cond = torch.vstack((
            num[:, sp],
            depth[:, sp] * ch_depth.iloc[sp, :].values,
            angle[:, sp],
            iz[:, sp],
            cv[:, sp] * ch_cv.iloc[sp, :].values,
            length[:, sp] * ch_len.iloc[sp, :].values,
        )).transpose(1, 0)
        zi = torch.randn(num_mus, cfg.Model.Generator.Latent)

        if args.device == 'cuda':
            cond, zi = cond.cuda(), zi.cuda()

        sim = generator.sample(num_mus, cond.float(), cond.device, zi)

        if args.device == 'cuda':
            sim = sim.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            sim = sim.permute(0, 2, 3, 1).detach().numpy()

        num_mu_dim, n_row_dim, n_col_dim, n_time_dim = sim.shape
        sim = filtfilt(b, a, sim.reshape(-1, n_time_dim))
        muaps.append(sim.reshape(num_mu_dim, n_row_dim, n_col_dim, n_time_dim).astype(np.float32))

    muaps = np.array(muaps)
    muaps = np.transpose(muaps, (1, 0, 2, 3, 4))
    print('--- %s seconds ---' % (time.time() - start_time))

    # plot muaps
    plot_muaps(muaps, args.res_path, np.arange(0, 100, 20), np.arange(0, steps, 5), suffix=ms_label)

    # PART FOUR: generate EMG signals
    _, _, n_row, n_col, time_length = muaps.shape
    emg = np.zeros((n_row, n_col, time_samples + time_length))
    for mu in np.arange(num_mus):
        emg = emg + generate_emg_mu(muaps[mu], spikes[mu], time_samples)

    print('All done, emg.shape: ', emg.shape)
    total_length = emg.shape[-1]
    t = np.linspace(0, duration, total_length)
    fig = plt.figure()
    row, col = 5, 10
    plt.plot(t, emg[row, col])
    plt.xlabel('time')
    plt.ylabel('emg')
    plt.savefig(os.path.join(args.res_path, 'emg_{}.jpg'.format(ms_label)))

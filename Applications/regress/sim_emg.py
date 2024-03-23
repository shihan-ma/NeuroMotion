"""
STEP THREE
data_pth needs to be customised

Generate EMG signals given default motor unit pools and changes of parameters during a movement
"""


import argparse
import os
import pickle
import sys
import torch

import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

sys.path.append('.')

from BioMime.models.generator import Generator
from BioMime.utils.basics import update_config, load_generator
from NeuroMotion.MNPoollib.mn_utils import generate_emg_mu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate EMG signals given experimental results')
    parser.add_argument('--cfg', type=str, default='config.yaml', help='Name of configuration file')
    parser.add_argument('--model_pth', required=True, type=str, help='file of best model')
    parser.add_argument('--subject_id', required=True, type=str, help='index of subject')
    parser.add_argument('--mn_pth', default='./Application/regress/mn_pool', type=str, help='file path of motoneuron pool files')
    parser.add_argument('--data_pth', default='/home/xx/sub_id/', type=str, help='path of dataset that contains changes of physiological parameters and muscle activations')
    parser.add_argument('--num_trials', default=5, type=int, help='number of trials for each movement')
    parser.add_argument('--repeats', default=1, type=int, help='number of repeats for each trial and each movement')
    parser.add_argument('--mov_type', required=True, type=str, help='type of movement')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')

    args = parser.parse_args()
    cfg = update_config('./config/' + args.cfg)

    data_pth = args.data_pth
    mn_pth = args.mn_pth
    mov_type = args.mov_type
    num_trials = args.num_trials
    subject_id = args.subject_id
    repeats = args.repeats

    # Device
    if args.device == 'cuda':
        assert torch.cuda.is_available()

    # BioMime Model
    generator = Generator(cfg.Model.Generator)
    generator = load_generator(args.model_pth, generator, args.device)
    generator.eval()
    if args.device == 'cuda':
        generator.cuda()

    # low-pass filtering for smoothing
    time_length = 96
    fs = 2048
    T = time_length / fs
    cutoff = 800
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    order = 4
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # muscles to traverse
    ms_labels = ['ECRB', 'ECRL', 'PL', 'FCU_u', 'FCU_h', 'ECU', 'ED', 'FDS']

    for trial in range(num_trials):
        res_pth_trial = os.path.join(data_pth, 'trial{}'.format(trial))
        # Load 'changes' paired with movement, contains change for all muscles
        with open(os.path.join(data_pth, 'trial{}/changes.pkl'.format(trial)), 'rb') as file:
            changes = pickle.load(file)
        mov_steps = changes['mov_steps']

        if repeats < 2:
            repeats = 1
        for ms_id, ms in enumerate(ms_labels):
            # load motor unit pools
            with open(os.path.join(mn_pth, 'ms_{}_pool.pkl'.format(ms)), 'rb') as file:
                mn = pickle.load(file)
            num_mu = mn.get_num_mu()
            properties = mn.get_properties()

            # init mu properties
            num = torch.from_numpy(properties['num']).reshape(num_mu, 1).repeat(1, mov_steps)
            depth = torch.from_numpy(properties['depth']).reshape(num_mu, 1).repeat(1, mov_steps)
            angle = torch.from_numpy(properties['angle']).reshape(num_mu, 1).repeat(1, mov_steps)
            iz = torch.from_numpy(properties['iz']).reshape(num_mu, 1).repeat(1, mov_steps)
            cv = torch.from_numpy(properties['cv']).reshape(num_mu, 1).repeat(1, mov_steps)
            length = torch.from_numpy(properties['len']).reshape(num_mu, 1).repeat(1, mov_steps)
            # changes of parameters
            if ms == 'FDS':
                lbl = ['FDSI'] * num_mu
            elif ms == 'ED':
                lbl = ['EDCI'] * num_mu
            elif ms == 'FCU_u' or ms == 'FCU_h':
                lbl = ['FCU'] * num_mu
            else:
                lbl = [ms] * num_mu
            ch_depth = changes['depth'].loc[:, lbl]
            ch_cv = changes['cv'].loc[:, lbl]
            ch_lens = changes['len'].loc[:, lbl]

            assert ch_depth.shape[0] == mov_steps and ch_cv.shape[0] == mov_steps and ch_lens.shape[0] == mov_steps, (ch_depth.shape[0], ch_cv.shape[0], ch_lens.shape[0], mov_steps)

            # sample latents
            zi = torch.randn(num_mu, cfg.Model.Generator.Latent)

            sim_muaps = []

            for sp in tqdm(range(mov_steps), dynamic_ncols=True):
                cond = torch.vstack((
                    num[:, sp],
                    depth[:, sp] * ch_depth.iloc[sp, :].values,
                    angle[:, sp],
                    iz[:, sp],
                    cv[:, sp] * ch_cv.iloc[sp, :].values,
                    length[:, sp] * ch_lens.iloc[sp, :].values,
                )).transpose(1, 0)

                if args.device == 'cuda':
                    cond = cond.cuda()

                if args.device == 'cuda':
                    zi = zi.cuda()
                sim = generator.sample(num_mu, cond.float(), cond.device, zi)
                if args.device == 'cuda':
                    sim = sim.permute(0, 2, 3, 1).cpu().detach().numpy()
                else:
                    sim = sim.permute(0, 2, 3, 1).detach().numpy()

                num_mu_dim, n_row_dim, n_col_dim, n_time_dim = sim.shape
                sim = sim.reshape(-1, n_time_dim)
                sim = filtfilt(b, a, sim.reshape(-1, n_time_dim))
                sim_muaps.append(sim.reshape(num_mu_dim, n_row_dim, n_col_dim, n_time_dim).astype(np.float32))

            sim_muaps = np.array(sim_muaps)
            sim_muaps = np.transpose(sim_muaps, (1, 0, 2, 3, 4))    # mu, time, row, col, duration

            for r in range(repeats):
                print('movement {}, trial {}, repeat {}'.format(mov_type, trial, r))
                if repeats > 1:
                    res_pth_repeat = os.path.join(data_pth, f"trial{trial}", f"repeat{r}")
                    if not os.path.exists(res_pth_repeat):
                        os.mkdir(res_pth_repeat)
                else:
                    res_pth_repeat = res_pth_trial
                print('generating muaps of muscle {}...'.format(ms))

                # load activations
                act_fl = os.path.join(res_pth_trial, 'act_emg.pkl')
                with open(act_fl, 'rb') as file:
                    act = pickle.load(file)
                ext = act[ms]
                spikes = mn.generate_spike_trains(ext)[1]
                num_spikes = np.zeros(len(spikes))
                for i in range(len(spikes)):
                    num_spikes[i] = np.sum(spikes[i])

                # generate EMG
                print('generating EMG of muscle {}...'.format(ms))
                _, _, n_row, n_col, time_length = sim_muaps.shape
                time_steps = ext.shape[-1]
                emg = np.zeros((n_row, n_col, time_steps + time_length))
                for mu in np.arange(num_mu):
                    emg = emg + generate_emg_mu(sim_muaps[mu], spikes[mu], time_steps)

                emg_data = {
                    'emg': emg,
                    'ext': ext,
                    'spikes': spikes,
                }
                print('saving data...')
                with open(os.path.join(res_pth_repeat, 'ms_{}.pkl'.format(ms)), 'wb') as file:
                    pickle.dump(emg_data, file, protocol=pickle.HIGHEST_PROTOCOL)
                print('Done!')

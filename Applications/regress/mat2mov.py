"""
STEP ONE
data_pth needs to be customised

Export mov.csv from angle.mat files
Save envelopes of EMG data and raw EMG data
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

from scipy import signal


# subject and protocol info
subject_id = sys.argv[1]
protocols = ['mcp', 'flex_ext']

# SET DATA PATH
data_pth = '/home/xx/sub_id/'

# Load angle file
angle_file = os.path.join(data_pth, 'angle.mat')
mat_file = sio.loadmat(angle_file)
angs = mat_file['angle'][0]
num_trials = len(angs) // 3

emg_file = os.path.join(data_pth, 'EMG_data.mat')
mat_file = sio.loadmat(emg_file)['EMG_data']
emg = mat_file[0]       # 15, 30,000 by 6

emg_raw_file = os.path.join(data_pth, 'EMG_raw.mat')
mat_file = sio.loadmat(emg_raw_file)['EMG_data']
emg_raw = np.abs(mat_file[0])   # 15, 30,000 by 6

# default poses
pose_pth = './Applications/regress/poses.csv'
pose_basis = pd.read_csv(pose_pth)
dof = len(pose_basis.iloc[:, 0])

# Process each trial
time = 15
fs_init = 2000      # for few-channel EMG
fs_tgt = 2048       # for few-channel EMG
for p_i, protocol in enumerate(protocols):
    dest_pth = os.path.join(data_pth, protocol)
    if not os.path.exists(dest_pth):
        os.mkdir(dest_pth)
    for i in range(num_trials):
        cur_pth = os.path.join(dest_pth, 'trial{}'.format(i))
        if not os.path.exists(cur_pth):
            os.mkdir(cur_pth)
        cur_trial = p_i * num_trials + i
        emg_act = emg[cur_trial]
        emg_act[emg_act < 0] = 0.
        time_samples = emg_act.shape[0]
        emg_act = signal.resample(emg_act, int(time_samples / fs_init * fs_tgt))
        with open(os.path.join(cur_pth, 'few_channels.pkl'), 'wb') as file:
            pickle.dump(emg_act, file, protocol=pickle.HIGHEST_PROTOCOL)

        emg_raw_act = emg_raw[cur_trial]
        emg_raw_act[emg_raw_act < 0] = 0.
        time_samples = emg_raw_act.shape[0]
        emg_raw_act = signal.resample(emg_raw_act, int(time_samples / fs_init * fs_tgt))
        with open(os.path.join(cur_pth, 'few_channels_raw.pkl'), 'wb') as file:
            pickle.dump(emg_raw_act, file, protocol=pickle.HIGHEST_PROTOCOL)

        emg_dict = {'ECRB': emg_act[:, 5], 'ECRL': emg_act[:, 5], 'FDS': emg_act[:, 1], 'ED': emg_act[:, 4], 'ECU': emg_act[:, 3], 'FCU_u': emg_act[:, 2], 'FCU_h': emg_act[:, 2], 'PL': emg_act[:, 0]}
        with open(os.path.join(cur_pth, 'act_emg.pkl'), 'wb') as file:
            pickle.dump(emg_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        # resample angles and save data
        ang = signal.resample(angs[cur_trial], 750)
        ang1 = ang[:, 1]                         # 1-wrist
        ang1[ang1 > 65] = 65
        ang1[ang1 < -65] = -65
        ang2 = ang[:, 0]                         # 0-MCP
        ang2[ang2 > 90] = 90
        ang2[ang2 < -45] = -45

        time_samples = ang.shape[0]
        mov = np.zeros((time_samples, dof))
        mov[:, 2] = ang1                            # 2-flex/ext
        for df in [9, 13, 17, 21]:                  # MCPs
            mov[:, df] = ang2
        x_ = np.linspace(0, time, time_samples)
        mov = np.concatenate((x_[:, None], mov), axis=1)
        mov = pd.DataFrame(data=mov, columns=['time', *pose_basis.iloc[:, 0].tolist()])
        mov.to_csv(os.path.join(cur_pth, 'mov.csv'), sep='\t', index=False)
        print('trial {} mov saved.'.format(i))

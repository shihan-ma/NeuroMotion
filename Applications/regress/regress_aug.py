"""
Train ridge regressors on the augmented dataset

train dataset: half (2 trials) from exp, half (2 trials) from synthetic
test dataset: one trial from exp other than the trials in train dataset
"""


import argparse
import os
import pickle

import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.linear_model import Ridge
from scipy import signal
from scipy import stats
from scipy.signal import butter, filtfilt
from tqdm import tqdm


# Low-pass filter to smooth the signals
fs = 50.
cutoff = 5
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
order = 4
b, a = butter(order, normal_cutoff, btype="low", analog=False)

def low_pass_filter(sig):
    filtered_sig = filtfilt(b, a, sig)
    return filtered_sig


def extract_features(sig, win_len, overlap, mode="rms"):
    if mode == "rms":
        feats = sep_windows(sig, win_len, overlap, "rms")
    elif mode == "var":
        feats = sep_windows(sig, win_len, overlap, "var")
    elif mode == "td":
        feats = []
        feats.append(sep_windows(sig, win_len, overlap, "mav"))
        feats.append(sep_windows(sig, win_len, overlap, "zc"))
        feats.append(sep_windows(sig, win_len, overlap, "ssc"))
        feats.append(sep_windows(sig, win_len, overlap, "wl"))
        feats = np.hstack(feats)
    else:
        feats = sep_windows(sig, win_len, overlap, mode)
    return feats


def sep_windows(sig, win_len, overlap, mode="rms"):
    """
    Feature: rms

    sig: time_samples, channel
    win_len: samples in one window
    overlap: samples in one overlap

    return: [num_win, 1]
    """
    time_samples, num_channels = sig.shape
    if overlap > 0:
        num_win = (time_samples - win_len) // overlap + 1
    else:
        num_win = time_samples // win_len
        overlap = win_len

    feats = np.zeros((num_win, num_channels))
    for i in range(num_win):
        feat = sig[i * overlap: i * overlap + win_len]      # [win_len, channels]
        if mode == "rms":
            feat = np.sqrt(np.mean(feat ** 2, axis=0))
        elif mode == "mean":
            feat = np.mean(feat, axis=0)
        elif mode == "var":
            feat = np.var(feat, axis=0)
        elif mode == "mav":
            feat = np.mean(np.abs(feat), axis=0)
        elif mode == "zc":
            feat = zcross(feat)
        elif mode == "ssc":
            feat = ssc(feat)
        elif mode == "wl":
            feat = np.sum(np.abs(np.diff(feat, axis=0)), axis=0)
        else:
            AssertionError

        feats[i] = feat

    return feats


def zcross(sig):
    """
    zero crossing
    sig: [time_samples, channels]
    """

    time_samples, num_channels = sig.shape
    cross = np.zeros(num_channels)
    for i in range(time_samples - 1):
        for ch in range(num_channels):
            if sig[i, ch] * sig[i + 1, ch] < 0:
                cross[ch] = cross[ch] + 1
    return cross


def ssc(sig):
    """
    slope sign changes
    sig: [time_samples, channels]
    """

    time_samples, num_channels = sig.shape
    ss_changes = np.zeros(num_channels)
    for i in range(time_samples - 2):
        for ch in range(num_channels):
            if (sig[i + 1, ch] - sig[i, ch]) * (sig[i + 1, ch] - sig[i + 2, ch]) > 0:
                ss_changes[ch] = ss_changes[ch] + 1
    return ss_changes


def normalise_std(sig):
    """
    sig: time_samples by num_channels
    """
    for i in range(sig.shape[1]):
        sig[:, i] = sig[:, i] - np.mean(sig[:, i])
        sig[:, i] = sig[:, i] / np.std(sig[:, i])
    return sig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Regress joint angles from EMG signals")
    parser.add_argument("--subject_id", required=True, type=str, help="index of subject")
    parser.add_argument("--feat", default="rms", type=str, help="features for regression, rms, td, var")
    parser.add_argument("--data_pth", default="/home/xx/sub_id/", type=str, help="path of dataset that contains changes of physiological parameters and muscle activations")
    parser.add_argument("--num_trials", default=5, type=int, help="number of trials for each movement")
    parser.add_argument("--win_len", default=200, type=float, help="length of window in ms")
    parser.add_argument("--overlap", default=0, type=float, help="overlap of window in ms")
    parser.add_argument("--fs", default=2000, type=int, help="sampling frequency in Hz")
    parser.add_argument("--test_trial", required=True, type=int, help="experimental trial id to be tested")

    args = parser.parse_args()

    data_pth = args.data_pth
    feat_type = args.feat
    num_trials = args.num_trials
    test_trial = args.test_trial

    # Dataset
    # Load data
    ms_labels = ["ECRB", "ECRL", "PL", "FCU_u", "FCU_h", "ECU", "ED", "FDS"]
    eles = np.array([[3, 2], [0, 6], [6, 26], [8, 24], [1, 20], [9, 15]])  # adjust for each subject to match few_channels exp data - FCR(PL), FD, FCU, ECU, ED, ECRL
    ms_labels_ele = ["PL", "FDS", "FCU_u", "ECU", "ED", "ECRL"]

    for dof in ["flex_ext", "mcp"]:
        # exp dataset
        exp_train_trials = combinations(list(set(np.arange(num_trials)) - set([test_trial])), 2)

        for exp_train_trial in exp_train_trials:
            train_data = []
            train_label1 = []
            train_label2 = []
            exp_data = []
            exp_label1 = []
            exp_label2 = []

            for trial in range(num_trials):
                with open(os.path.join(data_pth, "trial{}".format(trial), "few_channels_raw.pkl"), "rb") as file:
                    emg_data = np.abs(pickle.load(file))      # to mV
                emg_data = normalise_std(emg_data)
                time_samples = emg_data.shape[0]
                cur_feat = extract_features(emg_data, int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode=feat_type)

                if trial == test_trial:
                    test_data = cur_feat
                else:
                    exp_data.append(cur_feat)
                if trial in exp_train_trial:
                    train_data.append(cur_feat)

                # labels
                mov = pd.read_csv(os.path.join(data_pth, "trial{}".format(trial), "{}.csv".format("mov")), sep="\t")
                tgt1 = signal.resample(mov.loc[:, "flexion"], time_samples)  # tgt angles first resampled to the same as emg
                label1 = sep_windows(tgt1[:, None], int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode="mean")
                tgt2 = signal.resample(mov.loc[:, "2mcp_flexion"], time_samples)  # tgt angles first resampled to the same as emg
                label2 = sep_windows(tgt2[:, None], int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode="mean")
                if trial == test_trial:
                    test_label1 = label1
                    test_label2 = label2
                else:
                    exp_label1.append(label1)
                    exp_label2.append(label2)
                if trial in exp_train_trial:
                    train_label1.append(label1)
                    train_label2.append(label2)

            sim_train_trials = combinations(list(set(np.arange(num_trials)) - set([test_trial])), 2)
            for sim_train_trial in tqdm(sim_train_trials, dynamic_ncols=True):
                train_data_cp = train_data
                train_label1_cp = train_label1
                train_label2_cp = train_label2
                for trial in range(num_trials):
                    cur_emg = np.zeros(1)
                    for ms in ms_labels:
                        with open(os.path.join(data_pth, "trial{}".format(trial), "ms_{}.pkl".format(ms)), "rb") as file:
                            emg_data = pickle.load(file)["emg"]
                            emg_data = np.transpose(emg_data[eles[:, 0], eles[:, 1]], (1, 0))
                            time_samples = emg_data.shape[0]
                        cur_emg = cur_emg + emg_data
                    cur_emg = normalise_std(cur_emg)
                    cur_feat = extract_features(cur_emg, int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode=feat_type)

                    if trial in sim_train_trial:
                        train_data_cp.append(cur_feat)
                    # labels
                    mov = pd.read_csv(os.path.join(data_pth, "trial{}".format(trial), "{}.csv".format("mov")), sep="\t")
                    tgt1 = signal.resample(mov.loc[:, "flexion"], time_samples)        # tgt angles first resampled to the same as emg
                    label1 = sep_windows(tgt1[:, None], int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode="mean")
                    tgt2 = signal.resample(mov.loc[:, "2mcp_flexion"], time_samples)        # tgt angles first resampled to the same as emg
                    label2 = sep_windows(tgt2[:, None], int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode="mean")

                    if trial in sim_train_trial:
                        train_label1_cp.append(label1)
                        train_label2_cp.append(label2)

                train_data_cp = np.vstack(train_data_cp)
                train_label1_cp = np.vstack(train_label1_cp)
                train_label2_cp = np.vstack(train_label2_cp)
                exp_data = np.vstack(exp_data)
                exp_label1 = np.vstack(exp_label1)
                exp_label2 = np.vstack(exp_label2)

                # Model and Train
                # DoF 1
                clf1 = Ridge(alpha=1.0)
                clf1.fit(train_data_cp, train_label1_cp[:, 0])
                clf_exp1 = Ridge(alpha=1.0)
                clf_exp1.fit(exp_data, exp_label1[:, 0])

                # DoF 2
                clf2 = Ridge(alpha=1.0)
                clf2.fit(train_data_cp, train_label2_cp[:, 0])
                clf_exp2 = Ridge(alpha=1.0)
                clf_exp2.fit(exp_data, exp_label2[:, 0])

                # Test
                pred_clf1 = clf1.predict(test_data)
                pred_clf1 = low_pass_filter(pred_clf1)
                pred_clf_exp1 = clf_exp1.predict(test_data)
                pred_clf_exp1 = low_pass_filter(pred_clf_exp1)

                pred_clf2 = clf2.predict(test_data)
                pred_clf2 = low_pass_filter(pred_clf2)
                pred_clf_exp2 = clf_exp2.predict(test_data)
                pred_clf_exp2 = low_pass_filter(pred_clf_exp2)

                # R2
                r_clf1 = stats.pearsonr(test_label1[:, 0], pred_clf1)[0]
                r_clf_exp1 = stats.pearsonr(test_label1[:, 0], pred_clf_exp1)[0]

                r_clf2 = stats.pearsonr(test_label2[:, 0], pred_clf2)[0]
                r_clf_exp2 = stats.pearsonr(test_label2[:, 0], pred_clf_exp2)[0]

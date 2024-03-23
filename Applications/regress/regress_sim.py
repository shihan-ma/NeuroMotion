"""
Train ridge regressors to regress joint angles from synthetic EMG data
1. Preprocess simulated data
2. Extract RMS features from the windows for each DoF
3. Train regressors on each synthetic dataset and save the regressor
4. Test on experimental dataset
"""


import argparse
import pickle
import os

import numpy as np
import pandas as pd

from scipy import signal
from scipy import stats
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge


# Filtering
# Low-pass filter to smooth the signals
fs = 50.
cutoff = 5.
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
    Input
    sig: time_samples, channel
    win_len: number of samples in one window
    overlap: number of samples in one overlap (step)

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
        feat = sig[i * overlap: i * overlap + win_len]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ridge regressor to regress joint angles from synthetic EMG signals")
    parser.add_argument("--subject_id", required=True, type=int, help="index of subject")
    parser.add_argument("--feat", default="rms", type=str, help="features for regression, rms, td, var")
    parser.add_argument("--data_pth", default="/home/xx/sub_id/", type=str, help="path of dataset that contains changes of physiological parameters and muscle activations")
    parser.add_argument("--win_len", default=200, type=float, help="length of window in ms")
    parser.add_argument("--overlap", default=50, type=float, help="overlap of window in ms")
    parser.add_argument("--fs", default=2048, type=int, help="sampling frequency in Hz")
    parser.add_argument("--save", default=False, action="store_true", help="flag to save the regressors")
    parser.add_argument("--low_pass", default=False, action="store_true", help="flag to low-pass filter the synthesised EMG signals")
    parser.add_argument("--test_trial", required=True, type=int, help="experimental trial id to be tested")
    parser.add_argument("--repeat", default=-1, type=int, help="experimental repeat id to be tested")

    args = parser.parse_args()

    feat_type = args.feat
    repeat = args.repeat
    test_trial = args.test_trial
    data_pth = args.data_pth

    # Preparation for the dataset
    eles = np.array([[3, 2], [0, 6], [6, 26], [8, 24], [1, 20], [9, 15]])  # adjust for each subject to match few_channels exp data - FCR(PL), FD, FCU, ECU, ED, ECRL
    ms_labels = ["ECRB", "ECRL", "PL", "FCU_u", "FCU_h", "ECU", "ED", "FDS"]
    # FCR(PL), FD, FCU, ECU, ED, ECRL - for selecting few channels from simulated data
    ms_labels_ele = ["PL", "FDS", "FCU_u", "ECU", "ED", "ECRL"]

    # DoF1: flex/ext
    data_pth1 = os.path.join(data_pth, "flex_ext")
    cur_emg = np.zeros(1)
    for ms in ms_labels:
        if repeat >= 0:
            new_data_pth = os.path.join(data_pth1, "trial{}".format(test_trial), f"repeat{repeat}", "ms_{}.pkl".format(ms))
        else:
            new_data_pth = os.path.join(data_pth1, "trial{}".format(test_trial), "ms_{}.pkl".format(ms))
        with open(new_data_pth, "rb") as file:
            emg_data = pickle.load(file)["emg"]
            emg_data = np.transpose(emg_data[eles[:, 0], eles[:, 1]], (1, 0))
        cur_emg = cur_emg + emg_data

    # Normalise
    for i in range(len(ms_labels_ele)):
        cur_emg[:, i] = cur_emg[:, i] - np.mean(cur_emg[:, i])
        cur_emg[:, i] = cur_emg[:, i] / np.std(cur_emg[:, i])
    time_samples = cur_emg.shape[0]
    # Extract features
    train_data1 = extract_features(cur_emg, int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode=feat_type)

    # exp data
    with open(os.path.join(data_pth1, "trial{}".format(args.test_trial), "few_channels_raw.pkl"), "rb") as file:
        exp_emg_data = np.abs(pickle.load(file))      # to mV
    for i in range(6):
        exp_emg_data[:, i] = exp_emg_data[:, i] - np.mean(exp_emg_data[:, i])
        exp_emg_data[:, i] = exp_emg_data[:, i] / np.std(exp_emg_data[:, i])
    exp_data1 = extract_features(exp_emg_data, int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode=feat_type)

    # labels
    mov = pd.read_csv(os.path.join(data_pth1, "trial{}".format(test_trial), "{}.csv".format("mov")), sep="\t")
    tgt = mov.loc[:, "flexion"]
    tgt1 = signal.resample(tgt, time_samples)
    labels = sep_windows(tgt1[:, None], int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode="mean")
    train_label1 = labels
    tgt2 = signal.resample(tgt, exp_emg_data.shape[0])
    exp_label1 = sep_windows(tgt2[:, None], int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode="mean")

    # DoF2: MCP
    data_pth2 = os.path.join(data_pth, "mcp")
    cur_emg = np.zeros(1)
    for ms in ms_labels:
        if repeat >= 0:
            new_data_pth = os.path.join(data_pth2, "trial{}".format(test_trial), f"repeat{repeat}", "ms_{}.pkl".format(ms))
        else:
            new_data_pth = os.path.join(data_pth2, "test_trial{}".format(test_trial), "ms_{}.pkl".format(ms))
        with open(new_data_pth, "rb") as file:
            emg_data = pickle.load(file)["emg"]
            emg_data = np.transpose(emg_data[eles[:, 0], eles[:, 1]], (1, 0))
        cur_emg = cur_emg + emg_data

    # Normalise
    for i in range(len(ms_labels_ele)):
        cur_emg[:, i] = cur_emg[:, i] - np.mean(cur_emg[:, i])
        cur_emg[:, i] = cur_emg[:, i] / np.std(cur_emg[:, i])
    time_samples = cur_emg.shape[0]

    # Extract features
    train_data2 = extract_features(cur_emg, int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode=feat_type)

    # exp data
    with open(os.path.join(data_pth2, "trial{}".format(test_trial), "few_channels_raw.pkl"), "rb") as file:
        exp_emg_data = np.abs(pickle.load(file))      # to mV
    for i in range(6):
        exp_emg_data[:, i] = exp_emg_data[:, i] - np.mean(exp_emg_data[:, i])
        exp_emg_data[:, i] = exp_emg_data[:, i] / np.std(exp_emg_data[:, i])
    exp_data2 = extract_features(exp_emg_data, int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode=feat_type)

    # labels
    mov = pd.read_csv(os.path.join(data_pth2, "trial{}".format(test_trial), "{}.csv".format("mov")), sep="\t")
    tgt = mov.loc[:, "2mcp_flexion"]
    tgt1 = signal.resample(tgt, time_samples)
    labels = sep_windows(tgt1[:, None], int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode="mean")
    train_label2 = labels
    tgt2 = signal.resample(tgt, exp_emg_data.shape[0])
    exp_label2 = sep_windows(tgt2[:, None], int(args.win_len * args.fs / 1000), int(args.overlap * args.fs / 1000), mode="mean")

    # Train regressors
    # DoF 1
    clf1 = Ridge(alpha=1.0)
    clf1.fit(train_data1, train_label1[:, 0])

    # DoF 2
    clf2 = Ridge(alpha=1.0)
    clf2.fit(train_data2, train_label2[:, 0])

    # Test on experimental data
    # DoF1
    pred_clf1 = clf1.predict(exp_data1)
    # DoF2
    pred_clf2 = clf2.predict(exp_data2)

    if args.low_pass:
        pred_clf1 = low_pass_filter(pred_clf1)
        pred_clf2 = low_pass_filter(pred_clf2)

    # Correlation coefficient
    r_clf1 = stats.pearsonr(exp_label1[:, 0], pred_clf1)[0]
    r_clf2 = stats.pearsonr(exp_label2[:, 0], pred_clf2)[0]

    # shuffle the ground truth labels and compare with the predictions for ten times
    r_clf1_shuffled = []
    r_clf2_shuffled = []
    for i in range(10):
        rand_lbl1 = exp_label1[:, 0].copy()
        np.random.shuffle(rand_lbl1)
        tmp1 = stats.pearsonr(rand_lbl1, pred_clf1)[0]
        r_clf1_shuffled.append(tmp1)
        rand_lbl2 = exp_label2[:, 0].copy()
        np.random.shuffle(rand_lbl2)
        tmp3 = stats.pearsonr(rand_lbl2, pred_clf2)[0]
        r_clf2_shuffled.append(tmp3)

    print("shuffled labels:")
    print("test 1DoF flex-ext R: ridge {:.2f}".format(np.mean(r_clf1_shuffled) * 100))
    print("test 1DoF mcp R: ridge {:.2f}".format(np.mean(r_clf2_shuffled) * 100))

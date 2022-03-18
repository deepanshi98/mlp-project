import numpy as np
import pandas as pd
from min2net.utils import resampling
from min2net.preprocessing.config import CONSTANT

CONSTANT = CONSTANT["MBDEPV1"]
n_chs = CONSTANT["n_chs"]
n_trials = CONSTANT["n_trials"]
window_len = CONSTANT["trial_len"] * CONSTANT["orig_smp_freq"]
orig_chs = CONSTANT["orig_chs"]
trial_len = CONSTANT["trial_len"]
orig_smp_freq = CONSTANT["orig_smp_freq"]
n_trials_per_class = CONSTANT["n_trials_per_class"]


def read_raw(PATH, subject, training, num_class, id_chosen_chs):
    if training:
        print("reading data..")
        df = pd.read_csv(PATH+"/MindBigData-EP-v1.0.zip",  header=None, sep='\n')
    else:
        df = pd.read_csv(PATH+"/MindBigData-EP-v1.0.zip",  header=None, sep='\n')
    
    step = 10_000
    len256 = pd.DataFrame()
    for i in range(0,len(df), step):
        part = df[0][i:i+step].str.split(',', expand=True)
        details = part[0].str.split('\t', expand=True)
        part[0] = details[6]
        part["class_label"] = details[4].astype(int)
        part["length"] = details[5].astype(int)
        part["channel"] = details[3]
        len256 = len256.append(part[part["length"]==256], ignore_index=True)
    correct_len256 = len256.dropna(axis=1, how="all")
    valid_data = correct_len256[correct_len256["class_label"]!=-1]
    n_trials_per_class = 4000
    trial_data = pd.DataFrame()
    unique_classes = list(valid_data["class_label"].unique())
    for i in unique_classes:
        unique_data = pd.DataFrame()
        sampled_df = pd.DataFrame()
        unique_data = valid_data[valid_data["class_label"] == i]
        rows = np.random.choice(unique_data.index.values, n_trials_per_class)
        sampled_df = valid_data.loc[rows]
        trial_data = trial_data.append(sampled_df)
    dataset = valid_data[range(256)].to_numpy().astype(np.float64)
    labels = valid_data["class_label"].to_numpy()
    return dataset, labels


def chanel_selection(sel_chs):
    chs_id = []
    for name_ch in sel_chs:
        ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
        chs_id.append(ch_id)
        print("chosen_channel:", name_ch, "---", "Index_is:", ch_id)
    return chs_id


def load_crop_data(PATH, subject, start, stop, new_smp_freq, num_class, id_chosen_chs):
    start_time = int(start * new_smp_freq)
    stop_time = int(stop * new_smp_freq)
    print("Reading raw training data")
    EEG_train, y_tr = read_raw(
        PATH=PATH,
        subject=subject,
        training=True,
        num_class=num_class,
        id_chosen_chs=id_chosen_chs,
    )
    print(len(EEG_train))
    EEG_test, y_te = read_raw(
        PATH=PATH,
        subject=subject,
        training=False,
        num_class=num_class,
        id_chosen_chs=id_chosen_chs,
    )
    if new_smp_freq < orig_smp_freq:
        EEG_train = resampling(EEG_train, new_smp_freq, trial_len)
        EEG_test = resampling(EEG_test, new_smp_freq, trial_len)
    EEG_train = EEG_train[:, :, start_time:stop_time]
    EEG_test = EEG_test[:, :, start_time:stop_time]
    print(
        "Verify EEG dimension training {} and testing {}".format(
            EEG_train.shape, EEG_test.shape
        )
    )
    return EEG_train, y_tr, EEG_test, y_te

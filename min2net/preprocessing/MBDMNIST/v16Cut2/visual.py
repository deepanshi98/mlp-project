import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split 
import os
from min2net.utils import butter_bandpass_filter
from min2net.preprocessing.MBDMNIST.v16Cut2 import raw
from min2net.preprocessing.config import CONSTANT
CONSTANT = CONSTANT['MBDMNIST16Cut2']
raw_path = CONSTANT['raw_path']
n_subjs = CONSTANT['n_subjs']
n_trials = CONSTANT['n_trials']
n_chs = CONSTANT['n_chs']
orig_smp_freq = CONSTANT['orig_smp_freq']
MI_len = CONSTANT['MI']['len']

def subject_dependent_setting(k_folds, pick_smp_freq, bands, order, save_path, num_class=2, sel_chs=None):
    sel_chs = CONSTANT['sel_chs'] if sel_chs == None else sel_chs
    n_folds = k_folds
    save_path = save_path + '/MBDMNIST16Cut2/visual/{}_class/subject_dependent'.format(num_class)
    n_chs = len(sel_chs)


    id_chosen_chs = raw.chanel_selection(sel_chs)
    EEG_train, MNIST_train, y_train, EEG_test, MNIST_test, y_test = __load_MBDMNIST16Cut2(raw_path, 1, pick_smp_freq, num_class, id_chosen_chs)

    if  os.path.exists(save_path):
        print(f'save_path already exists: {save_path}\nSkipping subject_dependent_setting')
        return
    else:
        os.makedirs(save_path)

    print(f"Processing and saving data to {save_path}")
    # Carry out subject-dependent setting with 5-fold cross validation
    if len(EEG_train.shape) != 3:
        raise Exception('Dimension Error, must have 3 dimension')
    if len(MNIST_train.shape) != 3:
        raise Exception('Dimension Error, must have 3 dimension')

    if n_folds == 1:
        EEG_tr_cv, EEG_val_cv, y_tr, y_val = train_test_split(EEG_train, y_train, test_size = 0.2, random_state= 42, shuffle=True)
        MNIST_tr, MNIST_val, y_tr, y_val = train_test_split(MNIST_train, y_train, test_size = 0.2, random_state= 42, shuffle=True)
        print('Band-pass filtering from {} to {} Hz.'.format(bands[0],  bands[1]))
        EEG_tr_fil = butter_bandpass_filter(EEG_tr_cv,  bands[0],  bands[1], pick_smp_freq, order)
        EEG_val_fil = butter_bandpass_filter(EEG_val_cv,  bands[0],  bands[1], pick_smp_freq, order)
        EEG_te_fil = butter_bandpass_filter(EEG_test,  bands[0],  bands[1], pick_smp_freq, order)
        print('Check dimension of training data {}, val data {} and testing data {}'.format(EEG_tr_fil.shape, EEG_val_fil.shape, EEG_te_fil.shape))
        MNIST_tr_norm = MNIST_tr/255.
        MNIST_val_norm = MNIST_val/255.
        MNIST_test_norm = MNIST_test/255.
        SAVE_NAME = 'S001_fold001'
        __save_data_with_valset(save_path, SAVE_NAME, 
                                EEG_tr_fil,  MNIST_tr_norm,  y_tr,
                                EEG_val_fil, MNIST_val_norm, y_val, 
                                EEG_te_fil,  MNIST_test_norm, y_test)
        print('The preprocessing is DONE')
    else:
        raise NotImplementedError('Fold>1 not implemented yet for this dataset')
        skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
        for fold, (train_index, val_index) in enumerate(skf.split(X_tr , y_tr)):
            print('FOLD:', fold+1, 'TRAIN:', len(train_index), 'VALIDATION:', len(val_index))
            X_tr_cv, X_val_cv = X_tr[train_index], X_tr[val_index]
            y_tr_cv, y_val_cv = y_tr[train_index], y_tr[val_index]

            print('Band-pass filtering from {} to {} Hz.'.format(bands[0],  bands[1]))
            X_tr_fil = butter_bandpass_filter(X_tr_cv,  bands[0],  bands[1], pick_smp_freq, order)
            X_val_fil = butter_bandpass_filter(X_val_cv,  bands[0],  bands[1], pick_smp_freq, order)
            X_te_fil = butter_bandpass_filter(X_te,  bands[0],  bands[1], pick_smp_freq, order)
            print('Check dimension of training data {}, val data {} and testing data {}'.format(X_tr_fil.shape, X_val_fil.shape, X_te_fil.shape))
            SAVE_NAME = 'S{:03d}_fold{:03d}'.format(person+1, fold+1)
            __save_data_with_valset(save_path, SAVE_NAME, X_tr_fil, y_tr_cv, X_val_fil, y_val_cv, X_te_fil, y_te)
            print('The preprocessing of subject {} from fold {} is DONE!!!'.format(person+1, fold+1))


def __load_MBDMNIST16Cut2(PATH, subject, new_smp_freq, num_class, id_chosen_chs):
    start = CONSTANT['MI']['start'] # 0
    stop = CONSTANT['MI']['stop'] # 2
    EEG_train, MNIST_train, y_tr, EEG_test, MNIST_test, y_te = raw.load_crop_data(
        PATH=PATH, subject=subject, start=start, stop=stop, new_smp_freq=new_smp_freq, num_class=num_class, id_chosen_chs=id_chosen_chs)
    return EEG_train, MNIST_train, y_tr, EEG_test, MNIST_test, y_te

def __save_data_with_valset(save_path, NAME, 
                            EEG_train,  MNIST_train,  y_train,
                            EEG_val, MNIST_val, y_val, 
                            EEG_test,  MNIST_test, y_test):
    np.save(save_path+'/EEG_train_'+NAME+'.npy', EEG_train)
    np.save(save_path+'/MNIST_train_'+NAME+'.npy', MNIST_train)
    np.save(save_path+'/EEG_val_'+NAME+'.npy', EEG_val)
    np.save(save_path+'/MNIST_val_'+NAME+'.npy', MNIST_val)
    np.save(save_path+'/EEG_test_'+NAME+'.npy', EEG_test)
    np.save(save_path+'/MNIST_test_'+NAME+'.npy', MNIST_test)
    np.save(save_path+'/y_train_'+NAME+'.npy', y_train)
    np.save(save_path+'/y_val_'+NAME+'.npy', y_val)
    np.save(save_path+'/y_test_'+NAME+'.npy', y_test)
    print('save DONE')

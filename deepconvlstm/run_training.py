# coding: utf-8
import time
import re
import logging
from glob import glob
import pickle
from pathlib import Path
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
import nina_helper as nh
import deepconvlstm as dcl

K.set_image_data_format('channels_last')
K.set_learning_phase(1)
np.random.seed(1)
BATCH_SIZE = 16
LEARN_RATE = 0.001
# Size, in seconds, of an entire movement (5s movement, + 3s rest)
ENTIRE_MOV_DURATION = 8
# Size, em ms, of the time window where each classification will take place
OBS_WINDOW = 200
DATASETS_DICT = {
    "dataset_1": {
        "dataset_path": "../datasets/db1/",
        "weights_path": "../weights/db1_batchsize"+str(BATCH_SIZE)+"/",
        "log_dir": "../logs/db1_batchsize"+str(BATCH_SIZE)+"/",
        "import_func": nh.import_db1,
        "dataset_info": nh.db1_info(),
        "history_path": "../history/db1/",
        "window_factor": 0.1,
        "train_split": 3
    },
    "dataset_2": {
        "dataset_path": "../datasets/db2/",
        "weights_path": "../weights/db2_batchsize"+str(BATCH_SIZE)+"/",
        "log_dir": "../logs/db2_batchsize"+str(BATCH_SIZE)+"/",
        "import_func": nh.import_db2,
        "dataset_info": nh.db2_info(),
        "history_path": "../history/db2_batchsize"+str(BATCH_SIZE)+"/",
        "window_factor": 2,
        "train_split": 2
    },
    "dataset_3": {
        "dataset_path": "../datasets/db3/",
        "weights_path": "../weights/db3_batchsize"+str(BATCH_SIZE)+"/",
        "log_dir": "../logs/db3_batchsize"+str(BATCH_SIZE)+"/",
        "import_func": nh.import_db3,
        "dataset_info": nh.db3_info(),
        "history_path": "../history/db3/",
        "window_factor": 2,
        "train_split": 2
    },
}
WEIGHTS_PATTERN = "epoch:{epoch:02d}-acc:{acc:.4f}-val_acc:{val_acc:.4f}.hdf5"
logger = logging.getLogger("deepconvlstm_batchsize"+str(BATCH_SIZE))
hdlr = logging.FileHandler("deepconvlstm_batchsize"+str(BATCH_SIZE)+".log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def best_weight(folder, metric, filehead):
    """Return best weight based on metric.

    Arguments:
        folder (:obj:`str`): path of weight's folder
        metric (:obj:`str`): metric to use
            as comparision
        filehead (:obj:`str`): string to identify the type of data
            used to train the model

    Returns:
        best_weight (:obj:`str`): best weight path
        epoch (:obj:`int`): epoch of best weight

    """
    best_metric = 0
    weights_list = glob(folder+'/weights--{}--*.hdf5'.format(filehead))
    if not weights_list:
        return (False, None)
    b_weight = weights_list[0]
    reg = re.compile('-'+metric+':(\d.\d{4})')
    try:
        for filename in weights_list:
            res = reg.search(filename)
            if res:
                file_metric = (float(res.groups()[0]))
                if file_metric > best_metric:
                    best_metric = file_metric
                    b_weight = filename
        epoch = int(re.search('epoch:(\d+)-', b_weight).groups()[0])
    except AttributeError:
        return (False, None)
    return (b_weight, epoch)


def load_pretrained(model, folder, metric, subject):
    """Return a model with the best pretrained weights from a folder, if it exists.

    Arguments:
        model (:obj:`kerasModel`): model with basic structure
        folder (:obj:`str`): path of weight's folder
        metric (:obj:`str`, optional, *default*=acc): metric to use
            as comparision
        subject (:obj:`int`): 0 for gathering data from all subjects,
            or any other integer in the subjects range for a single subject

    Returns:
        model (:obj:`kerasModel`): a keras model loaded with best weights file,
            or None if no file could be found

    """
    if subject == 0:
        filehead = 'all'
    else:
        filehead = 'subject:{}'.format(subject)
    weight_file, epoch = best_weight(folder, metric, filehead)
    if not weight_file:
        return (None, None)
    model.load_weights(weight_file)
    return (model, epoch)


def get_deepconvlstm(input_shape, subject, class_number, dataset, monitor='val_acc'):
    """Return a deepconvlsm model with the best pretrained weights from a folder if it exists.

    Arguments:
        input_shape (:obj:`tuple`): shape of the input dataset:
            (num_timesteps, num_channels)
        subject (:obj:`int`): 0 for gathering data from all subjects,
            or any other integer in the subjects range for a single subject
        class_number (:obj:`int`,optional, *default* =53):
            Number of classes for classification task
        dataset (:obj:`str`): dataset number: dataset_1, dataset_2 or dataset_3
        monitor (:obj:`str`, optional, *default*=acc): metric to monitor in callback

    Returns:
        model (:obj:`kerasModel`): a keras model loaded with best weights file,
            or None if no file could be found

    """
    subn = 'subject:{}'.format(subject) if subject else 'all'
    file_weights = DATASETS_DICT[dataset]['weights_path']+'/weights--{}--'.format(subn) +\
        WEIGHTS_PATTERN
    model = dcl.model_deepconvlstm(input_shape, class_number=class_number,
                                   learn_rate=LEARN_RATE)
    checkpoint = ModelCheckpoint(file_weights, verbose=1, monitor=monitor,
                                 save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir=DATASETS_DICT[dataset]['log_dir']+"{}".format(
        time.strftime("%d/%m/%Y--%H:%M:%S")),
                              write_images=True)
    early_stopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                                  patience=5, min_lr=0.001)
    callbacks_list = [checkpoint, tensorboard, early_stopping, reduce_lr]
    return (model, callbacks_list)


def prepare_data(subject, timesteps_number, train_split, dataset,
                 subsamp_rate=1, select_classes=None):
    """Get data from dataset and assemble as 4D input tensor for a keras model.

    Arguments:
        subject (:obj:`int`): 0 for gathering data from all subjects,
            or any other integer in the subjects range for a single subject
        timesteps_number (:obj:`int`): number of time steps for each sequence
        train_split (:obj:`int`): 1-10, proportion of test-train split, based on
            number of repetitions (10) Ex: 3 indicates 30% of test samples
        dataset (:obj:`str`): dataset number: dataset_1, dataset_2 or dataset_3
        subsamp_rate (:obj:`int`, *default* =1): subsample rate. Ex: 2 indicates
            that all dataset will be subsampled to a half
        select_classes (:obj:`list`,optional, *default* =None): If given, the classes
            used for the classifier will be only these ones; else, all classes

    Returns:
        x_train (:obj:`4d-array`): a 4-D array to be used as input train samples
        y_train (:obj:`4d-array`): a 4-D array to be used as categorical output train samples
        x_test (:obj:`4d-array`): a 4-D array to be used as input test samples
        y_test (:obj:`4d-array`): a 4-D array to be used as categorical output test samples

    """
    inc_len = int((OBS_WINDOW*DATASETS_DICT[dataset]["window_factor"])/subsamp_rate)
    init_fs = int((DATASETS_DICT[dataset]["dataset_info"]["fs"]*(OBS_WINDOW/1000))/subsamp_rate)
    reps = DATASETS_DICT[dataset]["dataset_info"]['rep_labels']
    # Get EMG, repetition and movement data, don't cap maximum length of rest
    subject_dict = DATASETS_DICT[dataset]["import_func"](DATASETS_DICT[dataset]["dataset_path"],
                                                         subject)
    # Create a balanced test - training split based on repetition number
    train_reps, test_reps = nh.gen_split_balanced(reps, train_split)
    # Normalise EMG data based on training set
    emg_data = nh.normalise_emg(subject_dict['emg'][::subsamp_rate],
                                subject_dict['rep'][::subsamp_rate],
                                train_reps[0, :])
    # Window data: x_all data is 4D tensor [observation, time_step, channel, 1]
    # for use with Keras
    # y_all: movement label, length: number of windows
    # r_all: repetition label, length: number of windows
    x_all, y_all, r_all = nh.get_windows(reps, timesteps_number, inc_len,
                                         emg_data, subject_dict['move'][::subsamp_rate],
                                         subject_dict['rep'][::subsamp_rate],
                                         which_moves=select_classes,
                                         dtype=np.float16)
    x_sq = np.squeeze(x_all)
    win_1 = x_sq[0, :, 1]
    win_2 = x_sq[1, :, 1]
    win_3 = x_sq[2, :, 1]
    # Checking consistency of 3 first windows...
    first_check = win_1[2*init_fs-1] == win_2[init_fs-1]
    sec_check = win_2[init_fs] == win_3[0]
    if not (first_check and sec_check):
        expected_samples = subject_dict['emg'][::subsamp_rate].shape[0]/(inc_len)
        print ('Expected samples: {}'.format(expected_samples))
        print ('Proportion expected/got: {}'.format(expected_samples/x_all.shape[0]))
        raise Exception("Windows are not coherent!")
    # Window data: x_all data is 4D tensor [observation, time_step, channel, 1]
    train_idx = nh.get_idxs(r_all, train_reps[0, :])
    test_idx = nh.get_idxs(r_all, test_reps[0, :])
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    # Preparing data for one hot categorical
    if select_classes:
        for idx, val in enumerate(select_classes):
            y_train[y_train == val] = idx
            y_test[y_test == val] = idx
    y_train = nh.to_categorical(y_train)
    x_train = x_all[train_idx, :, :, :]
    y_test = nh.to_categorical(y_test)
    x_test = x_all[test_idx, :, :, :]
    return (x_train, y_train, x_test, y_test)


def run_training(dataset, timeback_reach, epochs, subsamp_rate, moves=None):
    """Train the deepconvlstm model, given a dataset.

    Arguments:
        dataset (:obj:`str`): dataset number: dataset_1, dataset_2 or dataset_3
        timeback_reach (:obj:`int`): time in seconds on how far back the lstm
            cells will look into the past
        timesteps_number (:obj:`int`): number of time steps for each sequence
        epochs (:obj:`int`): number of epochs for training
        subsamp_rate (:obj:`int`, *default* =1): subsample rate. Ex: 2 indicates
            that all dataset will be subsampled to a half
        moves (:obj:`list`,optional, *default* =None): If given, the classes
            used for the classifier will be only these ones; else, all classes

    """
    ds_dict = DATASETS_DICT[dataset]
    Path(ds_dict['weights_path']).mkdir(parents=True, exist_ok=True)
    Path(ds_dict['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(ds_dict['history_path']).mkdir(parents=True, exist_ok=True)
    timestep_num = int((ds_dict["dataset_info"]["fs"]*timeback_reach)/subsamp_rate)
    train_split = ds_dict["train_split"]
    try:
        classes = ds_dict["dataset_info"]["nb_moves"] if not moves else len(moves)
    except KeyError:
        classes = None
    w_folder = ds_dict["weights_path"]
    logger.info('Starting training process...')
    logger.info('Epochs:{}, timesteps_number:{}, step_len:{} ms, batch size:{} samples'.
                format(epochs, timestep_num, OBS_WINDOW, BATCH_SIZE))
    for subject_number in range(10, ds_dict["dataset_info"]["nb_subjects"]+1):
        if not classes:
            classes = ds_dict["dataset_info"]['subjects'][subject_number-1]["nb_moves"]
        print ('Subject {}, number of classes: {}'.format(subject_number, classes))
        logger.info('Running training for subject {}...'.format(subject_number))
        sub_data = prepare_data(subject_number, timestep_num, train_split,
                                dataset, subsamp_rate, moves)
        input_shape = sub_data[0].shape
        model, callbacks_list = get_deepconvlstm(input_shape[1:], subject_number,
                                                 classes, dataset)
        res = load_pretrained(model, w_folder, 'val_acc', subject_number)
        if res[0]:
            initial_epoch = res[1]
            logger.debug('Using pre-trained weights... resuming from epoch {}'.
                         format(initial_epoch))
            model = res[0]
        else:
            initial_epoch = 0
        model.summary()
        hist = model.fit(sub_data[0], sub_data[1], epochs=epochs, batch_size=BATCH_SIZE,
                         validation_split=0.33, callbacks=callbacks_list, verbose=1,
                         initial_epoch=initial_epoch)
        wfile, epoch = best_weight(w_folder, 'val_acc', 'subject:{}'.
                                   format(subject_number))
        logger.debug('Best results from epoch {}, saved in file {}'.
                     format(epoch, wfile))
        logger.debug('Saving history in a picke file...')
        filehistname = ds_dict["history_path"] +\
            "/subject:{}_history.pickle".format(subject_number)
        with open(filehistname, 'wb') as fname:
            pickle.dump(hist.history, fname)
        preds_train = model.evaluate(sub_data[0], sub_data[1])
        logger.info("Train Accuracy = " + str(preds_train[1]))
        preds_test = model.evaluate(sub_data[2], sub_data[3])
        logger.info("Test Accuracy = " + str(preds_test[1]))


if __name__ == "__main__":
    epochs = 150
    # Rate of subsampling (reduces memory and cpu usage but accuracy too)
    subsamp_rate = 16
    reach_number = 1.25
    timeback_reach = ENTIRE_MOV_DURATION*reach_number
    run_training('dataset_3', timeback_reach, epochs, subsamp_rate)
    for dataset in DATASETS_DICT.keys():
        if dataset == 'dataset_1':
            subsamp_rate = 1
        run_training(dataset, timeback_reach, epochs, subsamp_rate)

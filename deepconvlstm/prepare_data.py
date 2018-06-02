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
BATCH_SIZE = 40
LEARN_RATE = 0.001
DATASETS_DICT = {
    "dataset_1": {
        "dataset_path": "../datasets/db1/",
        "weights_path": "../weights/db1_batchsize"+str(BATCH_SIZE)+"/",
        "log_dir": "../logs/db1_batchsize"+str(BATCH_SIZE)+"/",
        "import_func": nh.import_db1,
        "dataset_info": nh.db1_info(),
        "history_path": "../history/db1/",
        "ts_number": 10,
        "train_split": 3
    },
    "dataset_2": {
        "dataset_path": "../datasets/db2/",
        "weights_path": "../weights/db2_batchsize"+str(BATCH_SIZE)+"/",
        "log_dir": "../logs/db2_batchsize"+str(BATCH_SIZE)+"/",
        "import_func": nh.import_db2,
        "dataset_info": nh.db2_info(),
        "history_path": "../history/db2_batchsize"+str(BATCH_SIZE)+"/",
        "ts_number": 400,
        "train_split": 2
    },
    "dataset_3": {
        "dataset_path": "../datasets/db3/",
        "weights_path": "../weights/db3_batchsize"+str(BATCH_SIZE)+"/",
        "log_dir": "../logs/db3_batchsize"+str(BATCH_SIZE)+"/",
        "import_func": nh.import_db3,
        "dataset_info": nh.db3_info(),
        "history_path": "../history/db3/",
        "ts_number": 400,
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


def prepare_data(subject, timesteps_number, inc_len, train_split, dataset, select_classes=None):
    """Get data from dataset and assemble as 4D input tensor for a keras model.

    Arguments:
        subject (:obj:`int`): 0 for gathering data from all subjects,
            or any other integer in the subjects range for a single subject
        timesteps_number (:obj:`int`): number of time steps for each sequence
        inc_len (:obj:`int`): size in ms of the moving window increment
            (only multiples of 10) Ex: 10, 20, 30, ...
        train_split (:obj:`int`): 1-10, proportion of test-train split, based on
            number of repetitions (10) Ex: 3 indicates 30% of test samples
        dataset (:obj:`str`): dataset number: dataset_1, dataset_2 or dataset_3
        select_classes (:obj:`list`,optional, *default* =None): If given, the classes
            used for the classifier will be only these ones; else, all classes

    Returns:
        x_train (:obj:`4d-array`): a 4-D array to be used as input train samples
        y_train (:obj:`4d-array`): a 4-D array to be used as categorical output train samples
        x_test (:obj:`4d-array`): a 4-D array to be used as input test samples
        y_test (:obj:`4d-array`): a 4-D array to be used as categorical output test samples

    """
    reps = DATASETS_DICT[dataset]["dataset_info"]['rep_labels']
    # Get EMG, repetition and movement data, don't cap maximum length of rest
    subject_dict = DATASETS_DICT[dataset]["import_func"](DATASETS_DICT[dataset]["dataset_path"],
                                                         subject)
    # Create a balanced test - training split based on repetition number
    train_reps, test_reps = nh.gen_split_balanced(reps, train_split)
    # Normalise EMG data based on training set
    emg_data = nh.normalise_emg(subject_dict['emg'], subject_dict['rep'],
                                train_reps[0, :])
    # Window data: x_all data is 4D tensor [observation, time_step, channel, 1]
    # for use with Keras
    # y_all: movement label, length: number of windows
    # r_all: repetition label, length: number of windows
    x_all, y_all, r_all = nh.get_windows(reps, timesteps_number, int(inc_len/10),
                                         emg_data, subject_dict['move'],
                                         subject_dict['rep'],
                                         which_moves=select_classes,
                                         dtype=np.float16)
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


def run_trainning(dataset, inc_len, epochs, moves=None):
    ds_dict = DATASETS_DICT[dataset]
    Path(ds_dict['weights_path']).mkdir(parents=True, exist_ok=True)
    Path(ds_dict['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(ds_dict['history_path']).mkdir(parents=True, exist_ok=True)
    timestep_num = ds_dict["ts_number"]
    train_split = ds_dict["train_split"]
    try:
        classes = ds_dict["dataset_info"]["nb_moves"] if not moves else len(moves)
    except KeyError:
        classes = None
    w_folder = ds_dict["weights_path"]
    logger.info('Starting training process...')
    logger.info('Epochs:{}, timesteps_number:{}, step_len:{} ms, batch size:{} samples'.
                format(epochs, timestep_num, inc_len, BATCH_SIZE))
    for subject_number in range(1, ds_dict["dataset_info"]["nb_subjects"]+1):
        if not classes:
            classes = ds_dict["dataset_info"]['subjects'][subject_number-1]["nb_moves"]
        logger.info('Running training for subject {}...'.format(subject_number))
        sub_data = prepare_data(subject_number, timestep_num, inc_len, train_split,
                                dataset, moves)
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
    inc_len = 100
    epochs = 150
    run_trainning('dataset_1', inc_len, epochs)
    for dataset in DATASETS_DICT.keys():
        run_trainning(dataset, inc_len, epochs)


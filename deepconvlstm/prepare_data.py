# coding: utf-8
import time
import re
from glob import glob
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.backend as K
import nina_helper as nh
import deepconvlstm as dcl

K.set_image_data_format('channels_last')
K.set_learning_phase(1)
np.random.seed(1)
DB1_PATH = "../datasets/db1/"
DB1_INFO = nh.db1_info()
DB1_WEIGHTS_PATH = "../weights/db1/"
WEIGHTS_PATTERN = "epoch:{epoch:02d}-acc:{acc:.2f}-val_acc:{val_acc:.2f}.hdf5"


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
    reg = re.compile('-'+metric+':(\d.\d{2})')
    try:
        for filename in weights_list:
            res = reg.search(filename)
            if res:
                file_metric = (float(res.groups()[0]))
                if file_metric > best_metric:
                    best_metric = file_metric
                    b_weight = filename
        epoch = int(re.search('epoch:(\d+)-', b_weight).groups()[0])
    except:
        return (False, None)
    return (b_weight, epoch)


def load_pretrained(model, folder, metric, subject):
    """Return a model with the best pretrained weights from a folder, if it exists.

    Arguments:
        model (:obj:`kerasModel`): model with basic structure
        folder (:obj:`str`): path of weight's folder
        metric (:obj:`str`, optional, *default*=acc): metric to use
            as comparision
        folder (:obj:`str`): path of weight's folder
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


def get_deepconvlstm(input_shape, subject, class_number, monitor='val_acc'):
    """Return a deepconvlsm model with the best pretrained weights from a folder if it exists.

    Arguments:
        folder (:obj:`str`): path of weight's folder
        metric (:obj:`str`, optional, *default*=acc): metric to use
            as comparision
        class_number (:obj:`int`,optional, *default* =53):
            Number of classes for classification task
        folder (:obj:`str`): path of weight's folder

    Returns:
        model (:obj:`kerasModel`): a keras model loaded with best weights file,
            or None if no file could be found

    """
    subn = 'subject:{}'.format(subject) if subject else 'all'
    file_weights = DB1_WEIGHTS_PATH+'/weights--{}--'.format(subn)+WEIGHTS_PATTERN
    model = dcl.model_deepconvlstm(input_shape, class_number=class_number)
    checkpoint = ModelCheckpoint(file_weights, verbose=1, monitor=monitor,
                                 save_best_only=True, mode='max')
    tensorboard = TensorBoard(log_dir="../logs/{}".format(time.strftime("%d/%m/%Y--%H:%M:%S")),
                              write_images=True)
    callbacks_list = [checkpoint, tensorboard]
    return (model, callbacks_list)


def prepare_data(subject, timesteps_number, inc_len, train_split, select_classes=None):
    """Get data from dataset and assemble as 4D input tensor for a keras model.

    Arguments:
        subject (:obj:`int`): 0 for gathering data from all subjects,
            or any other integer in the subjects range for a single subject
        timesteps_number (:obj:`int`): number of time steps for each sequence
        inc_len (:obj:`int`): size in ms of the moving window increment
            (only multiples of 10) Ex: 10, 20, 30, ...
        train_split (:obj:`int`): 1-10, proportion of test-train split, based on
            number of repetitions (10) Ex: 3 indicates 30% of test samples
        select_classes (:obj:`list`,optional, *default* =None): If given, the classes
            used for the classifier will be only these ones; else, all classes
            number of repetitions (10) Ex: 3 indicates 30% of test samples

    Returns:
        x_train (:obj:`4d-array`): a 4-D array to be used as input train samples
        y_train (:obj:`4d-array`): a 4-D array to be used as categorical output train samples
        x_test (:obj:`4d-array`): a 4-D array to be used as input test samples
        y_test (:obj:`4d-array`): a 4-D array to be used as categorical output test samples

    """
    reps = DB1_INFO['rep_labels']
    # Get EMG, repetition and movement data, don't cap maximum length of rest
    subject_dict = nh.import_db1(DB1_PATH, subject)
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
                                         which_moves=select_classes)
    train_idx = nh.get_idxs(r_all, train_reps[0, :])
    test_idx = nh.get_idxs(r_all, test_reps[0, :])
    y_train = nh.to_categorical(y_all[train_idx])
    x_train = x_all[train_idx, :, :, :]
    y_test = nh.to_categorical(y_all[test_idx])
    x_test = x_all[test_idx, :, :, :]
    return (x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    subject_number = 4
    timestep_num = 20
    inc_len = 100
    epochs = 400
    train_split = 3
    classes = 53
    w_folder = DB1_WEIGHTS_PATH
    sub_data = prepare_data(subject_number, timestep_num, inc_len, train_split)
    input_shape = sub_data[0].shape
    model, callbacks_list = get_deepconvlstm(input_shape[1:], subject_number, classes)
    model.summary()
    res = load_pretrained(model, w_folder, 'val_acc', subject_number)
    if res[0]:
        print ('Using pre-trained weights...')
        model = res[0]
        initial_epoch = res[1]
    else:
        initial_epoch = 0
    model.fit(sub_data[0], sub_data[1], epochs=epochs, batch_size=100,
              validation_split=0.2, callbacks=callbacks_list, verbose=1,
              initial_epoch=initial_epoch)
    preds_train = model.evaluate(sub_data[0], sub_data[1])
    print("Train Loss = " + str(preds_train[0]))
    print("Train Accuracy = " + str(preds_train[1]))
    preds_test = model.evaluate(sub_data[2], sub_data[3])
    print("Test Loss = " + str(preds_test[0]))
    print("Test Accuracy = " + str(preds_test[1]))

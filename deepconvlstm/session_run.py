# coding: utf-8
import time
import re
import os
import logging
from glob import glob
import pickle
from pathlib import Path
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
import nina_helper as nh
import deepconvlstm.model as dcl

K.set_image_data_format("channels_last")
K.set_learning_phase(1)
np.random.seed(1)
LOGGER = logging.getLogger("deepconvlstm")
DATABASES_DICT = {
    "database_1": {
        "database_path": "databases/db1/",
        "weights_path": "results/weights/db1/ts200/",
        "log_dir": "results/logs/db1/",
        "import_func": nh.import_db1,
        "database_info": nh.db1_info(),
        "history_path": "results/history/db1/ts200",
        "window_factor": 0.1,
        "subsamp_rate": 1,
        "train_split": 3
    },
    "database_2": {
        "database_path": "databases/db2/",
        "weights_path": "results/weights/db2/",
        "log_dir": "results/logs/db2/",
        "import_func": nh.import_db2,
        "database_info": nh.db2_info(),
        "history_path": "results/history/db2/",
        "window_factor": 2,
        "subsamp_rate": 16,
        "train_split": 2
    },
    "database_3": {
        "database_path": "databases/db3/",
        "weights_path": "results/weights/db3/",
        "log_dir": "results/logs/db3/",
        "import_func": nh.import_db3,
        "database_info": nh.db3_info(),
        "history_path": "results/history/db3/",
        "window_factor": 2,
        "subsamp_rate": 16,
        "train_split": 2
    },
}


def best_weight(folder, metric, filehead, sig_dig, file_pattern=None):
    """Return best weight based on metric.

    The weights file is supposed to be written similarly to
    weights--subject:1--epoch:11-acc:0.5971-val_acc:0.4687.hdf5

    Arguments:
        folder (:obj:`str`): path of weight's folder
        metric (:obj:`str`): metric to use
            as comparision
        filehead (:obj:`str`): string to identify the type of data
            used to train the model
        sig_dig (:obj:`int`): number of significant digits used when
            writing the file (Ex: 0.23 ->2, 0.3345 ->4)
        file_pattern (:obj:`str`,optional, *default* ='weights--{}--*.hdf5'):
            pattern used for writing the file
    Returns:
        best_weight (:obj:`str`): best weight path
        epoch (:obj:`int`): epoch of best weight

    """
    if not file_pattern:
        file_pattern = "weights--{}--*.hdf5"
    best_metric = 0
    weights_list = glob(folder+"/"+file_pattern.format(filehead))
    if not weights_list:
        return (False, None)
    b_weight = weights_list[0]
    reg = re.compile("-"+metric+":(\d.\d{})".format(sig_dig))
    try:
        for filename in weights_list:
            res = reg.search(filename)
            if res:
                file_metric = (float(res.groups()[0]))
                if file_metric > best_metric:
                    best_metric = file_metric
                    b_weight = filename
        epoch = int(re.search("epoch:(\d+)-", b_weight).groups()[0])
    except AttributeError:
        return (False, None)
    return (b_weight, epoch)


class DeepConvLstm(object):
    def __init__(self, subject, database, **kwargs):
        """Give Deepconvlstm model and all methods for training it.

        Arguments:
            subject (:obj:`int`): 0 for gathering data from all subjects,
                or any other integer in the subjects range for a single subject
            database (:obj:`str`): database number: database_1, database_2 or database_3
            timesteps_number (:obj:`int`, optional): number of time steps
                for each sequence
            train_split (:obj:`int`, optional): 1-10, proportion of test-train split,
                based on number of repetitions (10) Ex: 3 indicates 30% of test samples
            moves (:obj:`list`,optional, *default* =None): If given, the classes
                used for the classifier will be only these ones; else, all classes
            class_number (:obj:`int`,optional): number of classes for classification task
            learn_rate (:obj:`float`, optional, *default* =0.001): model's learning rate
            batch_size (:obj:`int`, optional, *default* =16): training's batch size
            sig_dig (:obj:`int`, optional, *default* =4): number of significant digits
                of metric to use when writing the model trained weights
            monitor (:obj:`str`, optional, *default* =val_acc): metric to monitor in callback
            databases_dict: (:obj:`dict`, optional, *default* =DATABASES_DICT): dictionary
                containing useful information about the databases (see DATABASES_DICT in
                this module for reference)
            timeback_reach (:obj:`float`): how many move durations far back the lstm
                cells will look into the past. Ex: 1.25 means 1.25*8 (move duration) =
                10 seconds back into the past
            epochs (:obj:`int`, optional, *default* =150): number of epochs for training
            base_path (:obj:`int`, optional, *default* =current dir): base path for writing
                results
            obs_window (:obj:`int`, optional, *default* =200): size of the window time,
                in miliseconds, where each classification will take place
            subsamp_rate (:obj:`int`, optional): subsample rate. Ex: 2 indicates
                that all dataset will be subsampled to a half

        """
        def_args = {
            "batch_size": 16,
            "learn_rate": 0.001,
            "databases_dict": DATABASES_DICT,
            "sig_dig": 4,
            "obs_window": 200,
            "timeback_reach": 1.25,
            "monitor": "val_acc",
            "base_path": os.getcwd(),
            "subsamp_rate": DATABASES_DICT[database]["subsamp_rate"],
            "moves": None
            }
        args = def_args.keys()
        def_args.update(kwargs)
        self.__dict__.update((key, val) for key, val in def_args.items() if key in args)
        self.db_dict = self.databases_dict[database]
        if self.moves:
            self.n_classes = len(self.moves)
        else:
            try:
                self.n_classes = self.db_dict["database_info"]["nb_moves"]
            except KeyError:
                self.n_classes = self.db_dict["database_info"]["subjects"]\
                                             [subject-1]["nb_moves"]
        freq = self.db_dict["database_info"]["fs"]
        self.timesteps_number = int((freq*self.timeback_reach)/self.subsamp_rate)
        self.subject = subject
        self.database = database
        # Size, in seconds, of an entire movement (5s movement, + 3s rest)
        self.move_duration = 8
        self.db_dict["database_path"] = self.base_path + '/' + self.db_dict["database_path"]
        self.db_dict["weights_path"] = self.base_path + '/' + self.db_dict["weights_path"]
        self.db_dict["log_dir"] = self.base_path + '/' + self.db_dict["log_dir"]
        self.db_dict["history_path"] = self.base_path + '/' + self.db_dict["history_path"]
        self.weights_pattern = "epoch:{epoch:02d}-acc:{acc:." + str(self.sig_dig) +\
            "f}-val_acc:{val_acc:." + str(self.sig_dig) + "f}.hdf5"

    def prepare_data(self):
        """Get data from dataset and assemble as 4D input tensor for a keras model.

        Returns:
            x_train (:obj:`4d-array`): a 4-D array to be used as input train samples
            y_train (:obj:`4d-array`): a 4-D array to be used as categorical output train samples
            x_test (:obj:`4d-array`): a 4-D array to be used as input test samples
            y_test (:obj:`4d-array`): a 4-D array to be used as categorical output test samples

        """
        win_factor = self.db_dict["window_factor"]
        inc_len = int((self.obs_window*win_factor)/self.subsamp_rate)
        reps = self.db_dict["database_info"]["rep_labels"]
        # Get EMG, repetition and movement data, don't cap maximum length of rest
        subject_dict = self.db_dict["import_func"](self.db_dict["database_path"],
                                                   self.subject)
        # Create a balanced test - training split based on repetition number
        train_reps, test_reps = nh.gen_split_balanced(reps, self.db_dict["train_split"])
        # Normalise EMG data based on training set
        emg_data = nh.normalise_emg(subject_dict["emg"][::self.subsamp_rate],
                                    subject_dict["rep"][::self.subsamp_rate],
                                    train_reps[0, :])
        # Window data: x_all data is 4D tensor [observation, time_step, channel, 1]
        # for use with Keras
        # y_all: movement label, length: number of windows
        # r_all: repetition label, length: number of windows
        x_all, y_all, r_all = nh.get_windows(reps, self.timesteps_number, inc_len,
                                             emg_data, subject_dict["move"][::self.subsamp_rate],
                                             subject_dict["rep"][::self.subsamp_rate],
                                             which_moves=self.moves,
                                             dtype=np.float16)
        # Window data: x_all data is 4D tensor [observation, time_step, channel, 1]
        train_idx = nh.get_idxs(r_all, train_reps[0, :])
        test_idx = nh.get_idxs(r_all, test_reps[0, :])
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        # Preparing data for one hot categorical
        if self.moves:
            for idx, val in enumerate(self.moves):
                y_train[y_train == val] = idx
                y_test[y_test == val] = idx
        y_train = nh.to_categorical(y_train)
        x_train = x_all[train_idx, :, :, :]
        y_test = nh.to_categorical(y_test)
        x_test = x_all[test_idx, :, :, :]
        return (x_train, y_train, x_test, y_test)

    def load_pretrained(self, model, metric="val_acc"):
        """Return a model with the best pretrained weights from a folder, if it exists.

        Arguments:
            model (:obj:`kerasModel`): model with basic structure
            metric (:obj:`str`, optional, *default*=val_acc): metric to use
                as comparision

        Returns:
            model (:obj:`kerasModel`): a keras model loaded with best weights file,
                or None if no file could be found

        """
        if not self.subject:
            filehead = "all"
        else:
            filehead = "subject:{}".format(self.subject)
        weight_file, epoch = best_weight(self.db_dict["weights_path"], metric,
                                         filehead, self.sig_dig)
        if not weight_file:
            return (None, None)
        model.load_weights(weight_file)
        return (model, epoch)

    def get_model(self, input_shape):
        """Return a deepconvlsm model with the best pretrained weights from a folder if it exists.

        Arguments:
            input_shape (:obj:`tuple`): shape of the input dataset:
                (num_timesteps, num_channels)
        Returns:
            model (:obj:`kerasModel`): a keras model loaded with best weights file,
                or None if no file could be found

        """
        subn = "subject:{}".format(self.subject) if self.subject else "all"
        file_weights = self.db_dict["weights_path"]+"/weights--{}--".format(subn) +\
            self.weights_pattern
        model = dcl.model_deepconvlstm(input_shape, class_number=self.n_classes,
                                       learn_rate=self.learn_rate)
        checkpoint = ModelCheckpoint(file_weights, verbose=1, monitor=self.monitor,
                                     save_best_only=True, mode="max")
        tensorboard = TensorBoard(log_dir=self.db_dict["log_dir"]+"{}".format(
                                  time.strftime("%d/%m/%Y--%H:%M:%S")),
                                  write_images=True)
        early_stopping = EarlyStopping(monitor=self.monitor, patience=20, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor=self.monitor, factor=0.2,
                                      patience=5, min_lr=0.001)
        callbacks_list = [checkpoint, tensorboard, early_stopping, reduce_lr]
        return (model, callbacks_list)

    def run_training(self):
        """Train the deepconvlstm model, given a dataset.

        """
        Path(self.db_dict["weights_path"]).mkdir(parents=True, exist_ok=True)
        Path(self.db_dict["log_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.db_dict["history_path"]).mkdir(parents=True, exist_ok=True)
        w_folder = self.db_dict["weights_path"]
        LOGGER.info("Starting training process...")
        LOGGER.info("Epochs:{}, timesteps_number:{}, step_len:{} ms, batch size:{} samples".
                    format(self.epochs, self.timestep_num, self.obs_window, self.batch_size))
        LOGGER.info("Subject {}, number of classes: {}".format(self.subject, self.n_classes))
        LOGGER.info("Running training for subject {}...".format(self.subject))
        sub_data = self.prepare_data()
        input_shape = sub_data[0].shape
        model, callbacks_list = self.get_model(input_shape[1:])
        res = self.load_pretrained(model, "val_acc")
        if res[0]:
            initial_epoch = res[1]
            LOGGER.debug("Using pre-trained weights... resuming from epoch {}".
                         format(initial_epoch))
            model = res[0]
        else:
            initial_epoch = 0
        model.summary()
        hist = model.fit(sub_data[0], sub_data[1], epochs=self.epochs,
                         batch_size=self.batch_size, validation_split=0.33,
                         callbacks=callbacks_list, verbose=1,
                         initial_epoch=initial_epoch)
        wfile, epoch = best_weight(w_folder, "val_acc", "subject:{}".
                                   format(self.subject), self.sig_dig)
        LOGGER.debug("Best results from epoch {}, saved in file {}".
                     format(epoch, wfile))
        LOGGER.debug("Saving history in a picke file...")
        filehistname = self.db_dict["history_path"] +\
            "/subject:{}_history.pickle".format(self.subject)
        with open(filehistname, "wb") as fname:
            pickle.dump(hist.history, fname)
        preds_train = model.evaluate(sub_data[0], sub_data[1])
        LOGGER.info("Train Accuracy = " + str(preds_train[1]))
        preds_test = model.evaluate(sub_data[2], sub_data[3])
        LOGGER.info("Test Accuracy = " + str(preds_test[1]))


def run_dbtraining(database):
    """Train the deepconvlstm model, given a dataset.

    Arguments:
        database (:obj:`str`): database number: database_1, database_2 or database_3

    """
    db_dict = DATABASES_DICT[database]
    LOGGER.info("Starting training process for entire database {}...".format(database))
    for subject_number in range(1, db_dict["database_info"]["nb_subjects"]+1):
        deepconv = DeepConvLstm(subject_number, database)
        deepconv.prepare_data()
        deepconv.run_training()


if __name__ == "__main__":
    HDLR = logging.FileHandler("deepconvlstm.log")
    FORMATTER = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    HDLR.setFormatter(FORMATTER)
    LOGGER.addHandler(HDLR)
    LOGGER.setLevel(logging.DEBUG)
    for data_base in DATABASES_DICT.keys():
        run_dbtraining(data_base)

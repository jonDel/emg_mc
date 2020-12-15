#!/usr/bin/env python
import gzip
import json
from glob import glob
import os
import numpy as np
import tensorflow as tf
from deepconvlstm import DeepConvLstm
import scripts.datasets_download as dl


np.random.seed(1)


def run_training_iterations(subject, database, epochs, batch_size,
                            subsamp_rate, early_patience, moves,
                            n_iterations, must_down=True, must_remove=False):
    iteration_list = []
    training_time_list = []
    test_accuracy_list = []
    peak_mem_list = []
    if must_down:
        dl.download_subject(subject, int(database[-1]))
    dcl = DeepConvLstm(subject, database, epochs=epochs, learn_rate=0.001, verbose=0,
                       timeback_reach=0.25, batch_size=batch_size, subsamp_rate=subsamp_rate,
                       moves=moves)
    for n_iter in range(n_iterations):
        if must_remove:
            os.system("rm -r results/logs/db{}/*".format(database[-1]))
            os.system("rm -r results/weights/db{}/*".format(database[-1]))
        training_time, test_accuracy, profile_summary, conf_matrix = dcl.run_training(early_patience=early_patience)
        iteration_list.append({
            "training_time": training_time,
            "test_accuracy": test_accuracy,
            "memory_profile_summary": profile_summary
        })
        training_time_list.append(training_time)
        test_accuracy_list.append(test_accuracy)
        try:
            peak_mem_list.append(int(profile_summary['peakStats']['peakBytesInUse']))
        except Exception as error:
            print(error)
        print("Iteration {} from {} completed.".format(n_iter + 1, n_iterations))
    stats = {
        "training_time": {
            "mean": np.mean(training_time_list),
            "std": np.std(training_time_list)
        },
        "test_accuracy": {
            "mean": np.mean(test_accuracy_list),
            "std": np.std(test_accuracy_list)
        },
        "peak_memory": {
            "mean": np.mean(peak_mem_list),
            "std": np.std(peak_mem_list)
        }
    }
    return iteration_list, stats, conf_matrix

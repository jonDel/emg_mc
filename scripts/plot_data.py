"""Simple example of how to plot the results saved in history."""

import matplotlib.pyplot as plt
import pickle
import sys
import os

CWD = os.getcwd()
HIST_PATH = CWD + "/results/history/db1/ts200/"


def plot_result(subject, path):
    """
    Plot the training/validation curves for a subject.

    It supposes pickle history files written in a given folder.

    Parameters:
        subject (:obj:`int`): subject number
        path (:obj:`str`): path to the history folder

    """
    with open(path+'subject:{}_history.pickle'.format(subject), 'rb') as hfile:
        data = pickle.load(hfile)
    plt.plot(data['acc'])
    plt.plot(data['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    USAGE = """Plot database_1 training/validation resulting curves for ts = 200
        python {script} <subject number>
        Ex: python {script} 10
    """
    if len(sys.argv) < 2:
        print(USAGE.format(script=sys.argv[0]))
    else:
        plot_result(sys.argv[1], HIST_PATH)
